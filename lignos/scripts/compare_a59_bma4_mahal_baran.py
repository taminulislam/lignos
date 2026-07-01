"""Baran Task 2 leave-IL-out CV — LIGNOS + Pick #1 (Baran as 4th BMA pillar)
and Pick #3 (Mahalanobis OOD gate → Baran fallback).

Fork of `compare_a59_tier2_vs_baran.py` that targets the paper's Task-2 OOD
failure mode (A5.9 ensemble R² = -0.10 ± 0.90; Tier-2 Stage-2 = -0.69 ± 1.40;
Baran GB = +0.52 ± 0.20). Two orthogonal fixes are implemented here:

  Pick #1 (Baran-as-BMA-pillar)
    Add Baran GB as a fourth pillar k=D in a scalar BMA router operating over
    the frozen outputs of GraphSpec/SurfSpec/SigmaSpec (pillars A/B/C). Each
    pillar emits a (mu, log σ²) pair per property; Baran emits the GB mean
    and a *constant* per-property lv = log(training OOB residual variance).
    Router softmax(-lv + scalar_bias) degenerates to "pick Baran" whenever
    the A/B/C pillars become simultaneously uncertain (LIGNOS OOD).
    32 trainable params (K=4, P=8).

  Pick #3 (Mahalanobis OOD gate)
    Fit a Gaussian on Morgan PCA-40 features of training rows; compute
    Mahalanobis distance for each test row. Rows with d_M > q_0.9(train d_M)
    are flagged as OOD and their prediction is OVERRIDDEN with the Baran
    GB prediction directly (bypassing the BMA router). This catches the
    fold-3 failure mode: A/B/C are low-variance but *confidently wrong*, so
    the inverse-variance anchor doesn't down-weight them. Feature-space
    distance catches novelty that the logvar heads miss.

Reports three headline numbers per fold:

  (1) 3-pillar ensemble mean     — baseline A5.9 (same as prior script)
  (2) 4-pillar BMA (Pick #1)     — adds Baran into the router softmax
  (3) 4-pillar BMA + Mahal gate  — (1) + force-route OOD test rows to Baran

Plus Baran-alone R² computed on the same per-fold features for reference.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, set_seed  # noqa
from train_a2_two_stage import build_chemprop_40d, preprocess_physchem  # noqa
from train_a5_bma_pipeline import (
    A5_BMA_Specialist, train_specialist, _assemble_bank, _standardize,  # noqa
    FRAME_DIM, COSMO_DIM, VIT_BANK, COSMO_BANK, LV_CLAMP,
)
from compare_a2_vs_baran import _load_baran_matched
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

RESULTS = V5 / "results"
IDX_LIGNIN = 7


# ==========================================================================
# Helpers
# ==========================================================================
def _gated(pred, y, total_sigma, quantile):
    if len(pred) == 0:
        return None
    thr = np.quantile(total_sigma, quantile)
    keep = total_sigma <= thr
    if keep.sum() < 2:
        return None
    yk, pk = y[keep], pred[keep]
    return {"r2": float(r2_score(yk, pk)),
            "mae": float(mean_absolute_error(yk, pk)),
            "n_keep": int(keep.sum()), "n_total": int(len(y))}


def _gauss_nll(mu, lv, y, valid):
    lv = lv.clamp(*LV_CLAMP)
    nll = 0.5 * torch.exp(-lv) * (mu - y) ** 2 + 0.5 * lv
    return (nll * valid.float()).sum() / valid.float().sum().clamp(min=1)


# ==========================================================================
# Pick #1: Baran GB as 4th BMA pillar
# ==========================================================================
def fit_baran_per_property(X_tr, y_tr, n_props=8, seed=42):
    """Fit one HistGradientBoostingRegressor per property on rows with valid labels.

    Returns list of (model, is_fitted, resid_var) per property.  Baran's
    per-property residual variance (on its own training set) serves as the
    constant logvar exposed to the BMA router."""
    scaler = StandardScaler().fit(X_tr)
    X_s = scaler.transform(X_tr)

    models, resid_vars = [], []
    for p in range(n_props):
        y_col = y_tr[:, p]
        mask = ~np.isnan(y_col)
        if mask.sum() < 10:
            # Too few training points for this property — fallback: constant predictor
            mu_const = float(np.nanmean(y_col)) if mask.sum() else 0.0
            var_const = float(np.nanvar(y_col)) if mask.sum() > 1 else 1.0
            models.append(("const", mu_const))
            resid_vars.append(max(var_const, 1e-4))
            continue
        gb = GradientBoostingRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=seed,
        )
        gb.fit(X_s[mask], y_col[mask])
        pred_tr = gb.predict(X_s[mask])
        resid = y_col[mask] - pred_tr
        models.append(("gb", gb))
        # Add a small floor so log is finite; var is on-training (optimistic),
        # scaled by a factor to temper overconfidence.
        resid_vars.append(max(float(resid.var()) * 1.5, 1e-4))
    return scaler, models, np.array(resid_vars, dtype=np.float32)


def baran_predict(scaler, models, X, n_props=8):
    """Returns (N, n_props) predictions from per-property GB models."""
    X_s = scaler.transform(X)
    out = np.zeros((X.shape[0], n_props), dtype=np.float32)
    for p, (kind, m) in enumerate(models):
        if kind == "gb":
            out[:, p] = m.predict(X_s).astype(np.float32)
        else:  # const
            out[:, p] = m
    return out


# ==========================================================================
# Scalar K=4 BMA router (local, operates on precomputed specialist outputs)
# ==========================================================================
class Scalar4PillarRouter(nn.Module):
    """Per-(K, P) scalar corrector on top of inverse-variance anchor.

    w_{k,p}(row) = softmax_k(-lv_{k,p}(row) + bias_{k,p})

    4 pillars × 8 properties = 32 parameters. Degenerates to "pick Baran"
    on OOD rows by pushing Baran's scalar bias up when the A/B/C pillars
    show high lv for those rows."""

    def __init__(self, K=4, P=8):
        super().__init__()
        self.K, self.P = K, P
        self.scalar_corr = nn.Parameter(torch.zeros(K, P))

    def forward(self, lv_stack):  # (B, K, P) -> (B, K, P)
        anchor = -lv_stack
        return F.softmax(anchor + self.scalar_corr.unsqueeze(0), dim=1)


def train_scalar_k4_router(mu_all, lv_all, y, device, epochs=100, patience=30):
    """Train scalar K=4 BMA router on precomputed pillar outputs.

    mu_all, lv_all: (N, K, P) numpy arrays over training rows
    y: (N, P) numpy with NaN for missing labels
    """
    mu_t = torch.from_numpy(mu_all).to(device)
    lv_t = torch.from_numpy(lv_all).to(device).clamp(*LV_CLAMP)
    y_t = torch.from_numpy(y).to(device)
    valid = ~torch.isnan(y_t)
    yf = torch.nan_to_num(y_t, 0.0)

    K, P = mu_t.shape[1], mu_t.shape[2]
    router = Scalar4PillarRouter(K=K, P=P).to(device)
    opt = AdamW(router.parameters(), lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    N = mu_t.shape[0]
    idx_all = torch.arange(N)
    ds = TensorDataset(idx_all, yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for ep in range(epochs):
        router.train()
        for batch_idx, yb, vm in loader:
            batch_idx = batch_idx.to(device); yb = yb.to(device); vm = vm.to(device)
            mu_b = mu_t[batch_idx]; lv_b = lv_t[batch_idx]
            w = router(lv_b)
            mu_fused = (w * mu_b).sum(dim=1)
            prec = torch.exp(-lv_b).sum(dim=1) + 1e-8
            lv_fused = -torch.log(prec)
            loss = _gauss_nll(mu_fused, lv_fused, yb, vm)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0); opt.step()
        sch.step()

        router.eval()
        with torch.no_grad():
            w_all = router(lv_t)
            mu_f = (w_all * mu_t).sum(dim=1)
            prec = torch.exp(-lv_t).sum(dim=1) + 1e-8
            lv_f = -torch.log(prec)
            tl = _gauss_nll(mu_f, lv_f, yf, valid).item()
        if np.isfinite(tl) and tl < best:
            best = tl
            state = {k: v.clone() for k, v in router.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if state is not None:
        router.load_state_dict(state)
    router.eval()
    return router


def fuse_k4(router, mu_all, lv_all):
    """Apply trained router to pillar outputs to return fused (mu, lv)."""
    mu_t = torch.from_numpy(mu_all)
    lv_t = torch.from_numpy(lv_all).clamp(*LV_CLAMP)
    with torch.no_grad():
        w = router(lv_t.to(next(router.parameters()).device))
        mu_t = mu_t.to(w.device)
        lv_t = lv_t.to(w.device)
        mu_f = (w * mu_t).sum(dim=1)
        prec = torch.exp(-lv_t).sum(dim=1) + 1e-8
        lv_f = -torch.log(prec)
    return mu_f.cpu().numpy(), lv_f.cpu().numpy(), w.cpu().numpy()


# ==========================================================================
# Pick #3: Mahalanobis OOD gate on Morgan PCA-40 features
# ==========================================================================
def fit_mahal_gate(X_tr, q=0.9, ridge=1e-4):
    """Gaussian on X_tr; returns (mean, cov_inv, threshold_d2, train_d2)."""
    mu = X_tr.mean(axis=0)
    diffs = X_tr - mu
    d = X_tr.shape[1]
    cov = (diffs.T @ diffs) / max(X_tr.shape[0] - 1, 1) + ridge * np.eye(d)
    cov_inv = np.linalg.pinv(cov)
    d2_tr = np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs)
    thr = float(np.quantile(d2_tr, q))
    return mu, cov_inv, thr, d2_tr


def mahal_d2(X, mu, cov_inv):
    diffs = X - mu
    return np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs)


# ==========================================================================
# Main fold loop
# ==========================================================================
def _specialist_pillar_outputs(specialists, feats, device, seed_agg="mean"):
    """Run all specialist seeds × kinds, aggregate to (N, K=3, P) mu and lv."""
    keys = ("v","i","t","cp","surf","vit","cos","hs","hv","hc")
    full = ("v4","morg","thermo","chemprop","surface","vit","cos",
             "has_surf","has_vit","has_cos")
    ts = {k: torch.from_numpy(feats[full_k]).to(device)
          for k, full_k in zip(keys, full)}
    mu_kind, lv_kind = [], []
    for kind in ("A", "B", "C"):
        seed_mus, seed_lvs = [], []
        for m in specialists[kind]:
            with torch.no_grad():
                mu, lv = m.forward_with_lv(
                    ts["v"], ts["i"], ts["t"], ts["cp"],
                    surface=ts["surf"], vit=ts["vit"], cos=ts["cos"],
                    has_surf=ts["hs"], has_vit=ts["hv"], has_cos=ts["hc"])
            seed_mus.append(mu.cpu().numpy())
            seed_lvs.append(lv.cpu().numpy())
        mu_kind.append(np.stack(seed_mus).mean(axis=0))
        # Aggregate seed logvars by log-mean-exp (matches BMA precision-merge)
        lv_arr = np.stack(seed_lvs)
        prec = np.exp(-lv_arr).mean(axis=0) + 1e-8
        lv_kind.append(-np.log(prec))
    mu_stack = np.stack(mu_kind, axis=1)  # (N, 3, P)
    lv_stack = np.stack(lv_kind, axis=1)
    return mu_stack.astype(np.float32), lv_stack.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-specialist-seeds", type=int, default=2)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--router-epochs", type=int, default=120)
    ap.add_argument("--mahal-q", type=float, default=0.9,
                    help="Train-set quantile for d_M threshold.")
    ap.add_argument("--fold", type=int, default=None,
                    help="Run only this fold index (for SLURM array).")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  specialist_seeds={args.n_specialist_seeds}  "
          f"mahal_q={args.mahal_q}")

    tr, va, te, lig_ils, baran_lig_mask = _load_baran_matched()

    np.random.seed(42)
    il_order = np.random.permutation(lig_ils)
    fold_size = max(1, len(il_order) // args.n_splits)
    folds = [il_order[i * fold_size : (i + 1) * fold_size if i < args.n_splits - 1 else None]
             for i in range(args.n_splits)]

    pool_smiles = np.concatenate([tr["smiles"], va["smiles"], te["smiles"]])
    pool_il = np.concatenate([tr["il_ids"], va["il_ids"], te["il_ids"]])
    pool_y = np.concatenate([tr["targets"], va["targets"], te["targets"]]).astype(np.float32)
    pool_th = np.concatenate([tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]]).astype(np.float32)
    pool_mg = np.concatenate([tr["morgan_fp"], va["morgan_fp"], te["morgan_fp"]]).astype(np.float32)
    pool_cp = np.concatenate([tr["chemprop_fp"], va["chemprop_fp"], te["chemprop_fp"]]).astype(np.float32)
    pool_pf = np.concatenate([tr["preds_fusion"], va["preds_fusion"], te["preds_fusion"]]).astype(np.float32)
    pool_pc = np.concatenate([tr["preds_chemprop"], va["preds_chemprop"], te["preds_chemprop"]]).astype(np.float32)
    pool_surf = np.concatenate([tr["surface_fp"], va["surface_fp"], te["surface_fp"]]).astype(np.float32)
    pool_ph = np.concatenate([tr["physchem_feat"], va["physchem_feat"], te["physchem_feat"]]).astype(np.float32)
    pool_hp = np.concatenate([tr["has_physchem"], va["has_physchem"], te["has_physchem"]]).astype(np.float32)
    pool_v4 = (0.4 * pool_pf + 0.6 * pool_pc).astype(np.float32)

    vit_bank = dict(zip(*[np.load(VIT_BANK, allow_pickle=True)[k]
                            for k in ("smiles", "vit_feat")]))
    cos_bank = dict(zip(*[np.load(COSMO_BANK, allow_pickle=True)[k]
                            for k in ("smiles", "cosmo_feat")]))
    pool_vit, pool_hv = _assemble_bank(pool_smiles, vit_bank, FRAME_DIM)
    pool_cos, pool_hc = _assemble_bank(pool_smiles, cos_bank, COSMO_DIM)
    pool_hs = (pool_surf != 0).any(axis=1).astype(np.float32)

    fold_iter = [(args.fold, folds[args.fold])] if args.fold is not None \
                else list(enumerate(folds))

    fold_results = []
    for k, held in fold_iter:
        te_mask = np.isin(pool_il, held) & baran_lig_mask
        tr_mask = ~np.isin(pool_il, held)
        if te_mask.sum() == 0:
            print(f"\n=== Fold {k}: 0 test rows — skip ===")
            continue

        held_names = sorted(set(pool_il[te_mask]))
        print(f"\n=== Fold {k} ({te_mask.sum()} test rows, held={held_names[:2]}...) ===")

        pca = PCA(40).fit(pool_mg[tr_mask])
        f_tr = pca.transform(pool_mg[tr_mask]).astype(np.float32)
        f_te = pca.transform(pool_mg[te_mask]).astype(np.float32)
        cp_tr_z, cp_te_z = build_chemprop_40d(pool_cp[tr_mask], pool_cp[te_mask])

        surf_tr_z, mu_p, sd_p = _standardize(pool_surf[tr_mask], pool_hs[tr_mask])
        surf_te_z = ((pool_surf[te_mask] - mu_p) / sd_p).astype(np.float32) * pool_hs[te_mask][:, None]
        vit_tr_z, mu_v, sd_v = _standardize(pool_vit[tr_mask], pool_hv[tr_mask])
        vit_te_z = ((pool_vit[te_mask] - mu_v) / sd_v).astype(np.float32) * pool_hv[te_mask][:, None]
        cos_tr_z, mu_c, sd_c = _standardize(pool_cos[tr_mask], pool_hc[tr_mask])
        cos_te_z = ((pool_cos[te_mask] - mu_c) / sd_c).astype(np.float32) * pool_hc[te_mask][:, None]

        phys_tr_z, phys_te_z = preprocess_physchem(
            pool_ph[tr_mask], pool_hp[tr_mask],
            pool_ph[te_mask], pool_hp[te_mask])

        feats_tr = {"v4": pool_v4[tr_mask], "morg": f_tr,
                    "thermo": pool_th[tr_mask], "chemprop": cp_tr_z,
                    "surface": surf_tr_z, "vit": vit_tr_z, "cos": cos_tr_z,
                    "has_surf": pool_hs[tr_mask], "has_vit": pool_hv[tr_mask],
                    "has_cos": pool_hc[tr_mask],
                    "physchem": phys_tr_z, "has_physchem": pool_hp[tr_mask]}
        feats_te = {"v4": pool_v4[te_mask], "morg": f_te,
                    "thermo": pool_th[te_mask], "chemprop": cp_te_z,
                    "surface": surf_te_z, "vit": vit_te_z, "cos": cos_te_z,
                    "has_surf": pool_hs[te_mask], "has_vit": pool_hv[te_mask],
                    "has_cos": pool_hc[te_mask],
                    "physchem": phys_te_z, "has_physchem": pool_hp[te_mask]}
        y_tr = pool_y[tr_mask]; y_te = pool_y[te_mask]

        # ---- Train 3 specialists × n_seeds ----
        specialists = {"A": [], "B": [], "C": []}
        for kind in ("A", "B", "C"):
            for seed in range(args.n_specialist_seeds):
                m_k, _ = train_specialist(kind, seed, feats_tr, y_tr, device,
                                           epochs=args.epochs, patience=40)
                m_k.eval()
                specialists[kind].append(m_k)

        # ---- Precompute pillar outputs (A, B, C) on train and test ----
        mu_abc_tr, lv_abc_tr = _specialist_pillar_outputs(specialists, feats_tr, device)
        mu_abc_te, lv_abc_te = _specialist_pillar_outputs(specialists, feats_te, device)

        # Baseline A5.9 ensemble-mean (for comparison): simple mean over A,B,C pillars
        pred_ens_tr = mu_abc_tr.mean(axis=1)  # (N, P)
        pred_ens_te = mu_abc_te.mean(axis=1)
        aleatoric_te = np.exp(lv_abc_te).mean(axis=1)
        epistemic_te = mu_abc_te.var(axis=1)
        total_sigma_row = np.sqrt(aleatoric_te + epistemic_te).mean(axis=-1)

        y_true = y_te[:, IDX_LIGNIN]
        pred_ens_lig = pred_ens_te[:, IDX_LIGNIN]
        r2_ens_all = float(r2_score(y_true, pred_ens_lig))
        ens_gated = {q: _gated(pred_ens_lig, y_true, total_sigma_row, q)
                      for q in (0.25, 0.5, 0.75)}

        # ---- Pick #1: Baran GB per property + 4th pillar ----
        X_baran_tr = np.column_stack([
            f_tr, pool_th[tr_mask], phys_tr_z,
            pool_hp[tr_mask].astype(np.float32).reshape(-1, 1),
        ])
        X_baran_te = np.column_stack([
            f_te, pool_th[te_mask], phys_te_z,
            pool_hp[te_mask].astype(np.float32).reshape(-1, 1),
        ])
        baran_scaler, baran_models, baran_resid_var = fit_baran_per_property(
            X_baran_tr, y_tr, n_props=8, seed=42)
        mu_baran_tr = baran_predict(baran_scaler, baran_models, X_baran_tr)
        mu_baran_te = baran_predict(baran_scaler, baran_models, X_baran_te)
        lv_baran = np.log(baran_resid_var).astype(np.float32)  # (P,)
        lv_baran_tr = np.broadcast_to(lv_baran, mu_baran_tr.shape).astype(np.float32)
        lv_baran_te = np.broadcast_to(lv_baran, mu_baran_te.shape).astype(np.float32)

        # Baran-alone lignin R² on test fold (reference)
        r2_baran_alone = float(r2_score(y_true, mu_baran_te[:, IDX_LIGNIN]))
        mae_baran_alone = float(mean_absolute_error(y_true, mu_baran_te[:, IDX_LIGNIN]))

        # 4-pillar stacks
        mu_all_tr = np.concatenate(
            [mu_abc_tr, mu_baran_tr[:, None, :]], axis=1).astype(np.float32)
        lv_all_tr = np.concatenate(
            [lv_abc_tr, lv_baran_tr[:, None, :]], axis=1).astype(np.float32)
        mu_all_te = np.concatenate(
            [mu_abc_te, mu_baran_te[:, None, :]], axis=1).astype(np.float32)
        lv_all_te = np.concatenate(
            [lv_abc_te, lv_baran_te[:, None, :]], axis=1).astype(np.float32)

        # ---- Train K=4 scalar router ----
        router_k4 = train_scalar_k4_router(
            mu_all_tr, lv_all_tr, y_tr, device,
            epochs=args.router_epochs, patience=30)
        mu_k4_te, lv_k4_te, w_k4_te = fuse_k4(router_k4, mu_all_te, lv_all_te)
        pred_k4_lig = mu_k4_te[:, IDX_LIGNIN]
        r2_k4_all = float(r2_score(y_true, pred_k4_lig))
        # Use new K=4 fused lv for the gating metric (and expose it for diagnostics)
        total_sigma_k4 = np.sqrt(np.exp(lv_k4_te) + epistemic_te).mean(axis=-1)
        k4_gated = {q: _gated(pred_k4_lig, y_true, total_sigma_k4, q)
                     for q in (0.25, 0.5, 0.75)}
        w_baran_mean_lig = float(w_k4_te[:, 3, IDX_LIGNIN].mean())

        # ---- Pick #3: Mahalanobis OOD gate on Morgan PCA-40 ----
        mu_m, cov_inv_m, thr_m, d2_tr_m = fit_mahal_gate(f_tr, q=args.mahal_q)
        d2_te_m = mahal_d2(f_te, mu_m, cov_inv_m)
        ood_mask = d2_te_m > thr_m
        pred_gated = pred_k4_lig.copy()
        pred_gated[ood_mask] = mu_baran_te[ood_mask, IDX_LIGNIN]
        r2_gated_all = float(r2_score(y_true, pred_gated))
        n_ood = int(ood_mask.sum())

        def _fmt(d, q):
            return f"{d[q]['r2']:+.4f}" if d[q] else "n/a"
        print(f"  A5.9 ens       : R² = {r2_ens_all:+.4f}  g50={_fmt(ens_gated, 0.5)}")
        print(f"  + Baran pillar : R² = {r2_k4_all:+.4f}  g50={_fmt(k4_gated, 0.5)}  "
              f"mean w_D(lig)={w_baran_mean_lig:.2f}")
        print(f"  + Mahal gate   : R² = {r2_gated_all:+.4f}  "
              f"({n_ood}/{len(y_true)} test rows overridden; thr d²={thr_m:.2f})")
        print(f"  Baran alone    : R² = {r2_baran_alone:+.4f}   MAE = {mae_baran_alone:.3f}")

        fold_results.append({
            "fold": k,
            "n": int(te_mask.sum()),
            "held_out_ils": held_names,
            "ensemble": {
                "r2_all": r2_ens_all,
                "r2_gated": {str(q): ens_gated[q] for q in (0.25, 0.5, 0.75)},
            },
            "k4_bma": {
                "r2_all": r2_k4_all,
                "r2_gated": {str(q): k4_gated[q] for q in (0.25, 0.5, 0.75)},
                "mean_w_baran_lignin": w_baran_mean_lig,
            },
            "k4_mahal": {
                "r2_all": r2_gated_all,
                "n_ood": n_ood,
                "n_total": int(len(y_true)),
                "threshold_d2": float(thr_m),
            },
            "baran_alone": {
                "r2": r2_baran_alone,
                "mae": mae_baran_alone,
            },
            "aleatoric_lignin_mean": float(aleatoric_te[:, IDX_LIGNIN].mean()),
            "epistemic_lignin_mean": float(epistemic_te[:, IDX_LIGNIN].mean()),
        })

    # Per-fold mode: write one JSON per fold.
    if args.fold is not None:
        out = RESULTS / f"lignos_bma4_mahal_fold_{args.fold}.json"
        json.dump({
            "fold": args.fold,
            "result": fold_results[0] if fold_results else None,
            "n_specialist_seeds": args.n_specialist_seeds,
            "mahal_q": args.mahal_q,
        }, open(out, "w"), indent=2)
        print(f"\nSaved fold {args.fold}: {out}")
        return

    def _agg(key):
        xs = [f[key]["r2_all"] for f in fold_results]
        return float(np.mean(xs)), float(np.std(xs))
    ens_m, ens_s = _agg("ensemble")
    k4_m, k4_s = _agg("k4_bma")
    mg_m, mg_s = _agg("k4_mahal")
    ba_m, ba_s = (
        float(np.mean([f["baran_alone"]["r2"] for f in fold_results])),
        float(np.std([f["baran_alone"]["r2"] for f in fold_results])),
    )

    print(f"\n{'='*70}\nBaran Task 2 — LIGNOS + BMA-K4 + Mahalanobis gate\n{'='*70}")
    print(f"A5.9 ensemble ALL        : R² = {ens_m:+.4f} ± {ens_s:.4f}")
    print(f"+ Baran BMA pillar ALL   : R² = {k4_m:+.4f} ± {k4_s:.4f}  (Δ={k4_m-ens_m:+.4f})")
    print(f"+ Mahalanobis gate ALL   : R² = {mg_m:+.4f} ± {mg_s:.4f}  (Δ={mg_m-ens_m:+.4f})")
    print(f"Baran alone (this run)   : R² = {ba_m:+.4f} ± {ba_s:.4f}")
    print(f"Baran GB (published)     : R² = +0.5238 ± 0.2015")

    out = RESULTS / "lignos_bma4_mahal_task2.json"
    json.dump({
        "folds": fold_results,
        "ensemble_r2_mean": ens_m, "ensemble_r2_std": ens_s,
        "k4_bma_r2_mean": k4_m, "k4_bma_r2_std": k4_s,
        "k4_mahal_r2_mean": mg_m, "k4_mahal_r2_std": mg_s,
        "baran_alone_r2_mean": ba_m, "baran_alone_r2_std": ba_s,
        "n_specialist_seeds": args.n_specialist_seeds,
        "mahal_q": args.mahal_q,
        "n_splits": args.n_splits,
    }, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
