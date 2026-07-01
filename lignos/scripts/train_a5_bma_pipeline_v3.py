"""A5.9 Pipeline v3 — corrections to v2's failed arms.

v2 ablation found:
  Arm A (extended thermo[:25] in deep lignin head): lignin regressed 0.70 → 0.23
    → Root cause: 20 extra mostly-zero dims + only ~140 Baran training rows
       → deep head (18k params) overfits / memorizes
  Arm B (GB as Specialist D with fixed logvar=-1): core7 regressed 0.834 → 0.804
    → Root cause: GB predicts RAW y, not residual against v4_base like neural
       specialists; scale mismatch + high precision (logvar=-1) over-weights GB
       in fusion → drags ensemble mean toward GB's weaker per-target accuracy

v3 corrections:
  A' — --shrink-lignin-head: deep head shrinks to hidden=32 (from 128+64),
        dropout 0.3, wd=1e-1. Prevents overfitting when extended thermo used.
  B' — --gb-residual + --gb-high-logvar: GB trains on (y - v4_base) residuals,
        matching neural specialists' output scale. Fixed logvar=+2 (σ²≈7)
        downweights GB to a corrective voice rather than a dominant one.
  AB' — both corrections combined.

Run with --arm in {A_prime, B_prime, AB_prime}.
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, r2_per_prop, set_seed  # noqa
from train_a2_two_stage import (
    build_chemprop_40d, preprocess_physchem, v4_base,
)
from train_a5_bma_pipeline import (
    A5_BMA_Specialist, A5_BMA_Router, train_router,
    _assemble_bank, _standardize, _load_split,
    FRAME_DIM, SURFACE_DIM, COSMO_DIM,
    VIT_BANK, COSMO_BANK, BMA_DIR,
)

CACHE = V5 / "data" / "LignoIL_A1"


class GBSpecialist_v3:
    """GB specialist that trains on residuals (y - v4_base) — same scale as
    neural specialists. Fixed logvar (tunable) controls its weight in fusion."""

    def __init__(self, n_props=8, fixed_logvar=+2.0, residual_target=True):
        self.n_props = n_props
        self.gbs = [None] * n_props
        self.fixed_logvar = fixed_logvar
        self.residual_target = residual_target

    def fit(self, X, y, v4_base_tr):
        """y: (N, n_props) with NaN; v4_base_tr: (N, n_props) teacher preds.
        Train each GB on residuals (y - v4_base) if residual_target=True."""
        target = y - v4_base_tr if self.residual_target else y
        for p in range(self.n_props):
            mask = ~np.isnan(y[:, p])
            if mask.sum() < 10:
                continue
            self.gbs[p] = GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            self.gbs[p].fit(X[mask], target[mask, p])

    def predict(self, X, v4_base):
        """Returns reconstructed μ = v4_base + GB residual (or raw GB)."""
        pred_resid = np.zeros((len(X), self.n_props), dtype=np.float32)
        for p in range(self.n_props):
            if self.gbs[p] is not None:
                pred_resid[:, p] = self.gbs[p].predict(X)
        if self.residual_target:
            return v4_base + pred_resid   # reconstruct absolute prediction
        return pred_resid

    def forward_with_lv(self, v, i, t, chemprop, surface=None, vit=None,
                         cos=None, has_surf=None, has_vit=None, has_cos=None):
        """v is v4_base per forward (it's the first arg across all specialists)."""
        X = torch.cat([i, t, chemprop], dim=-1).detach().cpu().numpy()
        v4 = v.detach().cpu().numpy()
        mu = torch.from_numpy(self.predict(X, v4)).to(i.device)
        lv = torch.full_like(mu, self.fixed_logvar)
        return mu, lv

    def eval(self): return self
    def train(self, mode=True): return self


class A5_BMA_Stage2_v3(nn.Module):
    """Stage-2 wrapper with configurable deep-lignin-head capacity."""
    def __init__(self, specialists_list, router, nf, n_props=8,
                 physchem_dim=12, extended_thermo=False, shrink_head=False):
        super().__init__()
        self.specialists = specialists_list
        self.router = router
        self.extended_thermo = extended_thermo

        for m in specialists_list:
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad = False
        if isinstance(router, nn.Module):
            for p in router.parameters():
                p.requires_grad = False

        first_neural = next(m for m in specialists_list if isinstance(m, nn.Module))
        self.gate_fn = first_neural.gate
        self.alpha_lignin = nn.Parameter(first_neural.alphas.data[7].clone())

        therm_dim = 25 if extended_thermo else 5
        head_in = nf + therm_dim + physchem_dim + 1

        if shrink_head:
            # Much smaller: ~3k params (vs ~18k). Heavier regularization.
            self.deep_lignin = nn.Sequential(
                nn.Linear(head_in, 32), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(32, 1),
            )
        else:
            self.deep_lignin = nn.Sequential(
                nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(64, 1),
            )
        with torch.no_grad():
            self.deep_lignin[-1].weight.mul_(0.01); self.deep_lignin[-1].bias.zero_()

    def _fused(self, v, i, t, chemprop, surface, vit, cos, hs, hv, hc):
        mus, lvs = [], []
        with torch.no_grad():
            for m in self.specialists:
                mu, lv = m.forward_with_lv(
                    v, i, t, chemprop,
                    surface=surface, vit=vit, cos=cos,
                    has_surf=hs, has_vit=hv, has_cos=hc)
                mus.append(mu); lvs.append(lv)
            mu_s = torch.stack(mus, dim=1)
            lv_s = torch.stack(lvs, dim=1)
            w = self.router(chemprop, surface, t, lv_s)
            mu_f = (w * mu_s).sum(dim=1)
            prec = torch.exp(-lv_s).sum(dim=1) + 1e-8
            lv_f = -torch.log(prec)
        return mu_f, lv_f

    def forward(self, v, i, t, chemprop, surface, vit, cos, hs, hv, hc,
                phys, has_phys):
        mu_f, lv_f = self._fused(v, i, t, chemprop, surface, vit, cos, hs, hv, hc)
        tmp = t[:, :25] if self.extended_thermo else t[:, :5]
        g = i * self.gate_fn(t[:, :5])
        hp = has_phys.float().unsqueeze(-1) if has_phys.ndim == 1 else has_phys.float()
        ctx = torch.cat([g, tmp, phys, hp], -1)
        res_lignin = self.deep_lignin(ctx).squeeze(-1)
        out = mu_f.clone()
        out[:, 7] = v[:, 7] + torch.sigmoid(self.alpha_lignin) * res_lignin
        return out, lv_f


def train_stage2(model, feats_tr, y_tr, device, seed, epochs=300, patience=50,
                  wd=1e-2):
    set_seed(seed)
    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW([{"params": model.deep_lignin.parameters(), "weight_decay": wd},
                  {"params": [model.alpha_lignin], "weight_decay": 0.0}], lr=1e-3)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    keys = ("v","i","t","cp","surf","vit","cos","hs","hv","hc","p","hp")
    ts = {k: torch.from_numpy(feats_tr[full]).to(device) for k, full in zip(keys,
           ("v4","morg","thermo","chemprop","surface","vit","cos",
            "has_surf","has_vit","has_cos","physchem","has_physchem"))}
    y_t = torch.from_numpy(y_tr).to(device)
    ds = TensorDataset(*[ts[k].cpu() for k in keys], y_t.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            *inputs, yb = batch
            pred, _ = model(*inputs)
            lg = ~torch.isnan(yb[:, 7])
            if lg.sum() == 0: continue
            loss = ((pred[lg, 7] - yb[lg, 7].nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            pred, _ = model(*[ts[k] for k in keys])
            lg = ~torch.isnan(y_t[:, 7])
            tl = ((pred[lg, 7] - y_t[lg, 7].nan_to_num(0)) ** 2).mean().item() if lg.any() else float("inf")
        if np.isfinite(tl) and tl < best:
            best = tl; state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: model.load_state_dict(state)
    model.eval()
    return model


def conf_gated_r2(pred, y, logvar, quantile=0.5):
    sigma = np.exp(0.5 * logvar).mean(axis=-1)
    thr = np.quantile(sigma, quantile)
    keep = sigma <= thr
    r = {}
    for i, p in enumerate(PROPS):
        valid = ~np.isnan(y[:, i]) & keep
        if valid.sum() < 2: continue
        yk = y[valid, i]; pk = pred[valid, i]
        ss_res = ((yk - pk) ** 2).sum()
        ss_tot = ((yk - yk.mean()) ** 2).sum() + 1e-12
        r[p] = 1.0 - ss_res / ss_tot
    r["avg_core7"] = float(np.mean([r[p] for p in PROPS[:7] if p in r and np.isfinite(r[p])]))
    return r


def summarize(name, r2s):
    c = [r["avg_core7"] for r in r2s]
    out = {"name": name, "avg_r2_core7": float(np.mean(c)),
           "std_r2_core7": float(np.std(c)), "per_prop": {}}
    for p in PROPS:
        vs = [r.get(p) for r in r2s if r.get(p) is not None and not np.isnan(r.get(p, float("nan")))]
        out["per_prop"][p] = float(np.mean(vs)) if vs else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A_prime", "B_prime", "AB_prime"], required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    extended_thermo = args.arm in ("A_prime", "AB_prime")
    use_gb = args.arm in ("B_prime", "AB_prime")
    shrink_head = extended_thermo  # shrink only when extended_thermo on

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  arm: {args.arm}")
    print(f"  extended_thermo={extended_thermo}  shrink_head={shrink_head}  use_gb={use_gb}")

    tr, va, te = _load_split("train"), _load_split("val"), _load_split("test")
    pca_m = PCA(40).fit(tr["morgan_fp"])
    m_tr, m_va, m_te = [pca_m.transform(x["morgan_fp"]).astype(np.float32) for x in (tr, va, te)]
    cp_tr, cp_te = build_chemprop_40d(tr["chemprop_fp"], te["chemprop_fp"])
    _, cp_va = build_chemprop_40d(tr["chemprop_fp"], va["chemprop_fp"])
    p_tr, p_te = preprocess_physchem(tr["physchem_feat"], tr["has_physchem"],
                                      te["physchem_feat"], te["has_physchem"])
    hp_tr = tr["has_physchem"].astype(np.float32)
    hp_te = te["has_physchem"].astype(np.float32)

    surf_tr = tr["surface_fp"].astype(np.float32); hs_tr = (surf_tr != 0).any(axis=1).astype(np.float32)
    surf_te = te["surface_fp"].astype(np.float32); hs_te = (surf_te != 0).any(axis=1).astype(np.float32)
    surf_tr, mu_p, sd_p = _standardize(surf_tr, hs_tr)
    surf_te = ((surf_te - mu_p) / sd_p).astype(np.float32) * hs_te[:, None]

    vit_bank = dict(zip(*[np.load(VIT_BANK, allow_pickle=True)[k] for k in ("smiles", "vit_feat")]))
    vit_tr, hv_tr = _assemble_bank(tr["smiles"], vit_bank, FRAME_DIM)
    vit_te, hv_te = _assemble_bank(te["smiles"], vit_bank, FRAME_DIM)
    vit_tr, mu_v, sd_v = _standardize(vit_tr, hv_tr)
    vit_te = ((vit_te - mu_v) / sd_v).astype(np.float32) * hv_te[:, None]

    cos_bank = dict(zip(*[np.load(COSMO_BANK, allow_pickle=True)[k] for k in ("smiles", "cosmo_feat")]))
    cos_tr, hc_tr = _assemble_bank(tr["smiles"], cos_bank, COSMO_DIM)
    cos_te, hc_te = _assemble_bank(te["smiles"], cos_bank, COSMO_DIM)
    cos_tr, mu_c, sd_c = _standardize(cos_tr, hc_tr)
    cos_te = ((cos_te - mu_c) / sd_c).astype(np.float32) * hc_te[:, None]

    v4_tr, v4_te = v4_base(tr), v4_base(te)
    y_tr, y_te = tr["targets"].astype(np.float32), te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]

    feats_tr = {"v4": v4_tr, "morg": m_tr, "thermo": th_tr, "chemprop": cp_tr,
                 "surface": surf_tr, "vit": vit_tr, "cos": cos_tr,
                 "has_surf": hs_tr, "has_vit": hv_tr, "has_cos": hc_tr,
                 "physchem": p_tr, "has_physchem": hp_tr}
    feats_te = {"v4": v4_te, "morg": m_te, "thermo": th_te, "chemprop": cp_te,
                 "surface": surf_te, "vit": vit_te, "cos": cos_te,
                 "has_surf": hs_te, "has_vit": hv_te, "has_cos": hc_te,
                 "physchem": p_te, "has_physchem": hp_te}

    # Load neural specialists
    neural = []
    for kind in ("A", "B", "C"):
        ck = torch.load(BMA_DIR / f"specialist_{kind}.pt",
                         map_location=device, weights_only=False)
        m = A5_BMA_Specialist(kind, m_tr.shape[1], y_tr.shape[1],
                                 chemprop_dim=cp_tr.shape[1]).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        neural.append(m)
        print(f"  [Sp {kind}] core7={ck.get('test_core7'):.4f}")

    # GB Specialist D with v3 corrections
    if use_gb:
        print("\nTraining GB Specialist D (v3: residual target, logvar=+2)...")
        gb = GBSpecialist_v3(n_props=y_tr.shape[1],
                              fixed_logvar=+2.0, residual_target=True)
        X_tr_gb = np.concatenate([m_tr, th_tr, cp_tr], axis=1)
        gb.fit(X_tr_gb, y_tr, v4_tr)
        neural.append(gb)
        print(f"  [Sp D] GB trained on residuals y - v4_base, fixed logvar=+2 (σ²≈7)")

    # Train router
    print(f"\nTraining scalar router on {len(neural)} specialists...")
    set_seed(2026)
    router = train_router(neural, feats_tr, y_tr, device, epochs=args.epochs // 2, mode="scalar")
    router.eval()

    # Stage-2
    nf = m_tr.shape[1]
    s2_r2s, s2_gated50 = [], []
    wd = 1e-1 if shrink_head else 1e-2
    for seed in range(args.n_seeds):
        print(f"\n[Stage-2 arm={args.arm}] seed {seed} ...")
        model = A5_BMA_Stage2_v3(neural, router, nf, y_tr.shape[1],
                                    extended_thermo=extended_thermo,
                                    shrink_head=shrink_head).to(device)
        model = train_stage2(model, feats_tr, y_tr, device, seed=seed,
                              epochs=args.epochs, wd=wd)
        with torch.no_grad():
            keys = ("v","i","t","cp","surf","vit","cos","hs","hv","hc","p","hp")
            ins = [torch.from_numpy(feats_te[full]).to(device) for full in
                    ("v4","morg","thermo","chemprop","surface","vit","cos",
                     "has_surf","has_vit","has_cos","physchem","has_physchem")]
            pred, lv = model(*ins)
            pred_te, lv_te = pred.cpu().numpy(), lv.cpu().numpy()
        r = r2_per_prop(pred_te, y_te)
        g50 = conf_gated_r2(pred_te, y_te, lv_te, quantile=0.5)
        s2_r2s.append(r); s2_gated50.append(g50)
        print(f"  core7={r['avg_core7']:.4f}  lignin={r.get('lignin_wt', float('nan')):.4f}  "
              f"gated@50%={g50['avg_core7']:.4f}")

    s = summarize(f"Stage2_A5_BMA_v3_{args.arm}", s2_r2s)
    s50 = summarize(f"Stage2_A5_BMA_v3_{args.arm}_gated50", s2_gated50)
    print(f"\n{'='*72}\nA5.9 v3 Stage-2 Summary — arm={args.arm}\n{'='*72}")
    print(f"{'Arm':<42}{'core7':>10}{'std':>10}{'lignin':>10}")
    for r in [s, s50]:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<42}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}{lig:>10.4f}")

    outf = V5 / "results" / f"a5_bma_v3_{args.arm}.json"
    json.dump([s, s50], open(outf, "w"), indent=2)
    print(f"\nSaved: {outf}")


if __name__ == "__main__":
    main()
