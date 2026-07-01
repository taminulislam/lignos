"""Baran Task 2 leave-IL-out CV — adds Baran-as-feature lignin head (idea #3)
and dumps per-row predictions for the meta-stacker (idea #1).

Sibling of `compare_a59_bma4_mahal_baran.py`. Reproduces all four headline
numbers (A5.9 ensemble, K4 BMA, K4+Mahal, Baran alone) on identical folds,
and adds:

  (4) Baran-as-feature lignin head (Pick #3-replacement, in-fold)
      A linear ridge on the lignin column with 8 inputs:
        [mu_A_lig, mu_B_lig, mu_C_lig, lv_A_lig, lv_B_lig, lv_C_lig,
         mu_baran_lig, lv_baran_lig]
      Asymmetric L2: lam_normal on the 6 specialist features, lam_baran on
      the 2 Baran features. Trained on the same training rows as the
      specialists (so it WILL overfit specialist outputs); reported for
      reference. The leakage-safe version lives in fit_meta_stacker.py.

  (5) Per-row CSV dump
      One row per test sample with predictions, uncertainties, Mahalanobis
      d², Tanimoto-NN-to-train, and IL name. Drives the leave-one-fold-out
      meta-stacker (idea #1) as a post-processor.
"""
from __future__ import annotations
import argparse, csv, json, sys
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
from compare_a59_bma4_mahal_baran import (
    _gated, _gauss_nll,
    fit_baran_per_property, baran_predict,
    Scalar4PillarRouter, train_scalar_k4_router, fuse_k4,
    fit_mahal_gate, mahal_d2,
    _specialist_pillar_outputs,
)
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error

RESULTS = V5 / "results"
IDX_LIGNIN = 7


# ==========================================================================
# Idea #3 (in-fold variant): Baran-as-feature lignin head with asymmetric L2
# ==========================================================================
def fit_asymmetric_ridge(X, y, lam_per_col):
    """Ridge with diagonal L2 (one penalty per input column).

    X : (N, D) float
    y : (N,)   float (NaNs dropped)
    lam_per_col : (D,) float, per-column ridge penalty (after standardization)

    Returns dict with weights, intercept, and standardization stats.
    """
    mask = ~np.isnan(y)
    X = X[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    if len(y) < 4:
        return None
    mu_x = X.mean(0)
    sd_x = X.std(0) + 1e-8
    Xs = (X - mu_x) / sd_x
    mu_y = float(y.mean())
    yc = y - mu_y
    Lam = np.diag(lam_per_col)
    XtX = Xs.T @ Xs
    Xty = Xs.T @ yc
    try:
        w = np.linalg.solve(XtX + Lam, Xty)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(XtX + Lam, Xty, rcond=None)[0]
    return {"w": w, "mu_x": mu_x, "sd_x": sd_x, "mu_y": mu_y}


def predict_asymmetric_ridge(fit, X):
    if fit is None:
        return np.full(X.shape[0], np.nan)
    Xs = (X.astype(np.float64) - fit["mu_x"]) / fit["sd_x"]
    return (Xs @ fit["w"]) + fit["mu_y"]


def build_baran_feat_matrix(mu_abc, lv_abc, mu_baran, lv_baran):
    """Build the (N, 8) feature matrix for the Baran-as-feature lignin head."""
    return np.column_stack([
        mu_abc[:, 0, IDX_LIGNIN],
        mu_abc[:, 1, IDX_LIGNIN],
        mu_abc[:, 2, IDX_LIGNIN],
        lv_abc[:, 0, IDX_LIGNIN],
        lv_abc[:, 1, IDX_LIGNIN],
        lv_abc[:, 2, IDX_LIGNIN],
        mu_baran[:, IDX_LIGNIN],
        lv_baran[:, IDX_LIGNIN],
    ]).astype(np.float64)


# ==========================================================================
# Tanimoto NN-to-train on Morgan FP (binary)
# ==========================================================================
def tanimoto_nn_max(fp_te, fp_tr, batch=256):
    """Max Tanimoto similarity from each test row to any train row.

    Treats fp as binary (>0). Returns (N_te,) float in [0, 1]."""
    A = (fp_te > 0).astype(np.float32)
    B = (fp_tr > 0).astype(np.float32)
    sb = B.sum(axis=1)  # (N_tr,)
    out = np.zeros(A.shape[0], dtype=np.float32)
    for i in range(0, A.shape[0], batch):
        Ab = A[i:i + batch]
        sa = Ab.sum(axis=1, keepdims=True)         # (b, 1)
        inter = Ab @ B.T                            # (b, N_tr)
        union = sa + sb[None, :] - inter
        sim = inter / np.maximum(union, 1e-8)
        out[i:i + batch] = sim.max(axis=1)
    return out


# ==========================================================================
# Main fold loop
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-specialist-seeds", type=int, default=2)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--router-epochs", type=int, default=120)
    ap.add_argument("--mahal-q", type=float, default=0.9)
    ap.add_argument("--lam-normal", type=float, default=1.0,
                    help="Ridge L2 on the 6 specialist mu/lv features.")
    ap.add_argument("--lam-baran", type=float, default=4.0,
                    help="Ridge L2 on the 2 Baran mu/lv features (heavier "
                         "penalty so the head doesn't trivially copy Baran).")
    ap.add_argument("--fold", type=int, default=None,
                    help="Run only this fold index (for SLURM array).")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  specialist_seeds={args.n_specialist_seeds}  "
          f"mahal_q={args.mahal_q}  lam_normal={args.lam_normal}  "
          f"lam_baran={args.lam_baran}")

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
    per_row_records = []

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

        mu_abc_tr, lv_abc_tr = _specialist_pillar_outputs(specialists, feats_tr, device)
        mu_abc_te, lv_abc_te = _specialist_pillar_outputs(specialists, feats_te, device)

        pred_ens_tr = mu_abc_tr.mean(axis=1)
        pred_ens_te = mu_abc_te.mean(axis=1)
        aleatoric_te = np.exp(lv_abc_te).mean(axis=1)
        epistemic_te = mu_abc_te.var(axis=1)
        total_sigma_row = np.sqrt(aleatoric_te + epistemic_te).mean(axis=-1)

        y_true = y_te[:, IDX_LIGNIN]
        pred_ens_lig = pred_ens_te[:, IDX_LIGNIN]
        r2_ens_all = float(r2_score(y_true, pred_ens_lig))
        ens_gated = {q: _gated(pred_ens_lig, y_true, total_sigma_row, q)
                      for q in (0.25, 0.5, 0.75)}

        # ---- Baran GB per property ----
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
        lv_baran = np.log(baran_resid_var).astype(np.float32)
        lv_baran_tr = np.broadcast_to(lv_baran, mu_baran_tr.shape).astype(np.float32)
        lv_baran_te = np.broadcast_to(lv_baran, mu_baran_te.shape).astype(np.float32)

        r2_baran_alone = float(r2_score(y_true, mu_baran_te[:, IDX_LIGNIN]))
        mae_baran_alone = float(mean_absolute_error(y_true, mu_baran_te[:, IDX_LIGNIN]))

        # 4-pillar BMA-K4 + Mahal gate (same as parent script)
        mu_all_tr = np.concatenate(
            [mu_abc_tr, mu_baran_tr[:, None, :]], axis=1).astype(np.float32)
        lv_all_tr = np.concatenate(
            [lv_abc_tr, lv_baran_tr[:, None, :]], axis=1).astype(np.float32)
        mu_all_te = np.concatenate(
            [mu_abc_te, mu_baran_te[:, None, :]], axis=1).astype(np.float32)
        lv_all_te = np.concatenate(
            [lv_abc_te, lv_baran_te[:, None, :]], axis=1).astype(np.float32)

        router_k4 = train_scalar_k4_router(
            mu_all_tr, lv_all_tr, y_tr, device,
            epochs=args.router_epochs, patience=30)
        mu_k4_te, lv_k4_te, w_k4_te = fuse_k4(router_k4, mu_all_te, lv_all_te)
        pred_k4_lig = mu_k4_te[:, IDX_LIGNIN]
        r2_k4_all = float(r2_score(y_true, pred_k4_lig))
        total_sigma_k4 = np.sqrt(np.exp(lv_k4_te) + epistemic_te).mean(axis=-1)
        k4_gated = {q: _gated(pred_k4_lig, y_true, total_sigma_k4, q)
                     for q in (0.25, 0.5, 0.75)}
        w_baran_mean_lig = float(w_k4_te[:, 3, IDX_LIGNIN].mean())

        mu_m, cov_inv_m, thr_m, d2_tr_m = fit_mahal_gate(f_tr, q=args.mahal_q)
        d2_te_m = mahal_d2(f_te, mu_m, cov_inv_m)
        ood_mask = d2_te_m > thr_m
        pred_gated = pred_k4_lig.copy()
        pred_gated[ood_mask] = mu_baran_te[ood_mask, IDX_LIGNIN]
        r2_gated_all = float(r2_score(y_true, pred_gated))
        n_ood = int(ood_mask.sum())

        # ---- Idea #3 (in-fold): Baran-as-feature lignin head ----
        X_head_tr = build_baran_feat_matrix(mu_abc_tr, lv_abc_tr,
                                            mu_baran_tr, lv_baran_tr)
        X_head_te = build_baran_feat_matrix(mu_abc_te, lv_abc_te,
                                            mu_baran_te, lv_baran_te)
        lam_per_col = np.array([args.lam_normal] * 6 + [args.lam_baran] * 2,
                               dtype=np.float64)
        head_fit = fit_asymmetric_ridge(X_head_tr, y_tr[:, IDX_LIGNIN],
                                         lam_per_col)
        pred_head_lig = predict_asymmetric_ridge(head_fit, X_head_te)
        r2_head_all = float(r2_score(y_true, pred_head_lig))
        head_gated = {q: _gated(pred_head_lig, y_true, total_sigma_row, q)
                       for q in (0.25, 0.5, 0.75)}

        # ---- Tanimoto-NN-to-train (raw Morgan FP) ----
        tan_nn = tanimoto_nn_max(pool_mg[te_mask], pool_mg[tr_mask])

        # ---- Per-row records for the meta-stacker ----
        il_names_te = pool_il[te_mask]
        for i in range(len(y_true)):
            per_row_records.append({
                "fold": k,
                "il_name": str(il_names_te[i]),
                "y_true": float(y_true[i]),
                "pred_ens_lig": float(pred_ens_lig[i]),
                "pred_k4_lig": float(pred_k4_lig[i]),
                "pred_gated": float(pred_gated[i]),
                "pred_baran_lig": float(mu_baran_te[i, IDX_LIGNIN]),
                "pred_baran_feat_head": float(pred_head_lig[i]),
                "sigma_alea_lig": float(aleatoric_te[i, IDX_LIGNIN]),
                "sigma_epi_lig": float(epistemic_te[i, IDX_LIGNIN]),
                "total_sigma_row": float(total_sigma_row[i]),
                "mahal_d2": float(d2_te_m[i]),
                "tanimoto_nn": float(tan_nn[i]),
                "ood_mask": int(ood_mask[i]),
                "w_baran_lig": float(w_k4_te[i, 3, IDX_LIGNIN]),
            })

        def _fmt(d, q):
            return f"{d[q]['r2']:+.4f}" if d[q] else "n/a"
        print(f"  A5.9 ens       : R² = {r2_ens_all:+.4f}  g50={_fmt(ens_gated, 0.5)}")
        print(f"  + Baran pillar : R² = {r2_k4_all:+.4f}  g50={_fmt(k4_gated, 0.5)}  "
              f"mean w_D(lig)={w_baran_mean_lig:.2f}")
        print(f"  + Mahal gate   : R² = {r2_gated_all:+.4f}  ({n_ood}/{len(y_true)} OOD; "
              f"thr d²={thr_m:.2f})")
        print(f"  Baran-feat hd  : R² = {r2_head_all:+.4f}  g50={_fmt(head_gated, 0.5)}  "
              f"(lam_n={args.lam_normal}, lam_b={args.lam_baran})")
        print(f"  Baran alone    : R² = {r2_baran_alone:+.4f}   MAE = {mae_baran_alone:.3f}")
        print(f"  tanimoto_nn    : mean={tan_nn.mean():.3f}  min={tan_nn.min():.3f}  "
              f"max={tan_nn.max():.3f}")

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
            "baran_feat_head": {
                "r2_all": r2_head_all,
                "r2_gated": {str(q): head_gated[q] for q in (0.25, 0.5, 0.75)},
                "weights": head_fit["w"].tolist() if head_fit else None,
                "lam_normal": args.lam_normal,
                "lam_baran": args.lam_baran,
            },
            "baran_alone": {
                "r2": r2_baran_alone,
                "mae": mae_baran_alone,
            },
            "aleatoric_lignin_mean": float(aleatoric_te[:, IDX_LIGNIN].mean()),
            "epistemic_lignin_mean": float(epistemic_te[:, IDX_LIGNIN].mean()),
            "tanimoto_nn_mean": float(tan_nn.mean()),
        })

    # ---- Write per-row CSV (always; one per fold or one aggregate) ----
    if per_row_records:
        if args.fold is not None:
            csv_path = RESULTS / f"lignos_baran_feat_meta_fold_{args.fold}_rows.csv"
        else:
            csv_path = RESULTS / "lignos_baran_feat_meta_rows.csv"
        fieldnames = list(per_row_records[0].keys())
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_row_records)
        print(f"\nSaved per-row CSV ({len(per_row_records)} rows): {csv_path}")

    # ---- Per-fold mode: write one JSON per fold ----
    if args.fold is not None:
        out = RESULTS / f"lignos_baran_feat_meta_fold_{args.fold}.json"
        json.dump({
            "fold": args.fold,
            "result": fold_results[0] if fold_results else None,
            "n_specialist_seeds": args.n_specialist_seeds,
            "mahal_q": args.mahal_q,
            "lam_normal": args.lam_normal,
            "lam_baran": args.lam_baran,
        }, open(out, "w"), indent=2)
        print(f"Saved fold {args.fold}: {out}")
        return

    def _agg(key):
        xs = [f[key]["r2_all"] for f in fold_results]
        return float(np.mean(xs)), float(np.std(xs))
    ens_m, ens_s = _agg("ensemble")
    k4_m, k4_s = _agg("k4_bma")
    mg_m, mg_s = _agg("k4_mahal")
    hd_m, hd_s = _agg("baran_feat_head")
    ba_m, ba_s = (
        float(np.mean([f["baran_alone"]["r2"] for f in fold_results])),
        float(np.std([f["baran_alone"]["r2"] for f in fold_results])),
    )

    print(f"\n{'='*70}")
    print("Baran Task 2 — LIGNOS + BMA-K4 + Mahal gate + Baran-feat head (in-fold)")
    print(f"{'='*70}")
    print(f"A5.9 ensemble ALL        : R² = {ens_m:+.4f} ± {ens_s:.4f}")
    print(f"+ Baran BMA pillar ALL   : R² = {k4_m:+.4f} ± {k4_s:.4f}")
    print(f"+ Mahalanobis gate ALL   : R² = {mg_m:+.4f} ± {mg_s:.4f}")
    print(f"Baran-feat head (in-fld) : R² = {hd_m:+.4f} ± {hd_s:.4f}")
    print(f"Baran alone (this run)   : R² = {ba_m:+.4f} ± {ba_s:.4f}")
    print(f"Baran GB (published)     : R² = +0.5238 ± 0.2015")

    out = RESULTS / "lignos_baran_feat_meta.json"
    json.dump({
        "folds": fold_results,
        "ensemble_r2_mean": ens_m, "ensemble_r2_std": ens_s,
        "k4_bma_r2_mean": k4_m, "k4_bma_r2_std": k4_s,
        "k4_mahal_r2_mean": mg_m, "k4_mahal_r2_std": mg_s,
        "baran_feat_head_r2_mean": hd_m, "baran_feat_head_r2_std": hd_s,
        "baran_alone_r2_mean": ba_m, "baran_alone_r2_std": ba_s,
        "n_specialist_seeds": args.n_specialist_seeds,
        "mahal_q": args.mahal_q,
        "lam_normal": args.lam_normal,
        "lam_baran": args.lam_baran,
        "n_splits": args.n_splits,
    }, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
