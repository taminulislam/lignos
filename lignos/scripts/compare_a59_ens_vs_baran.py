"""A5.9-ensemble Baran Task 2 CV with total (aleatoric + epistemic) uncertainty.

Uses the 3 A5.9 specialists (A, B, C) as a deep ensemble:
  - aleatoric σ²(row, prop) = mean over specialists of exp(lv_k)     # per-specialist noise
  - epistemic σ²(row, prop) = var over specialists of μ_k             # disagreement
  - total_σ²                = aleatoric + epistemic                   # Lakshminarayanan 2017

The key insight (vs the previous compare_a5unc_vs_baran.py using A5.2 alone):
A5.2's 3 seeds all converge to nearly-identical means because they share the
same frozen A2 backbone and only differ in their logvar heads — epistemic
variance is ≈ 0. A5.9's 3 specialists have DIFFERENT architectures (SMILES-
only, SMILES+Surface+Frame, SMILES+COSMO) so they disagree meaningfully on
OOD inputs, giving REAL epistemic signal.

For each of 5 IL-stratified folds on Baran-matched ILs:
  1. Re-train each specialist on the train fold (Baran held-out ILs removed)
  2. Predict (μ_k, lv_k) per specialist on held-out rows
  3. Ensemble μ = mean(μ_k); total σ² = mean(exp(lv_k)) + var(μ_k)
  4. Confidence-gated R² @ quantiles {25%, 50%, 75%}

Expected outcome if epistemic works:
  - Fold 3 (R² = −2.4 on [C2H4COOHmim][Cl]) will have high epistemic σ²
    because specialists disagree on novel cation; gating 50% should exclude it
  - Gated@50% R² mean should lift from A5.2's catastrophic −5e13 to ~0.3–0.5+
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, r2_per_prop, set_seed  # noqa
from train_a2_two_stage import build_chemprop_40d, v4_base, preprocess_physchem
from train_a5_bma_pipeline import (
    A5_BMA_Specialist, train_specialist, _assemble_bank, _standardize,
    FRAME_DIM, SURFACE_DIM, COSMO_DIM, VIT_BANK, COSMO_BANK,
)
from compare_a2_vs_baran import _load_baran_matched
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error

CACHE = V5 / "data" / "LignoIL_A1"
RESULTS = V5 / "results"
IDX_LIGNIN = 7


def _gated(pred, y, total_sigma, quantile):
    if len(pred) == 0: return None
    thr = np.quantile(total_sigma, quantile)
    keep = total_sigma <= thr
    if keep.sum() < 2: return None
    yk, pk = y[keep], pred[keep]
    return {"r2": float(r2_score(yk, pk)),
            "mae": float(mean_absolute_error(yk, pk)),
            "n_keep": int(keep.sum()), "n_total": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=2, help="Seeds per specialist per fold")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
    pool_v4 = (0.4 * pool_pf + 0.6 * pool_pc).astype(np.float32)

    # Per-IL modality banks (once, independent of fold)
    vit_bank = dict(zip(*[np.load(VIT_BANK, allow_pickle=True)[k]
                            for k in ("smiles", "vit_feat")]))
    cos_bank = dict(zip(*[np.load(COSMO_BANK, allow_pickle=True)[k]
                            for k in ("smiles", "cosmo_feat")]))
    pool_vit, pool_hv = _assemble_bank(pool_smiles, vit_bank, FRAME_DIM)
    pool_cos, pool_hc = _assemble_bank(pool_smiles, cos_bank, COSMO_DIM)
    pool_hs = (pool_surf != 0).any(axis=1).astype(np.float32)

    fold_results = []
    for k, held in enumerate(folds):
        te_mask = np.isin(pool_il, held) & baran_lig_mask
        tr_mask = ~np.isin(pool_il, held)
        if te_mask.sum() == 0:
            print(f"  Fold {k}: 0 test rows — skip")
            continue

        # PCA + standardization on this fold's train set
        pca = PCA(40).fit(pool_mg[tr_mask])
        f_tr = pca.transform(pool_mg[tr_mask]).astype(np.float32)
        f_te = pca.transform(pool_mg[te_mask]).astype(np.float32)
        cp_tr, cp_te = build_chemprop_40d(pool_cp[tr_mask], pool_cp[te_mask])

        # Standardize modality features on fold-train
        surf_tr = pool_surf[tr_mask]; hs_tr = pool_hs[tr_mask]
        surf_te = pool_surf[te_mask]; hs_te = pool_hs[te_mask]
        surf_tr_z, mu_p, sd_p = _standardize(surf_tr, hs_tr)
        surf_te_z = ((surf_te - mu_p) / sd_p).astype(np.float32) * hs_te[:, None]

        vit_tr = pool_vit[tr_mask]; hv_tr = pool_hv[tr_mask]
        vit_te = pool_vit[te_mask]; hv_te = pool_hv[te_mask]
        vit_tr_z, mu_v, sd_v = _standardize(vit_tr, hv_tr)
        vit_te_z = ((vit_te - mu_v) / sd_v).astype(np.float32) * hv_te[:, None]

        cos_tr = pool_cos[tr_mask]; hc_tr = pool_hc[tr_mask]
        cos_te = pool_cos[te_mask]; hc_te = pool_hc[te_mask]
        cos_tr_z, mu_c, sd_c = _standardize(cos_tr, hc_tr)
        cos_te_z = ((cos_te - mu_c) / sd_c).astype(np.float32) * hc_te[:, None]

        v4_tr = pool_v4[tr_mask]; v4_te = pool_v4[te_mask]
        th_tr = pool_th[tr_mask]; th_te = pool_th[te_mask]
        y_tr = pool_y[tr_mask]; y_te = pool_y[te_mask]

        feats_tr = {"v4": v4_tr, "morg": f_tr, "thermo": th_tr, "chemprop": cp_tr,
                     "surface": surf_tr_z, "vit": vit_tr_z, "cos": cos_tr_z,
                     "has_surf": hs_tr, "has_vit": hv_tr, "has_cos": hc_tr}

        # Train 3 specialists × n_seeds, collect per-(specialist, seed) predictions
        all_mus = []          # shape: (K, S, n_te, n_props)
        all_lvs = []
        for kind in ("A", "B", "C"):
            seed_mus, seed_lvs = [], []
            for seed in range(args.n_seeds):
                m_k, _ = train_specialist(kind, seed, feats_tr, y_tr, device,
                                            epochs=args.epochs, patience=40)
                mu, lv = m_k.forward_with_lv(
                    torch.from_numpy(v4_te).to(device),
                    torch.from_numpy(f_te).to(device),
                    torch.from_numpy(th_te).to(device),
                    torch.from_numpy(cp_te).to(device),
                    surface=torch.from_numpy(surf_te_z).to(device),
                    vit=torch.from_numpy(vit_te_z).to(device),
                    cos=torch.from_numpy(cos_te_z).to(device),
                    has_surf=torch.from_numpy(hs_te).to(device),
                    has_vit=torch.from_numpy(hv_te).to(device),
                    has_cos=torch.from_numpy(hc_te).to(device))
                seed_mus.append(mu.detach().cpu().numpy())
                seed_lvs.append(lv.detach().cpu().numpy())
            all_mus.append(np.stack(seed_mus))    # (S, N, P)
            all_lvs.append(np.stack(seed_lvs))
        all_mus = np.stack(all_mus)               # (K, S, N, P)
        all_lvs = np.stack(all_lvs)

        # Ensemble prediction: mean over (K, S)
        pred_ens = all_mus.mean(axis=(0, 1))      # (N, P)
        # Aleatoric: mean σ² over (K, S)
        aleatoric = np.exp(all_lvs).mean(axis=(0, 1))   # (N, P)
        # Epistemic: variance of μ over (K, S)
        epistemic = all_mus.reshape(-1, *all_mus.shape[2:]).var(axis=0)  # (N, P)
        total_var = aleatoric + epistemic                  # (N, P)
        total_sigma_row = np.sqrt(total_var).mean(axis=-1) # (N,) — per-row scalar

        y_true = y_te[:, IDX_LIGNIN]
        pred_lig = pred_ens[:, IDX_LIGNIN]

        r2_all = r2_score(y_true, pred_lig)
        mae_all = mean_absolute_error(y_true, pred_lig)
        gated = {}
        for q in [0.25, 0.5, 0.75]:
            gated[q] = _gated(pred_lig, y_true, total_sigma_row, q)

        held_names = sorted(set(pool_il[te_mask]))
        g50 = gated[0.5]["r2"] if gated[0.5] else None
        g25 = gated[0.25]["r2"] if gated[0.25] else None
        print(f"  Fold {k}: n={te_mask.sum()}  R² all={r2_all:+.4f}  "
              f"gated@50%={f'{g50:+.4f}' if g50 is not None else 'n/a'}  "
              f"gated@25%={f'{g25:+.4f}' if g25 is not None else 'n/a'}  "
              f"epistemic/aleatoric = {epistemic[:, IDX_LIGNIN].mean():.2f}/"
              f"{aleatoric[:, IDX_LIGNIN].mean():.2f}  ILs={held_names[:2]}...")
        fold_results.append({
            "fold": k, "r2_all": float(r2_all), "mae_all": float(mae_all),
            "r2_gated": {str(q): gated[q] for q in [0.25, 0.5, 0.75]},
            "aleatoric_mean": float(aleatoric[:, IDX_LIGNIN].mean()),
            "epistemic_mean": float(epistemic[:, IDX_LIGNIN].mean()),
            "n": int(te_mask.sum()), "held_out_ils": held_names,
        })

    r2_all_mean = float(np.mean([f["r2_all"] for f in fold_results]))
    r2_all_std = float(np.std([f["r2_all"] for f in fold_results]))
    for q_key, label in [("0.25", "25%"), ("0.5", "50%"), ("0.75", "75%")]:
        vals = [f["r2_gated"][q_key]["r2"] for f in fold_results if f["r2_gated"][q_key]]
        if vals:
            print(f"gated@{label:5s}: mean R² = {np.mean(vals):+.4f}  "
                   f"(over {len(vals)}/{len(fold_results)} folds)")

    print(f"\n{'='*70}\nA5.9 Ensemble Baran Task 2 CV (total = aleatoric + epistemic)\n{'='*70}")
    print(f"A5.9 ensemble ALL preds : R² = {r2_all_mean:+.4f} ± {r2_all_std:.4f}")
    for q_key, label in [("0.25", "25%"), ("0.5", "50%"), ("0.75", "75%")]:
        vals = [f["r2_gated"][q_key]["r2"] for f in fold_results if f["r2_gated"][q_key]]
        if vals:
            print(f"A5.9 gated @ {label:5s}   : R² = {np.mean(vals):+.4f}")
    print(f"A2 baseline (no gate)   : R² = -0.41 ± 1.04")
    print(f"A5.2 aleatoric only     : R² all=-1.35, gated@50%=-5e13")
    print(f"Baran GB (their CV)     : R² = +0.52 ± 0.20")

    out = RESULTS / "a59_ensemble_baran_task2.json"
    json.dump({"folds": fold_results,
               "r2_all_mean": r2_all_mean, "r2_all_std": r2_all_std,
               "n_seeds_per_specialist": args.n_seeds,
               "n_splits": args.n_splits}, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
