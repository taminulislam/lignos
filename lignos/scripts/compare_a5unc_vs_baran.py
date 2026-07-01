"""Baran Task 2 CV with A5.2 uncertainty head — confidence-gated evaluation.

For each of 5 IL-stratified folds on Baran-only ILs:
  1. Train A5.2 Stage-1 (A2 backbone frozen, logvar heads trainable) on the
     train fold using main NLL loss.
  2. Predict on held-out IL rows → (mean, log_var) per target.
  3. Report:
       - R² on all held-out predictions (should match A2 Task 2: -0.41)
       - R² on 50% lowest-epistemic-σ subset (where A5.2 is confident)
       - R² on 75% lowest-σ subset
  4. Expected outcome: fold-3 catastrophe (R²=−2.40 on [C2H4COOHmim][Cl])
     lands entirely in the HIGH-σ bucket → gated R² stays positive, mean
     goes negative.
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
from train_a2_two_stage import (
    A2Head, build_chemprop_40d, preprocess_physchem, v4_base,
)
from train_a5_uncertainty import (
    A5UncertaintyHead, gaussian_nll_loss, train_stage1_unc, predict_stage1,
)
from compare_a2_vs_baran import _baran_il_smiles_set, _load_baran_matched
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error

CACHE = V5 / "data" / "LignoIL_A1"
RESULTS = V5 / "results"

# Use index 7 of pred/y for lignin
IDX_LIGNIN = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--n-splits", type=int, default=5)
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
    pool_v4 = (0.4 * pool_pf + 0.6 * pool_pc).astype(np.float32)

    fold_results = []
    for k, held in enumerate(folds):
        te_mask = np.isin(pool_il, held) & baran_lig_mask
        tr_mask = ~np.isin(pool_il, held)
        if te_mask.sum() == 0:
            print(f"  Fold {k}: 0 test rows — skip")
            continue

        pca = PCA(40).fit(pool_mg[tr_mask])
        f_tr = pca.transform(pool_mg[tr_mask]).astype(np.float32)
        f_te = pca.transform(pool_mg[te_mask]).astype(np.float32)
        cp_tr, cp_te = build_chemprop_40d(pool_cp[tr_mask], pool_cp[te_mask])

        v4_tr = pool_v4[tr_mask]; v4_te = pool_v4[te_mask]
        th_tr = pool_th[tr_mask]; th_te = pool_th[te_mask]
        y_tr = pool_y[tr_mask]; y_te = pool_y[te_mask]

        # Train multiple seeds, average the mean and variance
        seed_preds, seed_lvs = [], []
        for seed in range(args.n_seeds):
            m = train_stage1_unc(seed, v4_tr, f_tr, th_tr, cp_tr, y_tr, device,
                                   epochs=300, patience=50)
            pred, lv = predict_stage1(m, v4_te, f_te, th_te, cp_te, device)
            seed_preds.append(pred)
            seed_lvs.append(lv)
        pred = np.mean(seed_preds, axis=0)
        # Total uncertainty = aleatoric (mean of σ²) + epistemic (var across seeds)
        aleatoric = np.exp(np.mean(seed_lvs, axis=0))  # mean σ² across seeds
        epistemic = np.var(np.stack(seed_preds), axis=0)
        total_var = aleatoric + epistemic
        total_sigma = np.sqrt(total_var).mean(axis=-1)  # (N,) per-row sigma

        y_true = y_te[:, IDX_LIGNIN]
        pred_lig = pred[:, IDX_LIGNIN]

        # Overall R² (should match A2 Task 2)
        r2_all = r2_score(y_true, pred_lig)
        mae_all = mean_absolute_error(y_true, pred_lig)

        # Confidence-gated R² at 50% and 75% lowest-sigma
        results_gated = {}
        for q in [0.25, 0.5, 0.75]:
            thr = np.quantile(total_sigma, q)
            keep = total_sigma <= thr
            if keep.sum() < 2:
                results_gated[q] = None; continue
            r2g = r2_score(y_true[keep], pred_lig[keep])
            mae_g = mean_absolute_error(y_true[keep], pred_lig[keep])
            results_gated[q] = {"r2": float(r2g), "mae": float(mae_g),
                                 "n_keep": int(keep.sum()),
                                 "n_total": int(len(y_true))}

        held_names = sorted(set(pool_il[te_mask]))
        g50_str = f"{results_gated[0.5]['r2']:+.4f}" if results_gated[0.5] else "n/a"
        g25_str = f"{results_gated[0.25]['r2']:+.4f}" if results_gated[0.25] else "n/a"
        print(f"  Fold {k}: n={te_mask.sum()}  R² all={r2_all:+.4f}  "
              f"R² gated@50%={g50_str}  R² gated@25%={g25_str}  "
              f"ILs={held_names[:2]}...")
        fold_results.append({
            "fold": k,
            "r2_all": float(r2_all),
            "mae_all": float(mae_all),
            "r2_gated": {str(q): results_gated[q] for q in [0.25, 0.5, 0.75]},
            "n": int(te_mask.sum()),
            "held_out_ils": held_names,
        })

    # Aggregate
    r2_all_mean = float(np.mean([f["r2_all"] for f in fold_results]))
    r2_all_std = float(np.std([f["r2_all"] for f in fold_results]))
    r2_g50_mean = float(np.mean([f["r2_gated"]["0.5"]["r2"] for f in fold_results if f["r2_gated"]["0.5"]]))
    r2_g25_mean = float(np.mean([f["r2_gated"]["0.25"]["r2"] for f in fold_results if f["r2_gated"]["0.25"]]))

    print(f"\n{'='*70}\nA5.2 Baran Task 2 CV — confidence-gated\n{'='*70}")
    print(f"A5.2 ALL preds       : R² = {r2_all_mean:+.4f} ± {r2_all_std:.4f}")
    print(f"A5.2 gated @ 50%     : R² = {r2_g50_mean:+.4f}")
    print(f"A5.2 gated @ 25%     : R² = {r2_g25_mean:+.4f}")
    print(f"Baran GB (their CV)  : R² = +0.5238 ± 0.2015 (reference)")

    out = RESULTS / "a5unc_baran_task2.json"
    json.dump({"folds": fold_results,
               "r2_all_mean": r2_all_mean, "r2_all_std": r2_all_std,
               "r2_gated_50": r2_g50_mean, "r2_gated_25": r2_g25_mean,
               "n_seeds": args.n_seeds, "n_splits": args.n_splits},
              open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
