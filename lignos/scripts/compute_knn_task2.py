"""Tanimoto-NN K-NN regression for Task 2 leave-IL-out CV.

Reproduces the same fold construction as `compare_a59_baran_feat_meta.py`,
then for each test row computes a distance-weighted K-NN regression on raw
Morgan fingerprints (binary Tanimoto similarity). Appends `pred_knn_lig`
column to each existing per-fold CSV in place.

Method: K=5, weights = (similarity + 0.01) for stability, NaN-safe over
training rows with missing lignin labels. Mirrors the implicit per-IL
"local average" of Baran 2024's per-IL gradient boosting.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from compare_a2_vs_baran import _load_baran_matched  # noqa

RESULTS = V5 / "results"
IDX_LIGNIN = 7


def tanimoto_sim(fp_te, fp_tr):
    """Pairwise Tanimoto similarity (n_te, n_tr) on binarized fingerprints."""
    A = (fp_te > 0).astype(np.float32)
    B = (fp_tr > 0).astype(np.float32)
    inter = A @ B.T
    sa = A.sum(axis=1, keepdims=True)
    sb = B.sum(axis=1, keepdims=True).T
    union = sa + sb - inter
    return inter / np.maximum(union, 1e-8)


def knn_predict(sim, y_tr, k=5):
    """Distance-weighted K-NN regression on rows with non-NaN y."""
    n_te = sim.shape[0]
    valid = ~np.isnan(y_tr)
    out = np.full(n_te, np.nan, dtype=np.float32)
    if valid.sum() == 0:
        return out
    sim_v = sim[:, valid]
    y_v = y_tr[valid]
    k_eff = min(k, valid.sum())
    for i in range(n_te):
        s = sim_v[i]
        top_idx = np.argpartition(-s, k_eff - 1)[:k_eff]
        w = s[top_idx] + 0.01  # avoid div-by-zero on identical zero rows
        out[i] = float(np.sum(w * y_v[top_idx]) / np.sum(w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    tr, va, te, lig_ils, baran_lig_mask = _load_baran_matched()
    np.random.seed(42)
    il_order = np.random.permutation(lig_ils)
    fold_size = max(1, len(il_order) // args.n_splits)
    folds = [il_order[i * fold_size : (i + 1) * fold_size if i < args.n_splits - 1 else None]
             for i in range(args.n_splits)]

    pool_il = np.concatenate([tr["il_ids"], va["il_ids"], te["il_ids"]])
    pool_y = np.concatenate([tr["targets"], va["targets"], te["targets"]]).astype(np.float32)
    pool_mg = np.concatenate([tr["morgan_fp"], va["morgan_fp"], te["morgan_fp"]]).astype(np.float32)

    for k, held in enumerate(folds):
        te_mask = np.isin(pool_il, held) & baran_lig_mask
        tr_mask = ~np.isin(pool_il, held)
        if te_mask.sum() == 0:
            print(f"Fold {k}: 0 test rows — skip")
            continue

        sim = tanimoto_sim(pool_mg[te_mask], pool_mg[tr_mask])
        y_tr_lig = pool_y[tr_mask, IDX_LIGNIN]
        pred_knn = knn_predict(sim, y_tr_lig, k=args.k)

        csv_path = RESULTS / f"lignos_baran_feat_meta_fold_{k}_rows.csv"
        if not csv_path.exists():
            print(f"Fold {k}: CSV not found at {csv_path} — skip")
            continue
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != len(pred_knn):
            print(f"Fold {k}: row count mismatch (csv={len(rows)} vs knn={len(pred_knn)}) — skip")
            continue

        for i, row in enumerate(rows):
            row["pred_knn_lig"] = float(pred_knn[i])
        fieldnames = list(rows[0].keys())
        if "pred_knn_lig" not in fieldnames:
            fieldnames.append("pred_knn_lig")
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Quick per-fold R²
        from sklearn.metrics import r2_score
        y_te = np.array([float(r["y_true"]) for r in rows])
        ok = ~np.isnan(pred_knn)
        if ok.sum() >= 2:
            r2 = float(r2_score(y_te[ok], pred_knn[ok]))
            tan_max = sim.max(axis=1)
            print(f"Fold {k}: K-NN R² = {r2:+.4f}  (n={int(ok.sum())}, "
                  f"mean tan_nn={tan_max.mean():.3f})")


if __name__ == "__main__":
    main()
