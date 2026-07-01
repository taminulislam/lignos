"""XGBoost on the full LIGNOS feature stack — isolates architecture from features.

LIGNOS consumes these streams in its specialists and backbone:
    chemprop_fp (300), surface_fp (256), thermo_feat (25),
    morgan_fp (2048 -> PCA-40), preds_fusion[:, :7], preds_chemprop[:, :7]
    = 635-D feature vector.

We drop target index 7 (lignin) from preds_* to avoid leakage (those slots are
precomputed lignin predictions on the same rows during pretraining).

Two protocols:
- Task 1: same as Chemprop/Baran baselines — train on cached_train+val
  (lignin-labeled), evaluate on cached_test (39 rows).
- Task 2: 5-fold leave-IL-out on Baran-matched 13-IL pool, mutates each
  per-fold row CSV with `pred_xgb_lig`.

Uses XGBoost 2.1 with reasonable-but-not-tuned defaults (tree_method=hist,
n_estimators=600, depth=5, lr=0.05, l2=1.0).
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from compare_a2_vs_baran import _load_baran_matched  # noqa
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error

CACHE = V5 / "data" / "LignoIL"
RESULTS = V5 / "results"
IDX_LIGNIN = 7


def build_full_stack(d: dict, pca_morgan: PCA | None = None, n_pca: int = 40):
    """Concatenate all LIGNOS feature streams into a single matrix.

    Returns (X, pca) where pca is the fitted (or reused) Morgan PCA.
    """
    mg = d["morgan_fp"].astype(np.float32)
    if pca_morgan is None:
        pca_morgan = PCA(n_components=n_pca).fit(mg)
    mg_pca = pca_morgan.transform(mg).astype(np.float32)
    pf = d["preds_fusion"][:, :IDX_LIGNIN].astype(np.float32)
    pc = d["preds_chemprop"][:, :IDX_LIGNIN].astype(np.float32)
    X = np.concatenate([
        d["chemprop_fp"].astype(np.float32),
        d["surface_fp"].astype(np.float32),
        d["thermo_feat"].astype(np.float32),
        mg_pca,
        pf,
        pc,
    ], axis=1)
    return X, pca_morgan


def fit_xgb(X_tr, y_tr, seed=42, n_estimators=600, max_depth=5, lr=0.05):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", random_state=seed, n_jobs=4, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model


def task1(args):
    tr = {k: v for k, v in np.load(CACHE / "cached_train.npz", allow_pickle=True).items()}
    va = {k: v for k, v in np.load(CACHE / "cached_val.npz", allow_pickle=True).items()}
    te = {k: v for k, v in np.load(CACHE / "cached_test.npz", allow_pickle=True).items()}

    # Fit PCA on train+val+test morgan for stability (unsupervised, no leakage).
    mg_all = np.concatenate([tr["morgan_fp"], va["morgan_fp"], te["morgan_fp"]]).astype(np.float32)
    pca = PCA(n_components=40).fit(mg_all)

    X_tr, _ = build_full_stack(tr, pca); y_tr_full = tr["targets"][:, IDX_LIGNIN]
    X_va, _ = build_full_stack(va, pca); y_va_full = va["targets"][:, IDX_LIGNIN]
    X_te, _ = build_full_stack(te, pca); y_te_full = te["targets"][:, IDX_LIGNIN]

    X_pool = np.concatenate([X_tr, X_va])
    y_pool = np.concatenate([y_tr_full, y_va_full])
    ok_pool = ~np.isnan(y_pool)
    X_pool = X_pool[ok_pool]; y_pool = y_pool[ok_pool]
    ok_te = ~np.isnan(y_te_full)
    X_te = X_te[ok_te]; y_te = y_te_full[ok_te]

    print(f"Task 1  XGBoost — X.shape = {X_pool.shape}, test = {X_te.shape}")

    seed_r2 = []
    seed_preds = []
    for s in range(args.n_seeds):
        m = fit_xgb(X_pool, y_pool, seed=42 + s)
        p = m.predict(X_te)
        r2 = float(r2_score(y_te, p))
        seed_r2.append(r2); seed_preds.append(p)
        print(f"  seed {s}: R² = {r2:+.4f}")
    pred_avg = np.stack(seed_preds).mean(axis=0)
    r2_avg = float(r2_score(y_te, pred_avg))
    mae_avg = float(mean_absolute_error(y_te, pred_avg))
    r2_mu = float(np.mean(seed_r2)); r2_sd = float(np.std(seed_r2))

    print(f"\nTask 1 XGBoost  per-seed R² = {r2_mu:+.4f} ± {r2_sd:.4f}")
    print(f"                R² on seed-averaged preds = {r2_avg:+.4f}   MAE = {mae_avg:.4f}")

    out = {
        "task": "task1", "method": "xgb_full_stack",
        "n_train": int(X_pool.shape[0]), "n_test": int(X_te.shape[0]),
        "n_features": int(X_pool.shape[1]), "n_seeds": args.n_seeds,
        "per_seed_r2": seed_r2,
        "r2_per_seed_mean": r2_mu, "r2_per_seed_std": r2_sd,
        "r2_on_avg_preds": r2_avg, "mae_on_avg_preds": mae_avg,
    }
    out_json = RESULTS / "lignos_xgb_full_stack_task1.json"
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_json.relative_to(PROJECT_ROOT)}")


def task2(args):
    tr, va, te, lig_ils, baran_lig_mask = _load_baran_matched()
    np.random.seed(42)
    il_order = np.random.permutation(lig_ils)
    fold_size = max(1, len(il_order) // args.n_splits)
    folds = [il_order[i * fold_size : (i + 1) * fold_size if i < args.n_splits - 1 else None]
             for i in range(args.n_splits)]

    # Fit a single PCA on the full pool's morgan (unsupervised, same for all folds)
    mg_pool = np.concatenate([tr["morgan_fp"], va["morgan_fp"], te["morgan_fp"]]).astype(np.float32)
    pca = PCA(n_components=40).fit(mg_pool)

    Xtr, _ = build_full_stack(tr, pca)
    Xva, _ = build_full_stack(va, pca)
    Xte, _ = build_full_stack(te, pca)
    pool_X = np.concatenate([Xtr, Xva, Xte])
    pool_il = np.concatenate([tr["il_ids"], va["il_ids"], te["il_ids"]])
    pool_y = np.concatenate([tr["targets"], va["targets"], te["targets"]]).astype(np.float32)

    per_fold = {}
    for k, held in enumerate(folds):
        te_mask = np.isin(pool_il, held) & baran_lig_mask
        tr_mask = (~np.isin(pool_il, held)) & (~np.isnan(pool_y[:, IDX_LIGNIN]))
        if te_mask.sum() == 0:
            print(f"Fold {k}: 0 test rows — skip"); continue
        X_tr_k = pool_X[tr_mask]; y_tr_k = pool_y[tr_mask, IDX_LIGNIN]
        X_te_k = pool_X[te_mask]

        preds = []
        for s in range(args.n_seeds):
            m = fit_xgb(X_tr_k, y_tr_k, seed=42 + s)
            preds.append(m.predict(X_te_k))
        pred_avg = np.stack(preds).mean(axis=0)

        # Mutate row CSV
        csv_path = RESULTS / f"lignos_baran_feat_meta_fold_{k}_rows.csv"
        if not csv_path.exists():
            print(f"Fold {k}: CSV missing — skip row update"); continue
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != len(pred_avg):
            raise RuntimeError(f"Fold {k}: row count mismatch")
        for i, row in enumerate(rows):
            row["pred_xgb_lig"] = float(pred_avg[i])
        fieldnames = list(rows[0].keys())
        if "pred_xgb_lig" not in fieldnames:
            fieldnames.append("pred_xgb_lig")
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        y_te_k = np.array([float(r["y_true"]) for r in rows], dtype=np.float32)
        r2 = float(r2_score(y_te_k, pred_avg))
        per_fold[k] = {"r2": r2, "n_test": int(te_mask.sum()),
                        "n_train": int(tr_mask.sum()), "held_ils": [str(x) for x in held]}
        print(f"Fold {k}: XGBoost R² = {r2:+.4f}  (n_test={int(te_mask.sum())}, "
              f"n_train={int(tr_mask.sum())}, n_seeds={args.n_seeds})")

    r2s = [per_fold[k]["r2"] for k in sorted(per_fold.keys())]
    agg = {"per_fold": per_fold,
            "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
            "n_folds": len(r2s), "n_features": int(pool_X.shape[1])}
    out_json = RESULTS / "lignos_xgb_full_stack_task2.json"
    with open(out_json, "w") as fh:
        json.dump(agg, fh, indent=2)
    print(f"\nTask 2 XGBoost  mean R² = {agg['r2_mean']:+.4f} ± {agg['r2_std']:.4f}")
    print(f"wrote {out_json.relative_to(PROJECT_ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["task1", "task2", "both"], default="both")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    if args.task in ("task1", "both"):
        task1(args)
    if args.task in ("task2", "both"):
        task2(args)


if __name__ == "__main__":
    main()
