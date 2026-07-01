"""Process-feature specialist for Task 2 leave-IL-out CV.

Reproduces the same fold construction as `compare_a59_baran_feat_meta.py`,
then for each fold trains a small gradient-boosted regressor on
**process-only features** (temperature, time, IL concentration, biomass
composition) plus IL macroscopic physicochemical descriptors:

    X = [thermo_feat[:, 0:5], physchem_feat * has_physchem (12 cols)]

This is Specialist E from §sec:future of the LIGNOS paper draft. The intent
is to add a candidate that does NOT depend on atom-level IL chemistry, so
it stays usable when novel IL chemistries are held out — the regime where
LIGNOS specialists collapse.

Appends `pred_procspec_lig` column to each existing per-fold CSV in place.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from compare_a2_vs_baran import _load_baran_matched  # noqa
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

RESULTS = V5 / "results"
IDX_LIGNIN = 7
N_PROCESS = 5  # thermo_feat[:, 0:5] = T, time, IL_conc, biomass C/H


def fit_procspec(X_tr, y_tr, seed=42):
    """Gradient-boosted regressor on process + physchem features."""
    mask = ~np.isnan(y_tr)
    if mask.sum() < 10:
        return None, float(np.nanmean(y_tr)) if mask.sum() else 0.0
    scaler = StandardScaler().fit(X_tr[mask])
    Xs = scaler.transform(X_tr[mask])
    gb = GradientBoostingRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, min_samples_leaf=2, random_state=seed,
    )
    gb.fit(Xs, y_tr[mask])
    return (scaler, gb), float(np.nanmean(y_tr[mask]))


def predict_procspec(fit, X, fallback):
    if fit is None:
        return np.full(X.shape[0], fallback, dtype=np.float32)
    scaler, gb = fit
    Xs = scaler.transform(X)
    return gb.predict(Xs).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="Average predictions over this many random seeds.")
    args = ap.parse_args()

    tr, va, te, lig_ils, baran_lig_mask = _load_baran_matched()
    np.random.seed(42)
    il_order = np.random.permutation(lig_ils)
    fold_size = max(1, len(il_order) // args.n_splits)
    folds = [il_order[i * fold_size : (i + 1) * fold_size if i < args.n_splits - 1 else None]
             for i in range(args.n_splits)]

    pool_il = np.concatenate([tr["il_ids"], va["il_ids"], te["il_ids"]])
    pool_y = np.concatenate([tr["targets"], va["targets"], te["targets"]]).astype(np.float32)
    pool_th = np.concatenate([tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]]).astype(np.float32)
    pool_ph = np.concatenate([tr["physchem_feat"], va["physchem_feat"], te["physchem_feat"]]).astype(np.float32)
    pool_hp = np.concatenate([tr["has_physchem"], va["has_physchem"], te["has_physchem"]]).astype(np.float32)

    # Build feature matrix: [process(5), physchem(12) * has_physchem]
    X_full = np.column_stack([
        pool_th[:, :N_PROCESS],
        pool_ph * pool_hp[:, None],
        pool_hp.reshape(-1, 1),
    ])
    print(f"ProcSpec input dim = {X_full.shape[1]} "
          f"(process={N_PROCESS}, physchem=12, has_physchem flag)")

    from sklearn.metrics import r2_score
    for k, held in enumerate(folds):
        te_mask = np.isin(pool_il, held) & baran_lig_mask
        tr_mask = ~np.isin(pool_il, held)
        if te_mask.sum() == 0:
            print(f"Fold {k}: 0 test rows — skip")
            continue

        y_tr_lig = pool_y[tr_mask, IDX_LIGNIN]

        # Multi-seed average for stability (n_seeds × small GB)
        preds = []
        for s in range(args.n_seeds):
            fit, fb = fit_procspec(X_full[tr_mask], y_tr_lig, seed=42 + s)
            preds.append(predict_procspec(fit, X_full[te_mask], fb))
        pred_proc = np.stack(preds).mean(axis=0)

        csv_path = RESULTS / f"lignos_baran_feat_meta_fold_{k}_rows.csv"
        if not csv_path.exists():
            print(f"Fold {k}: CSV not found — skip")
            continue
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != len(pred_proc):
            print(f"Fold {k}: row count mismatch (csv={len(rows)} vs "
                  f"proc={len(pred_proc)}) — skip")
            continue
        for i, row in enumerate(rows):
            row["pred_procspec_lig"] = float(pred_proc[i])
        fieldnames = list(rows[0].keys())
        if "pred_procspec_lig" not in fieldnames:
            fieldnames.append("pred_procspec_lig")
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        y_te = np.array([float(r["y_true"]) for r in rows])
        r2 = float(r2_score(y_te, pred_proc))
        print(f"Fold {k}: ProcSpec R² = {r2:+.4f}  (n={len(rows)}, "
              f"n_train={int(tr_mask.sum())}, seeds={args.n_seeds})")


if __name__ == "__main__":
    main()
