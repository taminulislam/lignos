"""Priority 2: Gradient boosting baseline on the winner cache.

Uses sklearn HistGradientBoostingRegressor (xgboost/lightgbm not in env).
One model per target, trained on rows where the target is non-NaN,
evaluated on the 39-row test split.

Features per row:
    Morgan-FP PCA(40)  +  thermo_feat(25)  +  v4_base(8)  =  73-D input

Compares:
  * Train-only fit                                — matches prior 10-seed deep setup
  * Train+val combined fit (more data)            — leverages val labels too
  * Per-prop R² on the held-out 39-row test

Writes to lignos/results/gbm_baseline.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, CORE_PROPS  # noqa: E402

CACHE = V5 / "data" / "LignoIL"
N_SEEDS = 5  # HGB is pretty deterministic; 5 seeds suffice


def load_split(s):
    d = np.load(CACHE / f"cached_{s}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    return (0.4 * c["preds_fusion"] + 0.6 * c["preds_chemprop"]).astype(np.float32)


def build_X(cache, pca):
    morgan = pca.transform(cache["morgan_fp"]).astype(np.float32)
    thermo = cache["thermo_feat"].astype(np.float32)
    v = v4_base(cache)
    return np.concatenate([morgan, thermo, v], axis=1)


def per_prop_r2(preds, targets):
    out = {}
    for j, name in enumerate(PROPS):
        mask = ~np.isnan(targets[:, j])
        if mask.sum() == 0:
            out[name] = float("nan"); continue
        t = targets[mask, j]; p = preds[mask, j]
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum() + 1e-8
        out[name] = float(1 - ss_res / ss_tot)
    valid_core7 = [out[p] for p in CORE_PROPS if not np.isnan(out[p])]
    out["avg_core7"] = float(np.mean(valid_core7)) if valid_core7 else float("nan")
    return out


def train_hgb_ensemble(X_tr, y_tr, X_te, n_seeds=N_SEEDS, hgb_kwargs=None):
    """Train one HGB per target per seed, average predictions."""
    default = dict(
        max_depth=None, max_iter=300, learning_rate=0.05,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30,
    )
    if hgb_kwargs: default.update(hgb_kwargs)

    preds_all = []
    for seed in range(n_seeds):
        preds = np.zeros((len(X_te), y_tr.shape[1]), dtype=np.float32)
        for j in range(y_tr.shape[1]):
            mask = ~np.isnan(y_tr[:, j])
            if mask.sum() < 20:
                continue
            gbm = HistGradientBoostingRegressor(random_state=seed, **default)
            gbm.fit(X_tr[mask], y_tr[mask, j])
            preds[:, j] = gbm.predict(X_te)
        preds_all.append(preds)
    return np.mean(preds_all, axis=0)


def main():
    print("Loading caches...")
    tr = load_split("train"); va = load_split("val"); te = load_split("test")
    pca = PCA(40).fit(tr["morgan_fp"])
    X_tr = build_X(tr, pca); X_va = build_X(va, pca); X_te = build_X(te, pca)
    y_tr = tr["targets"].astype(np.float32)
    y_va = va["targets"].astype(np.float32)
    y_te = te["targets"].astype(np.float32)
    print(f"Feature dim: {X_tr.shape[1]}  |  train={X_tr.shape[0]}  val={X_va.shape[0]}  test={X_te.shape[0]}")

    results = []
    for label, X, y in [
        ("train only", X_tr, y_tr),
        ("train+val", np.vstack([X_tr, X_va]), np.vstack([y_tr, y_va])),
    ]:
        print(f"\n--- {label}  (n={len(X)}) ---")
        preds = train_hgb_ensemble(X, y, X_te)
        r2 = per_prop_r2(preds, y_te)
        print(f"  core7={r2['avg_core7']:.4f}  lignin={r2['lignin_wt']:.4f}")
        for p in PROPS:
            print(f"    {p:12s}: {r2[p]:.4f}")
        results.append({"name": f"HGB/{label}", **r2})

    # Also try a core-7-only run to show where the ceiling is without lignin pull
    print(f"\n--- HGB/train+val, core7 only ---")
    y_core = np.vstack([y_tr, y_va])[:, :7]
    pca_c = PCA(40).fit(tr["morgan_fp"])
    # Reuse X but predict only core7
    preds_core = train_hgb_ensemble(np.vstack([X_tr, X_va]), y_core, X_te)
    # R² per core prop
    out = {}
    for j, name in enumerate(CORE_PROPS):
        mask = ~np.isnan(y_te[:, j])
        t = y_te[mask, j]; p = preds_core[mask, j]
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum() + 1e-8
        out[name] = float(1 - ss_res / ss_tot)
    avg_c7 = float(np.mean(list(out.values())))
    print(f"  core7={avg_c7:.4f}")
    for p in CORE_PROPS:
        print(f"    {p:12s}: {out[p]:.4f}")
    results.append({"name": "HGB/core7_only", **out, "avg_core7": avg_c7})

    print(f"\n{'='*68}\nGBM BASELINE — vs deep-head winner (0.8267 core7 / 0.6166 lignin)\n{'='*68}")
    print(f"{'Variant':<45}{'core7':>12}{'lignin':>12}")
    print("-"*68)
    for r in results:
        lw = r.get("lignin_wt", float("nan"))
        print(f"{r['name']:<45}{r['avg_core7']:>12.4f}{lw:>12.4f}")

    out_path = V5 / "results" / "gbm_baseline.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
