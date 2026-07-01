"""Verify peak 0.8430 reproduces on the pre-pollution cache (data/expanded/).

Hypothesis: data/LignoIL/cached_train.npz (Apr 17 11:00) was polluted by adding
1051 new rows to the P column with a different scale (density, not pressure),
collapsing the P distribution and bleeding gradient onto H_E/gamma1 via the
shared backbone. data/expanded/ cache (Apr 16 09:32) has the original 152-row
P-only column.

Runs only the Shallow+Unbalanced variant (the reported 0.8430 peak) for 10 seeds.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, predict, r2_per_prop, train_one_seed  # noqa


def load_exp(split):
    d = np.load(V5 / "data" / "expanded" / f"cached_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_v4(split):
    d = np.load(PROJECT_ROOT / "cosmobridge_v4" / "data" / f"cached_{split}.npz",
                 allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    return (0.4 * c["preds_fusion"] + 0.6 * c["preds_chemprop"]).astype(np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Cache: {V5/'data/expanded/cached_train.npz'}")

    tr = load_exp("train")
    te_exp = load_exp("test")  # for morgan_fp, thermo_feat
    te_v4 = load_v4("test")    # for 7-prop targets (identical SMILES, same order)

    pca = PCA(40).fit(tr["morgan_fp"])
    f_tr = pca.transform(tr["morgan_fp"]).astype(np.float32)
    f_te = pca.transform(te_exp["morgan_fp"]).astype(np.float32)
    v4_tr, v4_te = v4_base(tr), v4_base(te_exp)
    y_tr = tr["targets"].astype(np.float32)
    y_te = te_v4["targets"].astype(np.float32)
    th_tr = tr["thermo_feat"].astype(np.float32)
    th_te = te_exp["thermo_feat"].astype(np.float32)
    print(f"train={len(tr['smiles'])} rows, {len(set(str(s) for s in tr['smiles']))} unique SMILES")
    print(f"test={len(te_exp['smiles'])} rows")

    # Per-prop count sanity
    for i, p in enumerate(PROPS[:y_tr.shape[1]]):
        n = int((~np.isnan(y_tr[:, i])).sum())
        print(f"  {p:10s}: {n} train rows")

    r2s = []
    for seed in range(10):
        m = train_one_seed(seed, v4_tr, f_tr, th_tr, y_tr, device=device,
                           balance_props=False, depth="shallow")
        te_pred = predict(m, v4_te, f_te, th_te, device)
        r2s.append(r2_per_prop(te_pred, y_te))

    core7 = np.mean([r["avg_core7"] for r in r2s])
    std = np.std([r["avg_core7"] for r in r2s])
    print(f"\n===== Shallow+Unbalanced on data/expanded/ =====")
    print(f"  core7 = {core7:.4f} ± {std:.4f}  (target peak: 0.8430)")
    for p in PROPS:
        vs = [r.get(p) for r in r2s if r.get(p) is not None and not np.isnan(r.get(p, float("nan")))]
        val = float(np.mean(vs)) if vs else float("nan")
        print(f"  {p:12s}: {val:.4f}")

    out = V5 / "results" / "verify_peak_old_cache.json"
    json.dump({
        "name": "Shallow+Unbalanced on data/expanded",
        "avg_r2_core7": float(core7),
        "std_r2_core7": float(std),
        "per_prop": {p: float(np.mean([r.get(p, float("nan")) for r in r2s])) for p in PROPS},
    }, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
