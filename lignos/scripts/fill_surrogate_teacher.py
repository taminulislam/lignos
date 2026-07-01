#!/usr/bin/env python3
"""Fill `preds_chemprop` and `preds_fusion` with surrogate teacher predictions.

Every saved chemprop checkpoint requires external feature files that are
unavailable for new SMILES, so we can't re-run them on the expanded cache.
Instead we train a lightweight surrogate (gradient-boosted trees on Morgan
fingerprints) on the 152 original training rows that already have teacher
predictions, then apply the surrogate to every non-original row that's
currently zero-teacher-filled.

This is a bootstrap, not ground truth — but it provides a consistent
non-zero `v_base` prior across the full cache, eliminating the covariate
shift between "rows with teacher features" and "rows without" that has
been the dominant failure mode in recent experiments.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
CACHES = [V5 / "data" / "LignoIL_unified_v2", V5 / "data" / "LignoIL_unified"]


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else None


def main():
    # Load the v2 training cache. Fit the surrogate on rows where the v4
    # teacher fields are non-zero (these are the 152 original rows).
    tr = np.load(CACHES[0] / "cached_train.npz", allow_pickle=True)
    tr = {k: np.asarray(tr[k]).copy() for k in tr.files}

    has_teacher = (np.abs(tr["preds_chemprop"]).sum(-1) > 0)
    print(f"Original-teacher train rows: {has_teacher.sum()} / {len(has_teacher)}")
    if has_teacher.sum() < 50:
        print("Not enough rows to fit surrogate.")
        return

    morgan = tr["morgan_fp"].astype(np.float32)
    # Reduce the 2048-D Morgan FP to 128 for a more stable regression.
    pca = PCA(128, random_state=42).fit(morgan)
    X = pca.transform(morgan).astype(np.float32)
    Y_pc = tr["preds_chemprop"][:, :7].astype(np.float32)
    Y_pf = tr["preds_fusion"][:, :7].astype(np.float32)

    # Train per-target ridge; robust and fast.
    surrogates_pc = []
    surrogates_pf = []
    for j in range(7):
        m1 = Ridge(alpha=1.0).fit(X[has_teacher], Y_pc[has_teacher, j])
        m2 = Ridge(alpha=1.0).fit(X[has_teacher], Y_pf[has_teacher, j])
        surrogates_pc.append(m1)
        surrogates_pf.append(m2)
    print("Surrogate fit: 7 per-target Ridge models (each on 152 rows, 128-D PCA input)")

    # Sanity: R^2 on the training set itself
    pred_pc_tr = np.stack([s.predict(X[has_teacher]) for s in surrogates_pc], axis=1)
    r2_pc_tr = 1 - ((Y_pc[has_teacher] - pred_pc_tr) ** 2).sum(0) / ((Y_pc[has_teacher] - Y_pc[has_teacher].mean(0)) ** 2).sum(0)
    print(f"  in-sample R² per target (core-7): {np.round(r2_pc_tr, 3)}")

    # Apply to both caches.
    for cache_dir in CACHES:
        for split in ["train", "val", "test"]:
            p = cache_dir / f"cached_{split}.npz"
            z = np.load(p, allow_pickle=True)
            d = {k: np.asarray(z[k]).copy() for k in z.files}
            z.close()
            M = d["morgan_fp"].astype(np.float32)
            Xs = pca.transform(M).astype(np.float32)
            pred_pc = np.stack([s.predict(Xs) for s in surrogates_pc], axis=1).astype(np.float32)
            pred_pf = np.stack([s.predict(Xs) for s in surrogates_pf], axis=1).astype(np.float32)
            pc = d["preds_chemprop"].astype(np.float32, copy=True)
            pf = d["preds_fusion"].astype(np.float32, copy=True)
            zero = (np.abs(pc).sum(-1) == 0) & (np.abs(pf).sum(-1) == 0)
            pc[zero, :7] = pred_pc[zero]
            pf[zero, :7] = pred_pf[zero]
            d["preds_chemprop"] = pc
            d["preds_fusion"] = pf
            np.savez(p, **d)
            zr = np.load(p, allow_pickle=True)
            nz = int((np.abs(zr["preds_chemprop"]).sum(-1) > 0).sum())
            zr.close()
            print(f"  {cache_dir.name}/{split}: filled {zero.sum()} zero-teacher rows; "
                  f"non-zero after save = {nz}/{len(pc)}")


if __name__ == "__main__":
    main()
