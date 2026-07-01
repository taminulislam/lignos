#!/usr/bin/env python3
"""Train Combined(40D) on expanded dataset and compare to 0.8309 baseline.

Uses Morgan fingerprints PCA'd to 40D as the feature vector (substitute for
V-JEPA + supervised ViT image features). Evaluates on the UNCHANGED original
test set for apples-to-apples comparison.

Three ablations:
  1. expanded_morgan40:  expanded train + Morgan FP 40D + masked loss
  2. original_morgan40:  original train only + Morgan FP 40D (no expansion)
  3. original_baseline:  original train + original image features (the 0.83 baseline)
"""

import json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))

from audit_residuals import PROPS, predict, r2_per_prop, train_one_seed

def load_expanded(split):
    p = V5 / "data" / "LignoIL" / f"cached_{split}.npz"
    if not p.exists():
        p = V5 / "data" / "expanded" / f"cached_{split}.npz"
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}

def load_original(split):
    d = np.load(PROJECT_ROOT / "cosmobridge_v4" / "data" / f"cached_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}

def v4_base(cached):
    f, c = cached.get("preds_fusion"), cached.get("preds_chemprop")
    if f is not None and c is not None:
        return (0.4 * f + 0.6 * c).astype(np.float32)
    return np.zeros_like(cached["targets"], dtype=np.float32)

def build_morgan_40d(tr, va, te):
    pca = PCA(min(40, tr.shape[1])).fit(tr)
    return pca.transform(tr).astype(np.float32), pca.transform(va).astype(np.float32), pca.transform(te).astype(np.float32)

def run_eval(name, v4_tr, v4_te, f_tr, f_te, th_tr, th_te, y_tr, y_te,
             n_seeds=10, device="cpu", balance_props=True, depth="shallow"):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    seed_r2s = []
    for seed in range(n_seeds):
        model = train_one_seed(seed, v4_tr, f_tr, th_tr, y_tr, device=device,
                               balance_props=balance_props, depth=depth)
        te_pred = predict(model, v4_te, f_te, th_te, device)
        r2 = r2_per_prop(te_pred, y_te)
        seed_r2s.append(r2)
    avg_core7 = float(np.mean([m["avg_core7"] for m in seed_r2s]))
    std_core7 = float(np.std([m["avg_core7"] for m in seed_r2s]))
    avg_all = float(np.mean([m["avg"] for m in seed_r2s]))
    print(f"  {n_seeds}-seed: R2_core7 = {avg_core7:.4f} +/- {std_core7:.4f}  R2_all = {avg_all:.4f}")
    per_prop = {}
    for p in PROPS:
        vals = [m.get(p) for m in seed_r2s if m.get(p) is not None and not np.isnan(m.get(p, float("nan")))]
        per_prop[p] = float(np.mean(vals)) if vals else float("nan")
        print(f"    {p}: {per_prop[p]:.4f}")
    return {"name": name, "avg_r2_core7": avg_core7, "std_r2_core7": std_core7,
            "avg_r2_all": avg_all, "per_prop": per_prop}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    exp_tr = load_expanded("train")
    exp_va = load_expanded("val")
    exp_te = load_expanded("test")
    orig_tr = load_original("train")
    orig_te = load_original("test")
    y_te = orig_te["targets"].astype(np.float32)

    results = []

    # Shared features for expanded runs
    f_tr, f_va, f_te = build_morgan_40d(exp_tr["morgan_fp"], exp_va["morgan_fp"], exp_te["morgan_fp"])
    exp_v4_tr, exp_v4_te = v4_base(exp_tr), v4_base(exp_te)
    exp_y_tr = exp_tr["targets"].astype(np.float32)
    exp_th_tr, exp_th_te = exp_tr["thermo_feat"], exp_te["thermo_feat"]

    # 1. DEEP head + Expanded + Balanced (new)
    results.append(run_eval("DEEP+Expanded+Balanced", exp_v4_tr, exp_v4_te,
        f_tr, f_te, exp_th_tr, exp_th_te, exp_y_tr, y_te,
        device=device, balance_props=True, depth="deep"))

    # 2. DEEP head + Expanded + Unbalanced
    results.append(run_eval("DEEP+Expanded+Unbalanced", exp_v4_tr, exp_v4_te,
        f_tr, f_te, exp_th_tr, exp_th_te, exp_y_tr, y_te,
        device=device, balance_props=False, depth="deep"))

    # 3. Shallow head + Expanded + Balanced (previous best for comparison)
    results.append(run_eval("Shallow+Expanded+Balanced", exp_v4_tr, exp_v4_te,
        f_tr, f_te, exp_th_tr, exp_th_te, exp_y_tr, y_te,
        device=device, balance_props=True, depth="shallow"))

    # 4. Shallow head + Expanded + Unbalanced
    results.append(run_eval("Shallow+Expanded+Unbalanced", exp_v4_tr, exp_v4_te,
        f_tr, f_te, exp_th_tr, exp_th_te, exp_y_tr, y_te,
        device=device, balance_props=False, depth="shallow"))

    # 5. Original baseline (image features) — for reference
    sup = np.load(V5 / "data" / "supervised_vit_features.npz")["features"]
    vj = {}
    for s in ["train","val","test"]:
        p = V5 / "data" / f"cached_image_features_{s}.npz"
        if p.exists(): vj[s] = np.load(p)["vit_feat"]
    if "train" in vj and "test" in vj:
        n_tr, n_te = len(orig_tr["smiles"]), len(orig_te["smiles"])
        pca_vj = PCA(20).fit(vj["train"][:n_tr])
        pca_sup = PCA(20).fit(sup[:n_tr])
        f_tr3 = np.concatenate([pca_vj.transform(vj["train"][:n_tr]), pca_sup.transform(sup[:n_tr])], 1).astype(np.float32)
        f_te3 = np.concatenate([pca_vj.transform(vj["test"][:n_te]), pca_sup.transform(sup[n_tr+32:])], 1).astype(np.float32)
        results.append(run_eval("Original + Image(40D) [baseline]", v4_base(orig_tr), v4_base(orig_te),
            f_tr3, f_te3, orig_tr["thermo_feat"], orig_te["thermo_feat"],
            orig_tr["targets"].astype(np.float32), y_te, device=device))

    # Summary
    print(f"\n{'='*60}\n  COMPARISON\n{'='*60}")
    print(f"{'Name':<40} {'core7_r2':>8} {'std':>8} {'all_r2':>8}")
    print("-"*65)
    for r in results:
        print(f"{r['name']:<40} {r['avg_r2_core7']:>8.4f} {r['std_r2_core7']:>8.4f} {r['avg_r2_all']:>8.4f}")
    out = V5 / "results" / "deep_head_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out,"w"), indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
