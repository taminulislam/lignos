"""Asymmetric heads: shallow for core7, deep for lignin."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, CORE_PROPS, predict, r2_per_prop, train_one_seed

def load_split(split):
    d = np.load(V5 / "data/LignoIL" / f"cached_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}

def v4_base(c):
    f, p = c.get("preds_fusion"), c.get("preds_chemprop")
    if f is not None and p is not None:
        return (0.4*f + 0.6*p).astype(np.float32)
    return np.zeros_like(c["targets"], dtype=np.float32)

def run_eval(name, v4_tr, v4_te, f_tr, f_te, th_tr, th_te, y_tr, y_te,
             n_seeds=10, device="cpu", **kwargs):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    seed_r2s = []
    for seed in range(n_seeds):
        model = train_one_seed(seed, v4_tr, f_tr, th_tr, y_tr, device=device, **kwargs)
        te_pred = predict(model, v4_te, f_te, th_te, device)
        r2 = r2_per_prop(te_pred, y_te)
        seed_r2s.append(r2)
    avg_core7 = float(np.mean([m["avg_core7"] for m in seed_r2s]))
    std_core7 = float(np.std([m["avg_core7"] for m in seed_r2s]))
    print(f"  {n_seeds}-seed: R2_core7 = {avg_core7:.4f} +/- {std_core7:.4f}")
    per_prop = {}
    for p in PROPS:
        vals = [m.get(p) for m in seed_r2s if m.get(p) is not None and not np.isnan(m.get(p, float("nan")))]
        per_prop[p] = float(np.mean(vals)) if vals else float("nan")
        print(f"    {p}: {per_prop[p]:.4f}")
    return {"name": name, "avg_r2_core7": avg_core7, "std_r2_core7": std_core7, "per_prop": per_prop}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tr = load_split("train")
    te = load_split("test")
    pca = PCA(40).fit(tr["morgan_fp"])
    f_tr = pca.transform(tr["morgan_fp"]).astype(np.float32)
    f_te = pca.transform(te["morgan_fp"]).astype(np.float32)
    v4_tr, v4_te = v4_base(tr), v4_base(te)
    y_tr = tr["targets"].astype(np.float32)
    y_te = te["targets"].astype(np.float32)

    results = []

    # 1. Asymmetric: shallow core7 + deep lignin (index 7), unbalanced
    results.append(run_eval("Asymmetric(deep_lignin)+Unbalanced", v4_tr, v4_te,
        f_tr, f_te, tr["thermo_feat"], te["thermo_feat"], y_tr, y_te,
        device=device, balance_props=False, deep_head_indices=[7]))

    # 2. Asymmetric: shallow core7 + deep lignin, balanced
    results.append(run_eval("Asymmetric(deep_lignin)+Balanced", v4_tr, v4_te,
        f_tr, f_te, tr["thermo_feat"], te["thermo_feat"], y_tr, y_te,
        device=device, balance_props=True, deep_head_indices=[7]))

    # 3. Shallow all (previous best for reference)
    results.append(run_eval("Shallow_all+Unbalanced [prev best]", v4_tr, v4_te,
        f_tr, f_te, tr["thermo_feat"], te["thermo_feat"], y_tr, y_te,
        device=device, balance_props=False))

    # Summary
    print(f"\n{'='*60}\n  COMPARISON\n{'='*60}")
    print(f"{'Name':<45} {'core7':>7} {'std':>7} {'lignin':>7}")
    print("-"*70)
    for r in results:
        lig = r['per_prop'].get('lignin_wt', float('nan'))
        print(f"{r['name']:<45} {r['avg_r2_core7']:>7.4f} {r['std_r2_core7']:>7.4f} {lig:>7.4f}")

    out = V5 / "results" / "asymmetric_head_comparison.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
