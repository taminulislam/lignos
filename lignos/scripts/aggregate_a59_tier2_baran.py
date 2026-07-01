"""Merge per-fold JSONs from `compare_a59_tier2_vs_baran.py --fold k` runs
into a single aggregate at `results/a59_tier2_baran_task2.json`, matching the
shape of the non-array run."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
RESULTS = V5 / "results"


def main():
    fold_files = sorted(RESULTS.glob("a59_tier2_baran_fold_*.json"))
    if not fold_files:
        print("No per-fold files found. Did the array job finish?")
        return

    folds = []
    for fp in fold_files:
        d = json.load(open(fp))
        if d.get("result") is not None:
            folds.append(d["result"])
        print(f"  loaded {fp.name}  fold {d.get('fold')}")

    if not folds:
        print("All folds empty; nothing to aggregate.")
        return

    def _agg(key):
        xs = [f[key]["r2_all"] for f in folds]
        return float(np.mean(xs)), float(np.std(xs))
    ens_m, ens_s = _agg("ensemble")
    t2_m, t2_s = _agg("tier2")

    print(f"\n{'='*70}\nBaran Task 2 — A5.9 vs +Tier 2 Stage-2 #5 (aggregated)\n{'='*70}")
    print(f"A2 baseline (no gate)       : R² = -0.41 ± 1.04")
    print(f"A5.9 ensemble ALL           : R² = {ens_m:+.4f} ± {ens_s:.4f}")
    print(f"A5.9 +Tier 2 #5 ALL         : R² = {t2_m:+.4f} ± {t2_s:.4f}  "
          f"(Δ={t2_m - ens_m:+.4f})")
    for q_key, label in [("0.25", "25%"), ("0.5", "50%"), ("0.75", "75%")]:
        ens_vals = [f["ensemble"]["r2_gated"][q_key]["r2"]
                    for f in folds if f["ensemble"]["r2_gated"][q_key]]
        t2_vals = [f["tier2"]["r2_gated"][q_key]["r2"]
                   for f in folds if f["tier2"]["r2_gated"][q_key]]
        if ens_vals:
            print(f"  gated@{label:3s}  ensemble: {np.mean(ens_vals):+.4f}  "
                  f"+Tier2: {np.mean(t2_vals):+.4f}  "
                  f"(Δ={np.mean(t2_vals)-np.mean(ens_vals):+.4f})")
    print(f"Baran GB (their own CV)     : R² = +0.52 ± 0.20")

    out = RESULTS / "a59_tier2_baran_task2.json"
    json.dump({"folds": folds,
                "ensemble_r2_mean": ens_m, "ensemble_r2_std": ens_s,
                "tier2_r2_mean": t2_m, "tier2_r2_std": t2_s,
                "n_folds": len(folds)}, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
