"""Merge per-fold JSONs from `compare_a59_bma4_mahal_baran.py --fold k` runs
into a single aggregate at `results/lignos_bma4_mahal_task2.json`."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
RESULTS = V5 / "results"


def main():
    fold_files = sorted(RESULTS.glob("lignos_bma4_mahal_fold_*.json"))
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

    def _agg(path):
        xs = [f[path[0]][path[1]] for f in folds]
        return float(np.mean(xs)), float(np.std(xs))

    ens_m, ens_s = _agg(("ensemble", "r2_all"))
    k4_m, k4_s = _agg(("k4_bma", "r2_all"))
    mg_m, mg_s = _agg(("k4_mahal", "r2_all"))
    ba_m, ba_s = _agg(("baran_alone", "r2"))

    print(f"\n{'='*70}")
    print("Baran Task 2 — LIGNOS + BMA-K4 (Pick #1) + Mahalanobis gate (Pick #3)")
    print(f"{'='*70}")
    print(f"A5.9 ensemble ALL        : R² = {ens_m:+.4f} ± {ens_s:.4f}")
    print(f"+ Baran BMA pillar ALL   : R² = {k4_m:+.4f} ± {k4_s:.4f}  "
          f"(Δvs_ens={k4_m-ens_m:+.4f})")
    print(f"+ Mahalanobis gate ALL   : R² = {mg_m:+.4f} ± {mg_s:.4f}  "
          f"(Δvs_ens={mg_m-ens_m:+.4f})")
    print(f"Baran alone (this run)   : R² = {ba_m:+.4f} ± {ba_s:.4f}")
    print(f"Baran GB (published 5CV) : R² = +0.5238 ± 0.2015")

    print("\nPer-fold breakdown:")
    print(f"{'fold':>4}  {'n':>3}  {'ens':>8}  {'k4_bma':>8}  {'k4+mahal':>9}  "
          f"{'Baran':>8}  {'w_D(lig)':>9}  {'n_ood':>6}  held_out")
    for f in folds:
        hk = ",".join(f["held_out_ils"][:2])[:34]
        print(f"{f['fold']:>4}  {f['n']:>3}  "
              f"{f['ensemble']['r2_all']:>+8.4f}  "
              f"{f['k4_bma']['r2_all']:>+8.4f}  "
              f"{f['k4_mahal']['r2_all']:>+9.4f}  "
              f"{f['baran_alone']['r2']:>+8.4f}  "
              f"{f['k4_bma']['mean_w_baran_lignin']:>9.3f}  "
              f"{f['k4_mahal']['n_ood']:>6d}  {hk}")

    out = RESULTS / "lignos_bma4_mahal_task2.json"
    json.dump({
        "folds": folds,
        "ensemble_r2_mean": ens_m, "ensemble_r2_std": ens_s,
        "k4_bma_r2_mean": k4_m, "k4_bma_r2_std": k4_s,
        "k4_mahal_r2_mean": mg_m, "k4_mahal_r2_std": mg_s,
        "baran_alone_r2_mean": ba_m, "baran_alone_r2_std": ba_s,
        "n_folds": len(folds),
    }, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
