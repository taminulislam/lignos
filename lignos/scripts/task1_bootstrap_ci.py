"""Bootstrap 95% CI on Task-1 lignin R² across all Task-1 baselines.

For each method with 10-seed per-seed R² values stored on disk, resample
the 10 seeds with replacement B=10,000 times and report
    mean, seed-level 95% bootstrap CI, seed std, per-seed range.

Seed-level bootstrap quantifies *initialization uncertainty* over the
fixed 39-row Task-1 test set. Row-level bootstrap (resampling the 39
test rows) would quantify *data-split uncertainty* but requires per-row,
per-seed predictions which are not stored for the LIGNOS +#5+#6 run;
we note this as a scope limitation of the present CI analysis.

Outputs a clean Markdown table + JSON to results/task1_bootstrap_ci.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
B = 10_000

# (display name, JSON path, per-seed field, static fallback values if field missing)
METHODS = [
    ("Baran GB (published)",
     None, None, [0.440]),  # single published point — no per-seed
    ("Random Forest",
     None, None, [0.462]),
    ("XGBoost (full LIGNOS stack)",
     R / "lignos_xgb_full_stack_task1.json", "per_seed_r2", None),
    ("Chemprop D-MPNN (from-scratch)",
     R / "lignos_chemprop_task1.json", "per_seed_r2", None),
    ("ChemBERTa-77M (FT, two-phase)",
     R / "lignos_chembert_task1.json", "per_seed_r2", None),
    ("Chemprop pretrained + FT",
     R / "lignos_chemprop_pretrained_task1.json", "per_seed_r2", None),
    ("A2 two-stage",
     None, None, [0.706]),
    ("LIGNOS Stage 1 (BMA-fused)",
     None, None, [0.536]),
    ("LIGNOS Stage 2 baseline",
     R / "a5_bma_tier2_merged.json", "tier2_mu0_aug0:lignin_per_seed", None),
    ("LIGNOS + #5 (pred-mu feeding)",
     R / "a5_bma_tier2_merged.json", "tier2_mu1_aug0:lignin_per_seed", None),
    ("LIGNOS + #6 (input jitter)",
     R / "a5_bma_tier2_merged.json", "tier2_mu0_aug1:lignin_per_seed", None),
    ("**LIGNOS + #5 + #6 (full method)**",
     R / "a5_bma_tier2_merged.json", "tier2_mu1_aug1:lignin_per_seed", None),
]


def load_seeds(path, field, fallback):
    if path is None:
        return np.array(fallback, dtype=np.float64)
    if not path.exists():
        print(f"[warn] missing {path} — fallback")
        return np.array(fallback or [np.nan], dtype=np.float64)
    d = json.load(open(path))
    if field is None:
        return np.array(fallback or [np.nan], dtype=np.float64)
    if ":" in field:
        k1, k2 = field.split(":", 1)
        arr = d[k1][k2]
    else:
        arr = d[field]
    return np.array(arr, dtype=np.float64)


def bootstrap_ci(seeds, B=B, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    if len(seeds) < 2:
        return float(seeds.mean()), float("nan"), float("nan"), float("nan")
    n = len(seeds)
    idx = rng.integers(0, n, size=(B, n))
    boot_means = seeds[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return float(seeds.mean()), float(seeds.std(ddof=0)), lo, hi


def main():
    print(f"\nTask 1 Lignin R² — seed-level bootstrap 95% CI (B={B})\n")
    print(f"{'Method':<42}  {'n_seed':>6}  {'Mean':>7}  {'Std':>6}  {'95% CI':>19}  {'Range':>17}")
    print("-" * 105)
    rows = []
    for name, path, field, fallback in METHODS:
        seeds = load_seeds(path, field, fallback)
        mean, std, lo, hi = bootstrap_ci(seeds)
        seeds_valid = seeds[np.isfinite(seeds)]
        if len(seeds_valid) >= 1:
            rmin, rmax = float(seeds_valid.min()), float(seeds_valid.max())
        else:
            rmin, rmax = float("nan"), float("nan")
        rows.append({
            "method": name, "n_seeds": int(len(seeds)),
            "mean": mean, "std": std,
            "ci95_lo": lo, "ci95_hi": hi,
            "range": [rmin, rmax],
            "per_seed_r2": seeds.tolist(),
        })
        n_lbl = f"{len(seeds)}"
        ci_str = (f"[{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else "       ---       ")
        range_str = f"[{rmin:+.3f}, {rmax:+.3f}]"
        print(f"{name:<42}  {n_lbl:>6}  {mean:>+7.3f}  {std:>6.3f}  {ci_str:>19}  {range_str:>17}")

    out_path = R / "task1_bootstrap_ci.json"
    json.dump({"B": B, "alpha": 0.05, "methods": rows}, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
