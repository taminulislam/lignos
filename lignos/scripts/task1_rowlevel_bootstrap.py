"""Row-level bootstrap 95% CI on Task-1 lignin R² for LIGNOS +#5+#6.

Consumes the per-seed per-row lignin prediction matrix produced by
`train_a5_bma_tier2.py --save-rowpreds ...`. For each of 10 seeds we have a
39-row prediction vector; the ensemble prediction is the seed-average.

Two bootstrap flavours are reported:

  1. Row-level bootstrap on the seed-averaged predictions. Resamples the
     39 test rows with replacement; quantifies test-set-sampling
     uncertainty for the ensemble.
  2. Row-level bootstrap per seed, then across-seed aggregate. Resamples
     rows and seeds jointly; quantifies the combined uncertainty.
  3. Seed-level bootstrap for reference (matches the Table-4 CI).

All with B=10,000.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
B = 10_000


def r2(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 2:
        return float("nan")
    yk, pk = y[ok], p[ok]
    ss_res = ((yk - pk) ** 2).sum()
    ss_tot = ((yk - yk.mean()) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)


def main():
    src = R / "task1_tier2_mu1_aug1_rowpreds.npz"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing {src}. Run:\n"
            f"  sbatch lignos/jobs/slurm_task1_rowpreds.sh\n"
            f"first to generate per-seed per-row lignin predictions.")
    d = np.load(src, allow_pickle=True)
    preds = d["lignin_preds"].astype(np.float64)  # (n_seeds, n_test)
    y_true = d["y_true"].astype(np.float64)       # (n_test,)
    seeds = d["seeds"]
    n_seeds, n_test = preds.shape
    print(f"Loaded: {n_seeds} seeds × {n_test} test rows  "
          f"(config={str(d.get('config', 'unknown'))})")

    # Valid row mask (filter NaN labels, if any)
    ok = np.isfinite(y_true)
    y_true_v = y_true[ok]
    preds_v = preds[:, ok]
    n_valid = int(ok.sum())
    print(f"Valid test rows: {n_valid}")

    # Ensemble prediction (seed-averaged)
    pred_ens = preds_v.mean(axis=0)
    r2_ens = r2(y_true_v, pred_ens)
    print(f"\nEnsemble R² on seed-averaged preds: {r2_ens:+.4f}")

    # Per-seed R²
    seed_r2s = np.array([r2(y_true_v, preds_v[s]) for s in range(n_seeds)])
    print(f"Per-seed R²: min={seed_r2s.min():+.4f}  "
          f"mean={seed_r2s.mean():+.4f}  max={seed_r2s.max():+.4f}  "
          f"std={seed_r2s.std():.4f}")

    rng = np.random.default_rng(42)

    # ---- (1) Row-level bootstrap on seed-averaged predictions ----
    boot_r2_row = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n_valid, size=n_valid)
        boot_r2_row[b] = r2(y_true_v[idx], pred_ens[idx])
    lo_row = float(np.quantile(boot_r2_row, 0.025))
    hi_row = float(np.quantile(boot_r2_row, 0.975))
    print(f"\n(1) Row-level bootstrap on ensemble:")
    print(f"    R² = {r2_ens:+.4f}  95% CI [{lo_row:+.4f}, {hi_row:+.4f}]  "
          f"(B={B}, resample {n_valid} rows)")

    # ---- (2) Joint row + seed bootstrap ----
    rng2 = np.random.default_rng(123)
    boot_r2_joint = np.empty(B, dtype=np.float64)
    for b in range(B):
        row_idx = rng2.integers(0, n_valid, size=n_valid)
        seed_idx = rng2.integers(0, n_seeds, size=n_seeds)
        p_boot = preds_v[seed_idx][:, row_idx].mean(axis=0)
        boot_r2_joint[b] = r2(y_true_v[row_idx], p_boot)
    lo_j = float(np.quantile(boot_r2_joint, 0.025))
    hi_j = float(np.quantile(boot_r2_joint, 0.975))
    print(f"\n(2) Joint row+seed bootstrap (resample both):")
    print(f"    R² CI = [{lo_j:+.4f}, {hi_j:+.4f}]  "
          f"(B={B}, resample {n_valid} rows × {n_seeds} seeds)")

    # ---- (3) Seed-level bootstrap (matches Table-4) ----
    rng3 = np.random.default_rng(42)
    boot_r2_seed = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng3.integers(0, n_seeds, size=n_seeds)
        boot_r2_seed[b] = seed_r2s[idx].mean()
    lo_s = float(np.quantile(boot_r2_seed, 0.025))
    hi_s = float(np.quantile(boot_r2_seed, 0.975))
    print(f"\n(3) Seed-level bootstrap (matches Table 4):")
    print(f"    R² = {seed_r2s.mean():+.4f}  95% CI [{lo_s:+.4f}, {hi_s:+.4f}]")

    # Width comparison
    w_seed = hi_s - lo_s
    w_row = hi_row - lo_row
    w_joint = hi_j - lo_j
    print(f"\nCI widths:  seed-level={w_seed:.4f}  "
          f"row-level={w_row:.4f} ({w_row/w_seed:.2f}×)  "
          f"joint={w_joint:.4f} ({w_joint/w_seed:.2f}×)")

    out = {
        "n_seeds": int(n_seeds), "n_test_valid": n_valid,
        "r2_ensemble_avg_preds": float(r2_ens),
        "r2_per_seed_mean": float(seed_r2s.mean()),
        "r2_per_seed_std": float(seed_r2s.std()),
        "row_level_ci95": [lo_row, hi_row],
        "joint_row_seed_ci95": [lo_j, hi_j],
        "seed_level_ci95": [lo_s, hi_s],
        "B": B,
    }
    out_path = R / "task1_rowlevel_bootstrap.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
