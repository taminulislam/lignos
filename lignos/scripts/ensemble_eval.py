#!/usr/bin/env python3
"""Ensemble evaluation with bootstrap confidence intervals.

Aggregates predictions from 10 trained seeds and computes:
- Prediction-averaged ensemble metrics
- Bootstrap 95% CIs (B=10,000)
- Paired Wilcoxon signed-rank test vs. v4 baseline
- Per-property breakdown

Usage:
    python ensemble_eval.py --config configs/v5_full.yaml
    python ensemble_eval.py --predictions_dir results/seed_predictions/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


PROPERTY_NAMES = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


def compute_r2(preds, targets):
    """Compute R² per property."""
    r2 = []
    for i in range(targets.shape[1]):
        ss_res = ((targets[:, i] - preds[:, i]) ** 2).sum()
        ss_tot = ((targets[:, i] - targets[:, i].mean()) ** 2).sum()
        r2.append(1 - ss_res / (ss_tot + 1e-8))
    return np.array(r2)


def bootstrap_ci(preds, targets, n_bootstrap=10000, confidence=0.95,
                 il_ids=None):
    """Compute bootstrap confidence intervals for R².

    Args:
        preds: (N, P) ensemble predictions
        targets: (N, P) ground truth
        n_bootstrap: number of bootstrap iterations
        confidence: confidence level (default 0.95)
        il_ids: (N,) IL identity for stratified sampling

    Returns:
        dict with mean, std, lower, upper for each property
    """
    n_samples = len(targets)
    alpha = 1 - confidence

    boot_r2 = np.zeros((n_bootstrap, targets.shape[1]))

    for b in range(n_bootstrap):
        if il_ids is not None:
            # Stratified bootstrap by IL identity
            unique_ils = np.unique(il_ids)
            idx = []
            for il in unique_ils:
                il_mask = il_ids == il
                il_indices = np.where(il_mask)[0]
                boot_idx = np.random.choice(il_indices, size=len(il_indices),
                                            replace=True)
                idx.extend(boot_idx)
            idx = np.array(idx)
        else:
            idx = np.random.choice(n_samples, size=n_samples, replace=True)

        boot_r2[b] = compute_r2(preds[idx], targets[idx])

    results = {}
    for i, name in enumerate(PROPERTY_NAMES):
        values = boot_r2[:, i]
        results[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "lower": float(np.percentile(values, 100 * alpha / 2)),
            "upper": float(np.percentile(values, 100 * (1 - alpha / 2))),
        }

    # Average R²
    avg_r2 = boot_r2.mean(axis=1)
    results["avg"] = {
        "mean": float(np.mean(avg_r2)),
        "std": float(np.std(avg_r2)),
        "lower": float(np.percentile(avg_r2, 100 * alpha / 2)),
        "upper": float(np.percentile(avg_r2, 100 * (1 - alpha / 2))),
    }

    return results


def wilcoxon_test(preds_v5, preds_v4, targets):
    """Paired Wilcoxon signed-rank test: v5 vs v4 per property.

    Tests whether v5 has significantly lower squared errors than v4.

    Returns:
        dict with p-value and significant flag per property
    """
    from scipy.stats import wilcoxon

    results = {}
    for i, name in enumerate(PROPERTY_NAMES):
        err_v5 = (targets[:, i] - preds_v5[:, i]) ** 2
        err_v4 = (targets[:, i] - preds_v4[:, i]) ** 2

        try:
            stat, pval = wilcoxon(err_v4, err_v5, alternative="greater")
            results[name] = {
                "statistic": float(stat),
                "p_value": float(pval),
                "significant": pval < 0.05,
            }
        except ValueError:
            results[name] = {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "note": "all differences zero",
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation")
    parser.add_argument("--predictions_dir", type=str,
                        default=str(V5_ROOT / "results/seed_predictions"))
    parser.add_argument("--v4_predictions", type=str, default=None,
                        help="Path to v4 ensemble predictions for comparison")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--output", type=str,
                        default=str(V5_ROOT / "results/ensemble_metrics.json"))
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)

    # Load seed predictions
    seed_files = sorted(pred_dir.glob("seed_*.npz"))
    if not seed_files:
        print(f"No seed predictions found in {pred_dir}")
        print("Run train_v5.py first to generate predictions.")
        return

    print(f"Loading {len(seed_files)} seed predictions")

    all_preds = []
    targets = None
    for f in seed_files:
        data = np.load(f)
        all_preds.append(data["predictions"])
        if targets is None:
            targets = data["targets"]

    all_preds = np.stack(all_preds)  # (n_seeds, N, P)

    # Prediction-averaged ensemble
    ensemble_preds = all_preds.mean(axis=0)  # (N, P)

    # Point estimates
    per_seed_r2 = np.array([compute_r2(p, targets) for p in all_preds])
    ensemble_r2 = compute_r2(ensemble_preds, targets)

    print(f"\nPer-seed R² (mean +/- std):")
    for i, name in enumerate(PROPERTY_NAMES):
        print(f"  {name:8s}: {per_seed_r2[:, i].mean():.4f} +/- {per_seed_r2[:, i].std():.4f}")
    print(f"  {'avg':8s}: {per_seed_r2.mean(axis=1).mean():.4f} +/- {per_seed_r2.mean(axis=1).std():.4f}")

    print(f"\nEnsemble R² (prediction averaging):")
    for i, name in enumerate(PROPERTY_NAMES):
        print(f"  {name:8s}: {ensemble_r2[i]:.4f}")
    print(f"  {'avg':8s}: {ensemble_r2.mean():.4f}")

    # Bootstrap CIs
    print(f"\nComputing bootstrap 95% CIs (B={args.n_bootstrap})...")
    il_ids = None  # Load if available
    cis = bootstrap_ci(ensemble_preds, targets, args.n_bootstrap, il_ids=il_ids)

    print(f"\nBootstrap 95% CIs:")
    for name in PROPERTY_NAMES + ["avg"]:
        ci = cis[name]
        print(f"  {name:8s}: {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    # Statistical comparison with v4
    comparison = None
    if args.v4_predictions and Path(args.v4_predictions).exists():
        v4_data = np.load(args.v4_predictions)
        v4_preds = v4_data["predictions"]
        comparison = wilcoxon_test(ensemble_preds, v4_preds, targets)
        print(f"\nWilcoxon test (v5 vs v4):")
        for name in PROPERTY_NAMES:
            r = comparison[name]
            sig = "*" if r["significant"] else ""
            print(f"  {name:8s}: p={r['p_value']:.4f} {sig}")

    # Save results
    results = {
        "n_seeds": len(seed_files),
        "n_samples": len(targets),
        "per_seed_r2": {
            name: {
                "mean": float(per_seed_r2[:, i].mean()),
                "std": float(per_seed_r2[:, i].std()),
                "values": per_seed_r2[:, i].tolist(),
            }
            for i, name in enumerate(PROPERTY_NAMES)
        },
        "ensemble_r2": {
            name: float(ensemble_r2[i])
            for i, name in enumerate(PROPERTY_NAMES)
        },
        "ensemble_avg_r2": float(ensemble_r2.mean()),
        "bootstrap_cis": cis,
    }
    if comparison:
        results["wilcoxon_vs_v4"] = comparison

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
