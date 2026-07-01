#!/usr/bin/env python3
"""Compare SimCLR vs V-JEPA pre-training for COSMOBridge v5.

Reads test results from both training runs and produces:
1. Head-to-head R² comparison table
2. Per-property bar chart
3. Paired Wilcoxon test for statistical significance
4. Radar chart overlay

Usage:
    python compare_pretraining.py
    python compare_pretraining.py --simclr_dir results/seed_predictions --vjepa_dir results/vjepa/seed_predictions
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


def load_seed_results(pred_dir):
    """Load predictions from all seeds in a directory."""
    pred_dir = Path(pred_dir)
    seed_files = sorted(pred_dir.glob("seed_*.npz"))

    if not seed_files:
        return None, None, None

    all_preds = []
    targets = None
    for f in seed_files:
        data = np.load(f)
        all_preds.append(data["predictions"])
        if targets is None:
            targets = data["targets"]

    return np.stack(all_preds), targets, len(seed_files)


def compute_r2(preds, targets):
    """Compute R² per property."""
    r2 = []
    for i in range(targets.shape[1]):
        ss_res = ((targets[:, i] - preds[:, i]) ** 2).sum()
        ss_tot = ((targets[:, i] - targets[:, i].mean()) ** 2).sum()
        r2.append(1 - ss_res / (ss_tot + 1e-8))
    return np.array(r2)


def main():
    parser = argparse.ArgumentParser(description="Compare pre-training methods")
    parser.add_argument("--simclr_dir", type=str,
                        default=str(V5_ROOT / "results/seed_predictions"))
    parser.add_argument("--vjepa_dir", type=str,
                        default=str(V5_ROOT / "results/vjepa/seed_predictions"))
    parser.add_argument("--v4_predictions", type=str, default=None,
                        help="Optional v4 baseline predictions")
    parser.add_argument("--output_dir", type=str,
                        default=str(V5_ROOT / "results/comparison"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    methods = {}

    simclr_preds, simclr_targets, n_simclr = load_seed_results(args.simclr_dir)
    if simclr_preds is not None:
        methods["SimCLR"] = (simclr_preds, simclr_targets, n_simclr)
        print(f"SimCLR: {n_simclr} seeds, {simclr_targets.shape[0]} test samples")

    vjepa_preds, vjepa_targets, n_vjepa = load_seed_results(args.vjepa_dir)
    if vjepa_preds is not None:
        methods["V-JEPA"] = (vjepa_preds, vjepa_targets, n_vjepa)
        print(f"V-JEPA: {n_vjepa} seeds, {vjepa_targets.shape[0]} test samples")

    if len(methods) < 2:
        print(f"Need at least 2 methods to compare. Found: {list(methods.keys())}")
        print("Run both training pipelines first.")
        return

    # ── Compute per-seed R² ──
    results = {}
    for name, (preds, targets, n_seeds) in methods.items():
        per_seed_r2 = np.array([compute_r2(preds[i], targets) for i in range(n_seeds)])
        ensemble_preds = preds.mean(axis=0)
        ensemble_r2 = compute_r2(ensemble_preds, targets)

        results[name] = {
            "per_seed": per_seed_r2,       # (n_seeds, 7)
            "ensemble": ensemble_r2,        # (7,)
            "n_seeds": n_seeds,
        }

    # ── Print comparison table ──
    print(f"\n{'='*80}")
    print("HEAD-TO-HEAD COMPARISON: SimCLR vs V-JEPA Pre-training")
    print(f"{'='*80}")

    header = f"{'Property':>10s}"
    for name in methods:
        header += f"  {name + ' (mean±std)':>22s}  {name + ' (ens)':>12s}"
    print(header)
    print("-" * 80)

    for i, prop in enumerate(PROPERTY_NAMES):
        row = f"{prop:>10s}"
        for name in methods:
            r = results[name]
            mean = r["per_seed"][:, i].mean()
            std = r["per_seed"][:, i].std()
            ens = r["ensemble"][i]
            row += f"  {mean:>6.4f}±{std:.3f}  {ens:>10.4f}"
        print(row)

    # Average
    row = f"{'avg':>10s}"
    for name in methods:
        r = results[name]
        avg_per_seed = r["per_seed"].mean(axis=1)
        row += f"  {avg_per_seed.mean():>6.4f}±{avg_per_seed.std():.3f}  {r['ensemble'].mean():>10.4f}"
    print(row)

    # ── Statistical test ──
    if "SimCLR" in results and "V-JEPA" in results:
        from scipy.stats import wilcoxon

        print(f"\n{'='*60}")
        print("PAIRED WILCOXON TEST (per-seed avg R²)")
        print(f"{'='*60}")

        simclr_avgs = results["SimCLR"]["per_seed"].mean(axis=1)
        vjepa_avgs = results["V-JEPA"]["per_seed"].mean(axis=1)

        n = min(len(simclr_avgs), len(vjepa_avgs))
        try:
            stat, pval = wilcoxon(vjepa_avgs[:n], simclr_avgs[:n], alternative="greater")
            sig = "*" if pval < 0.05 else ""
            print(f"  V-JEPA > SimCLR: p={pval:.4f} {sig}")
            print(f"  SimCLR avg R²: {simclr_avgs.mean():.4f} ± {simclr_avgs.std():.4f}")
            print(f"  V-JEPA avg R²: {vjepa_avgs.mean():.4f} ± {vjepa_avgs.std():.4f}")
            print(f"  Difference: {vjepa_avgs.mean() - simclr_avgs.mean():+.4f}")
        except ValueError as e:
            print(f"  Cannot compute: {e}")

    # ── Generate plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(PROPERTY_NAMES) + 1)
        width = 0.35

        colors = {"SimCLR": "#2196F3", "V-JEPA": "#F44336"}

        for idx, (name, r) in enumerate(results.items()):
            means = list(r["per_seed"].mean(axis=0)) + [r["per_seed"].mean()]
            stds = list(r["per_seed"].std(axis=0)) + [r["per_seed"].mean(axis=1).std()]
            offset = (idx - 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=stds,
                         label=name, color=colors.get(name, f"C{idx}"),
                         alpha=0.8, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(PROPERTY_NAMES + ["avg"], fontsize=11)
        ax.set_ylabel("Test R²", fontsize=12)
        ax.set_title("SimCLR vs V-JEPA Pre-training: Per-Property Test R²",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.axhline(0.81, color="gray", linestyle="--", alpha=0.5, label="v4 baseline")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "simclr_vs_vjepa_comparison.png", dpi=200)
        plt.close(fig)
        print(f"\nSaved: {output_dir / 'simclr_vs_vjepa_comparison.png'}")

        # Radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(PROPERTY_NAMES), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        for name, r in results.items():
            values = list(r["ensemble"]) + [r["ensemble"][0]]
            ax.plot(angles, values, "o-", linewidth=2, label=name,
                   color=colors.get(name, "gray"))
            ax.fill(angles, values, alpha=0.1, color=colors.get(name, "gray"))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(PROPERTY_NAMES, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_title("Ensemble R² by Property", fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
        plt.tight_layout()
        fig.savefig(output_dir / "radar_simclr_vs_vjepa.png", dpi=200)
        plt.close(fig)
        print(f"Saved: {output_dir / 'radar_simclr_vs_vjepa.png'}")

    except ImportError:
        print("matplotlib not available for plots")

    # Save results JSON
    save_results = {}
    for name, r in results.items():
        save_results[name] = {
            "per_property_mean": {p: float(r["per_seed"][:, i].mean())
                                  for i, p in enumerate(PROPERTY_NAMES)},
            "per_property_std": {p: float(r["per_seed"][:, i].std())
                                 for i, p in enumerate(PROPERTY_NAMES)},
            "ensemble_r2": {p: float(r["ensemble"][i])
                           for i, p in enumerate(PROPERTY_NAMES)},
            "avg_r2_mean": float(r["per_seed"].mean()),
            "avg_r2_std": float(r["per_seed"].mean(axis=1).std()),
            "n_seeds": r["n_seeds"],
        }

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"Saved: {output_dir / 'comparison_results.json'}")


if __name__ == "__main__":
    main()
