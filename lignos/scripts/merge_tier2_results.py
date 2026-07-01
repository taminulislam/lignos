"""Merge Tier 2 per-seed results from multiple run logs into one JSON.

pt1 (17797222) ran mu0_aug0 + mu0_aug1 fully + 4 seeds of mu1_aug0 before timeout.
pt2 (17802217) resumed and ran mu1_aug0 10/10 + mu1_aug1 9/10 before timeout.
seed9 fill-in job ran mu1_aug1 seed 9 only.

This script parses the stdout logs (line format:
  `  seed N: core7=X  lignin=Y  g50=Z`
  `--- Config tier2_muM_augA (...) ---`
), dedupes by (config, seed) taking the latest occurrence, and emits
`results/a5_bma_tier2_merged.json` with per-config mean/std + per-seed lists.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

LOG_DIR = Path("/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/jobs/logs")
RESULTS_DIR = Path("/work/nvme/bgte/kahmed2/Dataset_Chemistry/lignos/results")

DEFAULT_LOGS = [
    "a5_tier2_17797222.out",     # pt1
    "a5_tier2b_17802217.out",    # pt2
    # seed9 fill-in log is autodetected by glob below.
]

CFG_RE = re.compile(r"^--- Config (tier2_mu\d_aug\d)")
SEED_RE = re.compile(
    r"^\s*seed\s+(\d+):\s+core7=([\d.]+)\s+lignin=([\d.]+)\s+g50=([\d.]+)"
)


def parse_log(path: Path) -> list[tuple[str, int, float, float, float]]:
    rows: list[tuple[str, int, float, float, float]] = []
    current_cfg: str | None = None
    for line in path.read_text().splitlines():
        m = CFG_RE.match(line)
        if m:
            current_cfg = m.group(1)
            continue
        m = SEED_RE.match(line)
        if m and current_cfg is not None:
            rows.append(
                (current_cfg, int(m.group(1)),
                 float(m.group(2)), float(m.group(3)), float(m.group(4)))
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", default=None,
                    help="Log filenames (under jobs/logs/). Default autodetects.")
    ap.add_argument("--out", default="a5_bma_tier2_merged.json")
    args = ap.parse_args()

    logs = args.logs or (DEFAULT_LOGS + sorted(p.name for p in
                                               LOG_DIR.glob("a5_tier2_s9_*.out")))
    # dedupe (cfg, seed) -> row, later logs overwrite earlier.
    store: dict[tuple[str, int], tuple[float, float, float]] = {}
    for name in logs:
        path = LOG_DIR / name
        if not path.exists():
            print(f"[warn] missing log: {path}")
            continue
        for cfg, seed, c7, lig, g50 in parse_log(path):
            store[(cfg, seed)] = (c7, lig, g50)
        print(f"  parsed {path.name}")

    # regroup by config
    by_cfg: dict[str, list[tuple[int, float, float, float]]] = {}
    for (cfg, seed), vals in store.items():
        by_cfg.setdefault(cfg, []).append((seed, *vals))
    for cfg in by_cfg:
        by_cfg[cfg].sort()  # by seed

    merged: dict[str, dict] = {}
    for cfg, rows in by_cfg.items():
        seeds = [r[0] for r in rows]
        c7 = np.array([r[1] for r in rows])
        lig = np.array([r[2] for r in rows])
        g50 = np.array([r[3] for r in rows])
        merged[cfg] = {
            "n_seeds": len(rows),
            "seed_ids": seeds,
            "core7_mean": float(c7.mean()),
            "core7_std": float(c7.std()),
            "lignin_mean": float(lig.mean()),
            "lignin_std": float(lig.std()),
            "g50_mean": float(g50.mean()),
            "g50_std": float(g50.std()),
            "lignin_per_seed": lig.tolist(),
        }

    out_path = RESULTS_DIR / args.out
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"\nWrote {out_path}")

    print(f"\n{'='*70}\nTier 2 merged summary (ranked by lignin)\n{'='*70}")
    print(f"  {'config':<22}{'n':>4}{'core7':>10}{'lignin':>12}{'std':>8}{'g50':>10}")
    ranked = sorted(merged.items(), key=lambda kv: kv[1]["lignin_mean"], reverse=True)
    for cfg, r in ranked:
        print(f"  {cfg:<22}{r['n_seeds']:>4}{r['core7_mean']:>10.4f}"
              f"{r['lignin_mean']:>12.4f}{r['lignin_std']:>8.4f}{r['g50_mean']:>10.4f}")


if __name__ == "__main__":
    main()
