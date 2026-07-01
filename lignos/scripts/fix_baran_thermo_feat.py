#!/usr/bin/env python3
"""Fix: populate thermo_feat[:, 0:3] (T, time, IL_conc) for the 110 Baran-extended
lignin rows in LignoIL_unified/cached_*.npz.

Problem:
  extend_cache_with_unified_lignin.py wrote thermo_feat=zeros for the 110 new
  rows. Since the baseline PerPropHead uses t[:, :5] as its context (gate input +
  head input), those rows train with fake "mean" conditions — corrupting the
  gating signal for 11.6% of lignin data.

Fix:
  1. Match row-by-row against unified_lignin.csv (by order, since extend script
     preserves order for the new rows).
  2. Recover z-score transform (mu, sigma) for T, time, IL_conc from existing
     non-zero rows via regression on SMILES-overlapping pairs.
  3. Write z-scored values into thermo_feat[:, 0:3] for the affected rows.

Run this after extend_cache_with_unified_lignin.py (or whenever the unified
cache is regenerated). Backs up to cached_train.pre_thermo_fix.npz.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UNI_DIR = PROJECT_ROOT / "lignos" / "data" / "LignoIL_unified"
ORIG_DIR = PROJECT_ROOT / "lignos" / "data" / "LignoIL"


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


def derive_zscore_stats(raw_col, sm_canon_uni, sm_canon_cache, z_cache, lig_mask):
    """Fit (mu, sigma) by taking SMILES where BOTH have exactly one unique value
    — this gives clean (raw, z-scored) pairs without spurious alignments."""
    pairs = []
    for s_c in set(sm_canon_uni) & set(sm_canon_cache[lig_mask]):
        raws_sub = raw_col[sm_canon_uni == s_c].dropna().unique()
        zs_sub = np.unique(z_cache[(sm_canon_cache == s_c) & lig_mask].round(4))
        if len(raws_sub) == 1 and len(zs_sub) == 1:
            pairs.append((float(raws_sub[0]), float(zs_sub[0])))
    if len(pairs) < 3:
        return None, None, 0
    P = np.array(pairs)
    slope, intercept = np.polyfit(P[:, 0], P[:, 1], 1)
    if not np.isfinite(slope) or abs(slope) < 1e-9:
        return None, None, len(pairs)
    sigma = 1.0 / slope
    mu = -intercept * sigma
    return mu, sigma, len(pairs)


def main():
    # Load CSV + unified cache + original cache
    uni = pd.read_csv(UNI_DIR / "unified_lignin.csv")
    uni["smi_canon"] = uni["smiles"].map(canon)

    orig = np.load(ORIG_DIR / "cached_train.npz", allow_pickle=True)
    orig_sm = np.array([s.decode() if isinstance(s, bytes) else s for s in orig["smiles"]])
    orig_sm_canon = np.array([canon(s) for s in orig_sm])
    orig_t = orig["thermo_feat"]
    orig_y = orig["targets"]
    orig_lig = ~np.isnan(orig_y[:, 7])

    # Derive z-score stats for each of T, time, IL_conc
    print("Deriving z-score stats from SMILES overlap (unified CSV ↔ original cache)...")
    raw_cols = {
        0: ("temperature_C", uni["temperature_C"]),
        1: ("time_min", uni["time_min"]),
        2: ("il_conc_in_solvent_pct", uni["il_conc_in_solvent_pct"]),
    }
    stats = {}
    for idx, (name, raw_col) in raw_cols.items():
        mu, sigma, npairs = derive_zscore_stats(
            raw_col, uni["smi_canon"], orig_sm_canon, orig_t[:, idx], orig_lig
        )
        if mu is None:
            # fallback: use raw mean/std directly
            valid = raw_col.dropna()
            mu = float(valid.mean()) if len(valid) else 0.0
            sigma = float(valid.std()) if len(valid) else 1.0
            print(f"  col{idx} ({name}): FALLBACK μ={mu:.3f}, σ={sigma:.3f} (too few overlap pairs)")
        else:
            print(f"  col{idx} ({name}): μ={mu:.3f}, σ={sigma:.3f}  (from {npairs} pairs)")
        stats[idx] = (mu, sigma)

    # Now fix each split
    for split in ("train", "val", "test"):
        p = UNI_DIR / f"cached_{split}.npz"
        backup = UNI_DIR / f"cached_{split}.pre_thermo_fix.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        t = data["thermo_feat"].copy()
        sm_list = [s.decode() if isinstance(s, bytes) else s for s in data["smiles"]]
        sm_canon = np.array([canon(s) for s in sm_list])

        # Identify broken rows: thermo_feat all-zero AND lignin target present
        broken_mask = (t == 0).all(axis=1) & (~np.isnan(data["targets"][:, 7]))
        n_broken = int(broken_mask.sum())
        if n_broken == 0:
            print(f"  {split}: no broken rows found, skipping")
            continue
        if not backup.exists():
            np.savez(backup, **data)

        # Map broken rows → unified CSV entries
        # Strategy: for each broken row, find matching row in unified CSV
        # by SMILES + biomass_source + lignin target value (since those
        # are preserved through the extend script). Use first match.
        uni_small = uni.dropna(subset=["smi_canon", "lignin_value_pct"]).reset_index(drop=True)
        broken_idx = np.where(broken_mask)[0]
        filled = 0
        for row_i in broken_idx:
            s_c = sm_canon[row_i]
            # lignin_wt in cache is z-scored (ref_mean=45, ref_std=25 per extend script)
            lig_z = float(data["targets"][row_i, 7])
            lig_raw = lig_z * 25.0 + 45.0
            # Find best match: same SMILES + closest lignin_value_pct
            cand = uni_small[uni_small["smi_canon"] == s_c]
            if len(cand) == 0:
                continue
            diffs = (cand["lignin_value_pct"] - lig_raw).abs()
            best = cand.iloc[diffs.argmin()]
            # Z-score each of T, time, IL_conc using derived stats
            for idx, col_name in [(0, "temperature_C"), (1, "time_min"), (2, "il_conc_in_solvent_pct")]:
                raw_val = best[col_name]
                if pd.isna(raw_val):
                    continue  # leave as zero if raw is missing
                mu, sigma = stats[idx]
                t[row_i, idx] = (float(raw_val) - mu) / sigma
            filled += 1

        data["thermo_feat"] = t
        np.savez(p, **data)
        print(f"  {split}: fixed {filled}/{n_broken} rows (thermo_feat[:, 0:3] populated)")

    print("Done.")


if __name__ == "__main__":
    main()
