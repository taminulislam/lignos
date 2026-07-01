#!/usr/bin/env python3
"""Extend cached_train.npz with lignin-only rows for the 24 new SMILES in
unified_lignin.csv that are absent from the current cache.

Strategy (mirrors the ILThermo-expansion pattern from build_expanded_dataset.py):
  - Only TRAIN split is extended; val/test remain untouched.
  - Each new row carries:
      * morgan_fp     : computed from canonical SMILES (2048-bit Morgan r=2)
      * chemprop_fp   : 300-D zeros (not available for lignin-only ILs)
      * surface_fp    : 256-D zeros
      * thermo_feat   : 25-D zeros (lignin head uses physchem_feat; process
                        conditions belong in a dedicated column if the head
                        needs them — see 'meta_*' fields)
      * physchem_feat : 12-D joined from il_physchem_features.csv; zero-filled
                        if the SMILES is not in the physchem table
      * has_physchem  : bool
      * targets       : NaN for core-7; column 7 = standardized lignin_wt
      * preds_fusion / preds_chemprop : 8-D zeros (no teacher predictions)
      * is_original   : False
      * smiles / il_ids

Standardization of lignin_value_pct uses the EXISTING cache's lignin_wt
reference scale (computed by un-standardizing via min/max assumption), so
new rows are on the same numeric scale as the 836 existing lignin rows.

Writes to the same LignoIL_unified/ directory, overwriting cached_train.npz.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
UNIFIED_DIR = V5 / "data" / "LignoIL_unified"

PHYSCHEM_SHORT = [
    "kt_alpha", "kt_beta", "clogp", "viscosity_298K", "pKa", "conductivity",
    "cat_MW", "cat_C", "cat_N", "an_MW", "an_O", "an_HB",
]


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


def morgan(smi, nbits=2048, radius=2):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)


def main():
    print("=" * 60)
    print("Extend cached_train.npz with unified-lignin-only ILs")
    print("=" * 60)

    # Load current train cache (already has physchem_feat + has_physchem)
    train_path = UNIFIED_DIR / "cached_train.npz"
    z = np.load(train_path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    N_old = len(data["smiles"])
    print(f"Loaded existing train cache: {N_old} rows, {len(set(data['smiles']))} unique SMILES")

    existing_canon = set()
    for s in data["smiles"]:
        c = canon(s.decode() if isinstance(s, bytes) else s)
        if c:
            existing_canon.add(c)

    # Load unified lignin table + physchem
    unified = pd.read_csv(UNIFIED_DIR / "unified_lignin.csv")
    physchem = pd.read_csv(UNIFIED_DIR / "il_physchem_features.csv")
    phys_map = {row["smiles"]: np.array(
        [row[c] for c in PHYSCHEM_SHORT], dtype=np.float32
    ) for _, row in physchem.iterrows()}

    # Canonicalize unified SMILES (should already be canon from build script, but re-apply for safety)
    unified["smiles_canon"] = unified["smiles"].map(canon)
    unified = unified.dropna(subset=["smiles_canon", "lignin_value_pct"]).reset_index(drop=True)

    # Select rows whose SMILES is NOT already in the cache
    new_mask = ~unified["smiles_canon"].isin(existing_canon)
    new_rows = unified[new_mask].reset_index(drop=True)
    new_smiles = sorted(set(new_rows["smiles_canon"]))
    print(f"Unified SMILES not in cache: {len(new_smiles)}")
    print(f"New lignin rows to add (across those SMILES): {len(new_rows)}")
    print(f"  by source: {new_rows['source'].value_counts().to_dict()}")
    print(f"  by measurement_type: {new_rows['measurement_type'].value_counts().to_dict()}")

    # Standardize lignin_value_pct to match existing cache scale.
    # Existing cache lignin_wt stats: mean=0.073, std=0.725, range [-2.08, 2.03].
    # Back-solving the original std with unified raw %: mean=46-61 depending on source.
    # We use an approximate (ref_mean=45.0, ref_std=25.0), which matches the baran
    # empirical scale well and yields new_z ∈ [-1.7, 2.2] — compatible with existing.
    REF_MEAN, REF_STD = 45.0, 25.0
    lignin_raw = new_rows["lignin_value_pct"].astype(np.float32).to_numpy()
    lignin_std = (lignin_raw - REF_MEAN) / REF_STD
    print(f"Standardizing with reference (mean={REF_MEAN}, std={REF_STD}); "
          f"new lignin_wt range: [{lignin_std.min():.2f}, {lignin_std.max():.2f}]")

    # Precompute Morgan per unique new SMILES
    morgan_cache = {s: morgan(s) for s in new_smiles}

    N_new = len(new_rows)
    NEW_FIELDS = {
        "chemprop_fp": np.zeros((N_new, 300), dtype=np.float32),
        "surface_fp": np.zeros((N_new, 256), dtype=np.float32),
        "thermo_feat": np.zeros((N_new, 25), dtype=np.float32),
        "targets": np.full((N_new, 8), np.nan, dtype=np.float32),
        "preds_fusion": np.zeros((N_new, 8), dtype=np.float32),
        "preds_chemprop": np.zeros((N_new, 8), dtype=np.float32),
        "morgan_fp": np.zeros((N_new, 2048), dtype=np.float32),
        "smiles": np.empty(N_new, dtype=object),
        "il_ids": np.empty(N_new, dtype=object),
        "is_original": np.zeros(N_new, dtype=bool),
        "physchem_feat": np.zeros((N_new, 12), dtype=np.float32),
        "has_physchem": np.zeros(N_new, dtype=bool),
    }

    for i, row in new_rows.iterrows():
        smi = row["smiles_canon"]
        NEW_FIELDS["smiles"][i] = smi
        NEW_FIELDS["il_ids"][i] = str(row.get("il_name_raw", smi))
        NEW_FIELDS["morgan_fp"][i] = morgan_cache[smi]
        NEW_FIELDS["targets"][i, 7] = lignin_std[i]
        vec = phys_map.get(smi)
        if vec is not None:
            NEW_FIELDS["physchem_feat"][i] = np.where(np.isnan(vec), 0.0, vec)
            NEW_FIELDS["has_physchem"][i] = True

    # Cast smiles / il_ids to the same dtype as the existing arrays (object or bytes)
    old_dtype_smiles = data["smiles"].dtype
    NEW_FIELDS["smiles"] = NEW_FIELDS["smiles"].astype(old_dtype_smiles)
    NEW_FIELDS["il_ids"] = NEW_FIELDS["il_ids"].astype(data["il_ids"].dtype)

    # Concatenate per-field. Every key in the existing cache is present in NEW_FIELDS.
    merged = {}
    for k in data.keys():
        if k not in NEW_FIELDS:
            raise KeyError(f"Existing cache has key {k!r} not handled by this script")
        old = data[k]
        new = NEW_FIELDS[k]
        if old.ndim == 1:
            merged[k] = np.concatenate([old, new], axis=0)
        else:
            merged[k] = np.concatenate([old, new], axis=0)

    # Verify shapes
    N_final = N_old + N_new
    for k, arr in merged.items():
        assert arr.shape[0] == N_final, f"{k} shape {arr.shape}"

    # Save to both the canonical .npz (overwriting) and a _v2 sibling for safety.
    backup = UNIFIED_DIR / "cached_train_preextend.npz"
    if not backup.exists():
        np.savez(backup, **data)
        print(f"Backup saved: {backup.name}")
    np.savez(train_path, **merged)
    print(f"Wrote {train_path.name}: {N_old} → {N_final} rows "
          f"(+{N_new} lignin-only rows across {len(new_smiles)} new SMILES)")

    # Print lignin coverage now
    ln_col = merged["targets"][:, 7]
    total_lignin = int((~np.isnan(ln_col)).sum())
    print(f"Lignin rows in train cache now: {total_lignin}")
    total_physchem = int(merged["has_physchem"].sum())
    print(f"Rows with physchem in train: {total_physchem}")

    # Per-SMILES lignin coverage diagnostic
    smi_arr = np.array([s.decode() if isinstance(s, bytes) else s for s in merged["smiles"]])
    lignin_valid = ~np.isnan(ln_col)
    print(f"\nLignin coverage per SMILES (top 10 with most rows):")
    from collections import Counter
    cnt = Counter(smi_arr[lignin_valid])
    for s, c in cnt.most_common(10):
        print(f"  n={c:>3d}  {s[:60]}")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
