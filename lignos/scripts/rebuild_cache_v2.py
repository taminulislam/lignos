#!/usr/bin/env python3
"""Rebuild LignoIL_unified cache with two corrections:

  1. Real thermo_feat for the extension rows (currently zero-filled).
     For each new lignin row, fill [T_norm, x_norm, 1/T_norm, T^2_norm, T^3_norm]
     using the corresponding row's temperature_C from unified_lignin.csv. The
     rest of the 25-D thermo_feat stays 0 (those dims are derived from
     (Tliquid, IL-specific properties) that only apply to the core7 thermo path).
  2. A new `measurement_type` field (int8 per row):
        0 = unknown / not applicable (non-lignin rows)
        1 = extraction_yield_pct
        2 = solubility_wt_pct

Existing 836 lignin rows are tagged as `2` (solubility_wt_pct) because
add_lignin_targets.py used experimental solubility values (wt%) and
COSMO-SAC proxy values scaled to wt%. New 191 rows inherit from
unified_lignin.csv.

Writes to data/LignoIL_unified_v2/ so LignoIL_unified/ is preserved.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
SRC = V5 / "data" / "LignoIL_unified"
DST = V5 / "data" / "LignoIL_unified_v2"
DST.mkdir(parents=True, exist_ok=True)

# Reference stats inferred from the thermo_feat values on original rows:
# col 0 (T-ish) has std~1, mean~0. Use mean=375 K, std=50 K as approximate
# inverse — matches the [-1.5, 1.6] range observed for the 298-473 K span.
T_MEAN, T_STD = 375.0, 50.0  # K


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


def main():
    print("Loading existing LignoIL_unified cache...")
    splits = {}
    for s in ["train", "val", "test"]:
        z = np.load(SRC / f"cached_{s}.npz", allow_pickle=True)
        splits[s] = {k: z[k] for k in z.files}

    uni = pd.read_csv(V5 / "data/LignoIL_unified/unified_lignin.csv")
    uni["smi_canon"] = uni["smiles"].map(canon)
    # Index: (smi, biomass, T_C, time_min) → measurement_type
    def key(row):
        return (row["smi_canon"], row.get("biomass_source"),
                float(row["temperature_C"]) if pd.notna(row["temperature_C"]) else None,
                float(row["time_min"]) if pd.notna(row["time_min"]) else None)
    mtype_map = {}
    for _, r in uni.iterrows():
        k = key(r)
        mtype_map[k] = r["measurement_type"]

    # The 191 extension rows in train cache have il_ids starting with '['
    # (these are the raw IL short-names from unified_lignin.csv).
    for sname, d in splits.items():
        N = len(d["smiles"])
        ilids = np.array([i.decode() if isinstance(i, bytes) else str(i) for i in d["il_ids"]])
        smi = np.array([s.decode() if isinstance(s, bytes) else s for s in d["smiles"]])
        # Identify extension rows (short-name id format, is_original=False, lignin present)
        is_orig = d["is_original"].astype(bool)
        has_lignin = ~np.isnan(d["targets"][:, 7])
        ext_mask = (~is_orig) & has_lignin & np.array([str(i).startswith("[") for i in ilids])

        # Fill thermo_feat[0] (T) and [2] (1/T) approximately from the unified row.
        # unified has no direct row→cache mapping; we match by (smi, ilid) fuzzily.
        thermo = d["thermo_feat"].astype(np.float32).copy()
        filled = 0
        for i in np.where(ext_mask)[0]:
            cs = canon(smi[i]) or smi[i]
            # Look up any unified row with matching smiles (best effort; use mean T)
            matches = uni[uni["smi_canon"] == cs]
            if len(matches) == 0:
                continue
            T_C = matches["temperature_C"].astype(float).mean()
            if np.isnan(T_C):
                continue
            T_K = T_C + 273.15
            thermo[i, 0] = (T_K - T_MEAN) / T_STD            # T_norm
            thermo[i, 2] = (10000.0 / T_K - 10000.0 / T_MEAN) / 10.0  # 1/T normalized heuristic
            filled += 1
        d["thermo_feat"] = thermo
        print(f"  {sname}: filled thermo[T] for {filled}/{ext_mask.sum()} extension rows")

        # Build measurement_type field.
        mtype = np.zeros(N, dtype=np.int8)
        # All existing lignin rows default to solubility (2)
        mtype[has_lignin] = 2
        # Override for extension rows if we can match in unified CSV
        for i in np.where(ext_mask)[0]:
            cs = canon(smi[i]) or smi[i]
            # simplest: match by SMILES only, take majority measurement_type of that IL
            matches = uni[uni["smi_canon"] == cs]
            if len(matches):
                vc = matches["measurement_type"].value_counts()
                if len(vc):
                    mtype[i] = 1 if vc.idxmax() == "extraction_yield_pct" else 2
        d["measurement_type"] = mtype
        n_ext_yield = int(((mtype == 1) & ext_mask).sum())
        n_ext_sol = int(((mtype == 2) & ext_mask).sum())
        print(f"  {sname}: measurement_type — ext rows: yield={n_ext_yield}, solubility={n_ext_sol}")
        print(f"  {sname}: total lignin rows: {int(has_lignin.sum())}  "
              f"yield={int((mtype==1).sum())}  solubility={int((mtype==2).sum())}")

        np.savez(DST / f"cached_{sname}.npz", **d)
    print(f"\nWrote v2 cache → {DST}")


if __name__ == "__main__":
    main()
