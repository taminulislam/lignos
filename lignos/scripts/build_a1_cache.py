"""Build the A1 cache: data/LignoIL base (0.833 floor) + physchem backfill +
P-column denoise. Writes to data/LignoIL_A1/.

Operations:
  1. Copy data/LignoIL/cached_{train,val,test}.npz → data/LignoIL_A1/
  2. Add physchem_feat + has_physchem by looking up canonical SMILES against
     LignoIL_unified/il_physchem_features.csv (which covers 52/52 LignoIL ILs
     after 2026-04-18 fill).
  3. NaN-out the P target for rows whose SMILES is NOT in the 19 "original"
     SMILES from data/expanded/ (those 19 have the clean 152-row P distribution
     in [-0.564, 4.349]; the other 1,739 rows have pollution from density data
     z-scored differently, mean -0.311).
"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
SRC = V5 / "data" / "LignoIL"
DST = V5 / "data" / "LignoIL_A1"
DST.mkdir(exist_ok=True, parents=True)

PHYSCHEM_CSV = V5 / "data" / "LignoIL_unified" / "il_physchem_features.csv"
EXPANDED = V5 / "data" / "expanded"

PHYSCHEM_SHORT = [
    "kt_alpha", "kt_beta", "clogp", "viscosity_298K", "pKa", "conductivity",
    "cat_MW", "cat_C", "cat_N", "an_MW", "an_O", "an_HB",
]


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


def main():
    # Load physchem map
    phys = pd.read_csv(PHYSCHEM_CSV)
    phys_map = {}
    for _, row in phys.iterrows():
        c = canon(row["smiles"])
        if c is None:
            continue
        phys_map[c] = np.array([row[col] for col in PHYSCHEM_SHORT], dtype=np.float32)

    # Pre-process physchem: log-transform viscosity (idx 3) and conductivity (idx 5),
    # z-score on the 52-row table itself (so A1 is self-contained).
    phys_mat = np.stack(list(phys_map.values()))
    phys_mat[:, 3] = np.log1p(np.maximum(phys_mat[:, 3], 0.0))
    phys_mat[:, 5] = np.log1p(np.maximum(phys_mat[:, 5], 0.0))
    mu = phys_mat.mean(0)
    sd = phys_mat.std(0) + 1e-6
    phys_z_map = {}
    for s, v in phys_map.items():
        v2 = v.copy()
        v2[3] = np.log1p(max(v2[3], 0.0))
        v2[5] = np.log1p(max(v2[5], 0.0))
        phys_z_map[s] = ((v2 - mu) / sd).astype(np.float32)
    print(f"Physchem map: {len(phys_z_map)} canon SMILES, mu/sd fit across 52 rows.")

    # The 19 "clean" P SMILES
    ep = np.load(EXPANDED / "cached_train.npz", allow_pickle=True)
    ep_sm = [s.decode() if isinstance(s, bytes) else s for s in ep["smiles"]]
    ep_p_mask = ~np.isnan(ep["targets"][:, 6])
    clean_p_smiles = set(s for s, m in zip(ep_sm, ep_p_mask) if m)
    print(f"Clean P SMILES (from expanded cache): {len(clean_p_smiles)}")

    for split in ("train", "val", "test"):
        src_path = SRC / f"cached_{split}.npz"
        dst_path = DST / f"cached_{split}.npz"
        z = np.load(src_path, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        N = len(data["smiles"])
        sm_list = [s.decode() if isinstance(s, bytes) else s for s in data["smiles"]]

        # Add physchem_feat + has_physchem
        phys_feat = np.zeros((N, 12), dtype=np.float32)
        has_phys = np.zeros(N, dtype=bool)
        for i, smi in enumerate(sm_list):
            c = canon(smi) or smi
            if c in phys_z_map:
                phys_feat[i] = phys_z_map[c]
                has_phys[i] = True
        data["physchem_feat"] = phys_feat
        data["has_physchem"] = has_phys

        # Denoise P column on train/val only; test is already clean (identical
        # to the pre-pollution data/expanded/ test cache).
        p_col = data["targets"][:, 6].copy()
        p_mask_before = ~np.isnan(p_col)
        if split in ("train", "val"):
            smiles_in_clean = np.array([s in clean_p_smiles for s in sm_list])
            polluted = p_mask_before & ~smiles_in_clean
            p_col[polluted] = np.nan
            data["targets"] = data["targets"].copy()
            data["targets"][:, 6] = p_col
            p_mask_after = ~np.isnan(p_col)
            print(f"  {split:5s}: {N} rows  phys_covered={has_phys.sum()}  P: {p_mask_before.sum()}→{p_mask_after.sum()} rows "
                  f"(removed {polluted.sum()} polluted)")
        else:
            # Test: leave untouched
            print(f"  {split:5s}: {N} rows  phys_covered={has_phys.sum()}  P: {p_mask_before.sum()} rows "
                  f"(test left unchanged)")

        np.savez(dst_path, **data)

    print(f"\nA1 cache written to: {DST}")


if __name__ == "__main__":
    main()
