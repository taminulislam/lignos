#!/usr/bin/env python3
"""Back-fill physchem_feat / has_physchem in the LignoIL_unified cached_*.npz
files, using the updated il_physchem_features.csv (now covering all 52 ILs).

Unlike extend_cache_with_unified_lignin.py (which adds new rows), this script
updates the physchem columns in-place for existing rows. Runs on train/val/test.
Writes a .pre_physchem_backfill backup once per file.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UNI_DIR = PROJECT_ROOT / "lignos" / "data" / "LignoIL_unified"

PHYSCHEM_SHORT = [
    "kt_alpha", "kt_beta", "clogp", "viscosity_298K", "pKa", "conductivity",
    "cat_MW", "cat_C", "cat_N", "an_MW", "an_O", "an_HB",
]


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


def main():
    print("Loading updated physchem table...")
    phys = pd.read_csv(UNI_DIR / "il_physchem_features.csv")
    phys["smi_canon"] = phys["smiles"].map(canon)
    phys_map = {}
    for _, row in phys.iterrows():
        key = row["smi_canon"] or row["smiles"]
        phys_map[key] = np.array([row[c] for c in PHYSCHEM_SHORT], dtype=np.float32)
    print(f"  loaded {len(phys_map)} SMILES → physchem vectors")

    for split in ("train", "val", "test"):
        path = UNI_DIR / f"cached_{split}.npz"
        backup = UNI_DIR / f"cached_{split}.pre_physchem_backfill.npz"
        if not path.exists():
            print(f"  {split}: {path.name} not found, skipping")
            continue
        z = np.load(path, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        N = len(data["smiles"])

        if not backup.exists():
            np.savez(backup, **data)
        before_has = int(data["has_physchem"].sum()) if "has_physchem" in data else 0

        smiles_arr = [s.decode() if isinstance(s, bytes) else s for s in data["smiles"]]
        if "physchem_feat" not in data:
            data["physchem_feat"] = np.zeros((N, 12), dtype=np.float32)
        if "has_physchem" not in data:
            data["has_physchem"] = np.zeros(N, dtype=bool)

        filled = 0
        unmatched = set()
        for i, smi in enumerate(smiles_arr):
            c = canon(smi) or smi
            vec = phys_map.get(c)
            if vec is None:
                vec = phys_map.get(smi)
            if vec is None:
                unmatched.add(c)
                continue
            data["physchem_feat"][i] = np.where(np.isnan(vec), 0.0, vec)
            if not data["has_physchem"][i]:
                filled += 1
            data["has_physchem"][i] = True
        after_has = int(data["has_physchem"].sum())
        unique_smi = len(set(smiles_arr))

        np.savez(path, **data)
        print(f"  {split:5s}: {N} rows / {unique_smi} unique SMILES  "
              f"has_physchem {before_has} → {after_has} (+{filled} newly filled, "
              f"{len(unmatched)} unique SMILES still unmatched)")
        if unmatched:
            for u in sorted(unmatched)[:5]:
                print(f"       unmatched ex: {u}")

    print("Done.")


if __name__ == "__main__":
    main()
