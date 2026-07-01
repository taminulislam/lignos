"""A5.1 helper — precompute cation/anion Morgan fingerprints per cache row.

Splits each IL SMILES into its charged fragments (cation, anion) via RDKit,
computes 2048-bit Morgan(r=2) fingerprints separately for each, and writes
them as {cation_morgan, anion_morgan, has_split} arrays aligned to the cache.

Later used by train_a5_ionsplit.py as the input to a disentangled encoder
branch. The hypothesis: a novel IL like [C2H4COOHmim][Cl] that collapsed A2
because its whole-SMILES encoding was OOD will still be partially in-dist on
the anion side ([Cl] is seen in training) and partially in-dist on the cation
side ([mim] core is seen) — separate encoders can combine these signals.

Output: lignos/data/LignoIL_A1/ion_split_{train,val,test}.npz
  cation_morgan : (N, 2048) float32
  anion_morgan  : (N, 2048) float32
  has_split     : (N,)       float32  (1 if both cation+anion parsed, else 0)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
CACHE = V5 / "data" / "LignoIL_A1"

NBITS = 2048
RADIUS = 2


def morgan_fp(mol):
    if mol is None:
        return np.zeros(NBITS, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS)
    arr = np.zeros(NBITS, dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def split_il(smi: str):
    """Return (cation_mol, anion_mol) or (None, None)."""
    if not isinstance(smi, str):
        return None, None
    frags = [f for f in smi.split(".") if f]
    cat_mol = an_mol = None
    for f in frags:
        m = Chem.MolFromSmiles(f)
        if m is None:
            continue
        q = sum(a.GetFormalCharge() for a in m.GetAtoms())
        if q > 0 and cat_mol is None:
            cat_mol = m
        elif q < 0 and an_mol is None:
            an_mol = m
    return cat_mol, an_mol


def build_split(split):
    p = CACHE / f"cached_{split}.npz"
    d = np.load(p, allow_pickle=True)
    smis = d["smiles"]
    n = len(smis)
    cat = np.zeros((n, NBITS), dtype=np.float32)
    an = np.zeros((n, NBITS), dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    for i, s in enumerate(smis):
        c_m, a_m = split_il(s)
        if c_m is not None and a_m is not None:
            cat[i] = morgan_fp(c_m)
            an[i] = morgan_fp(a_m)
            mask[i] = 1.0
    out = CACHE / f"ion_split_{split}.npz"
    np.savez_compressed(out, cation_morgan=cat, anion_morgan=an, has_split=mask)
    print(f"{split}: {int(mask.sum())}/{n} rows split → {out.name}")


def main():
    for split in ("train", "val", "test"):
        build_split(split)


if __name__ == "__main__":
    main()
