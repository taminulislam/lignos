#!/usr/bin/env python3
"""Tier-3 prep: generate cation/anion/pair geometries for the 156 SMILES in
`lignos/data/missing_dft_smiles.txt` that lack DFT-COSMO surfaces.

Assigns stable IDs `TIER3_000` through `TIER3_{N-1}`, writes .xyz files to
data/pipeline/geometries/, and emits a task list for the downstream DFT
array job (one line per fragment needed).

Reuses the RDKit + MMFF94 + pair-COM-offset logic from
scripts/prep_new_il_compounds.py.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
PIPELINE = PROJECT_ROOT / "data" / "pipeline"
GEOM_DIR = PIPELINE / "geometries"
GEOM_DIR.mkdir(parents=True, exist_ok=True)

MISSING_LIST = V5 / "data" / "missing_dft_smiles.txt"
OUT_CSV = PIPELINE / "tier3_compounds.csv"
TASK_LIST = PIPELINE / "tier3_dft_task_list.txt"


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else None


def split_il_fragments(smi):
    frags = [f for f in smi.split(".") if f]
    cat = an = None
    for f in frags:
        m = Chem.MolFromSmiles(f)
        if m is None:
            continue
        q = sum(a.GetFormalCharge() for a in m.GetAtoms())
        if q > 0 and cat is None:
            cat = f
        elif q < 0 and an is None:
            an = f
    return cat, an


def smiles_to_xyz(smi, n_conformers=10):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3(); params.randomSeed = 42
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if len(cids) == 0:
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        if len(cids) == 0:
            return None, None
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
    best_cid = min(range(len(results)), key=lambda i: results[i][1])
    conf = mol.GetConformer(best_cid)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = np.array([[conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z]
                       for i in range(mol.GetNumAtoms())])
    return atoms, coords


def write_xyz(path, atoms, coords, comment=""):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for a, xyz in zip(atoms, coords):
            f.write(f"{a} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}\n")


def combine_pair(cat_a, cat_c, an_a, an_c, sep=3.0):
    cc = cat_c - cat_c.mean(0); ac = an_c - an_c.mean(0)
    cc[:, 0] -= sep / 2; ac[:, 0] += sep / 2
    return cat_a + an_a, np.vstack([cc, ac])


def main():
    if not MISSING_LIST.exists():
        print(f"missing {MISSING_LIST}"); return
    smis = [l.strip() for l in open(MISSING_LIST) if l.strip()]
    print(f"Preparing {len(smis)} SMILES for DFT.")

    records = []; tasks = []; failed = []
    for i, smi in enumerate(smis):
        cid = f"TIER3_{i:03d}"
        cat, an = split_il_fragments(smi)
        if not (cat and an):
            print(f"  SKIP {cid}: fragment-split failed → {smi}")
            failed.append(smi); continue
        cat_a, cat_c = smiles_to_xyz(cat)
        an_a, an_c = smiles_to_xyz(an)
        if not (cat_a and an_a is not None):
            print(f"  SKIP {cid}: geometry failed → {smi}")
            failed.append(smi); continue
        write_xyz(GEOM_DIR / f"{cid}_cation.xyz", cat_a, cat_c, f"{cid} cation {cat}")
        write_xyz(GEOM_DIR / f"{cid}_anion.xyz",  an_a,  an_c,  f"{cid} anion {an}")
        pair_a, pair_c = combine_pair(cat_a, cat_c, an_a, an_c)
        write_xyz(GEOM_DIR / f"{cid}_pair.xyz",   pair_a, pair_c, f"{cid} pair {smi}")
        records.append({
            "compound_id": cid, "name": f"tier3_{i:03d}", "smiles": smi,
            "cation_smiles": cat, "anion_smiles": an,
            "n_cation_atoms": len(cat_a), "n_anion_atoms": len(an_a),
            "n_pair_atoms": len(pair_a),
        })
        for frag in ("cation", "anion", "pair"):
            tasks.append(f"{cid}_{frag}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    with open(TASK_LIST, "w") as f:
        for t in tasks: f.write(t + "\n")
    print(f"\nWrote {len(records)} compounds → {OUT_CSV}")
    print(f"Wrote {len(tasks)} DFT tasks → {TASK_LIST}")
    if failed:
        print(f"Failed: {len(failed)}")
        for s in failed[:10]: print(f"  {s}")


if __name__ == "__main__":
    main()
