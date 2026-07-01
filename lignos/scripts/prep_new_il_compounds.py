#!/usr/bin/env python3
"""Prepare geometry inputs (cation/anion/pair .xyz) for the 32 new LignoIL
SMILES that aren't yet in the DFT pipeline.

Reuses the RDKit + MMFF94 path from scripts/pipeline/step2_geometry_optimization.py.
Assigns stable compound_ids LIGNO_000..LIGNO_031, writes them to
data/pipeline/geometries/ (same directory the existing pipeline already uses)
so downstream step3/step4 scripts can pick them up by ID.

Also writes data/pipeline/lignoil_new_compounds.csv with the new compounds and
lignoil_new_dft_task_list.txt with one line per (compound_id)_(fragment)
needed by the array job.
"""
from __future__ import annotations
import csv, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
PIPELINE_DIR = PROJECT_ROOT / "data" / "pipeline"
GEOM_DIR = PIPELINE_DIR / "geometries"
GEOM_DIR.mkdir(parents=True, exist_ok=True)

COMPOUND_CSV = PIPELINE_DIR / "lignoil_new_compounds.csv"
TASK_LIST = PIPELINE_DIR / "lignoil_new_dft_task_list.txt"
UNIFIED_CSV = V5 / "data" / "LignoIL_unified" / "unified_lignin.csv"


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


def split_il_fragments(smi):
    frags = [f for f in smi.split(".") if f]
    cation = anion = None
    for f in frags:
        m = Chem.MolFromSmiles(f)
        if m is None:
            continue
        q = sum(a.GetFormalCharge() for a in m.GetAtoms())
        if q > 0 and cation is None:
            cation = f
        elif q < 0 and anion is None:
            anion = f
    return cation, anion


def smiles_to_xyz(smi, n_conformers=10):
    """RDKit ETKDGv3 + MMFF94. Returns (atoms, coords) or (None, None)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
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


def combine_pair(cat_atoms, cat_coords, an_atoms, an_coords, sep=3.0):
    """Place cation and anion with their centers of mass offset by `sep` Å along x."""
    c_com = cat_coords.mean(0); a_com = an_coords.mean(0)
    cat_c = cat_coords - c_com
    an_c = an_coords - a_com
    cat_c[:, 0] -= sep / 2
    an_c[:, 0] += sep / 2
    return cat_atoms + an_atoms, np.vstack([cat_c, an_c])


def main():
    print(f"Reading unified lignin CSV → {UNIFIED_CSV}")
    uni = pd.read_csv(UNIFIED_CSV)
    # Which SMILES aren't already in the cached training data?
    cache = np.load(V5 / "data/LignoIL_unified/cached_train.npz", allow_pickle=True)
    cached_smi = set()
    for s in cache["smiles"]:
        c = canon(s.decode() if isinstance(s, bytes) else s)
        if c:
            cached_smi.add(c)

    uni["smi_canon"] = uni["smiles"].map(canon)
    # include ALL unique SMILES in unified_lignin.csv that we want DFT for.
    # Interpretation: "new ILs" = those in unified_lignin.csv. If they're
    # already in cache, skip only the 28 originals (they already have DFT).
    all_uni_smi = sorted(set(uni["smi_canon"].dropna()))
    # Legacy DFT compounds live in data/pipeline/geometries/; we only want to
    # avoid the original 28 ILs (which already have geometry). Everything else
    # — including the 32 supplementary-map ILs — gets processed here.
    # Take unique unified SMILES minus whichever are already in the main
    # cosmobridge_v4 cached npz (the original 28).
    orig = np.load(V5 / "data/LignoIL/cached_train.npz", allow_pickle=True)
    orig_smi = set()
    for s in orig["smiles"]:
        if not orig["is_original"][list(orig["smiles"]).index(s)]:
            continue
        c = canon(s.decode() if isinstance(s, bytes) else s)
        if c:
            orig_smi.add(c)
    # simpler: just take unified SMILES not in the original 28
    orig_smi_set = set(canon(s.decode() if isinstance(s, bytes) else s)
                       for s in orig["smiles"][orig["is_original"].astype(bool)])
    new_smi = [s for s in all_uni_smi if s not in orig_smi_set]
    print(f"Unified unique SMILES: {len(all_uni_smi)}")
    print(f"Original (DFT already exists): {len(orig_smi_set)}")
    print(f"New SMILES needing DFT: {len(new_smi)}")

    # Build compound records
    records = []
    tasks = []
    failed = []
    for i, smi in enumerate(sorted(new_smi)):
        compound_id = f"LIGNO_{i:03d}"
        cation, anion = split_il_fragments(smi)
        if cation is None or anion is None:
            print(f"  SKIP {compound_id}: could not split into cation/anion → {smi}")
            failed.append(smi)
            continue

        # Build each fragment's geometry
        cat_atoms, cat_coords = smiles_to_xyz(cation)
        an_atoms, an_coords = smiles_to_xyz(anion)
        if cat_atoms is None or an_atoms is None:
            print(f"  SKIP {compound_id}: geometry failed → {smi}")
            failed.append(smi)
            continue

        # Write .xyz for cation, anion, pair
        write_xyz(GEOM_DIR / f"{compound_id}_cation.xyz", cat_atoms, cat_coords,
                  comment=f"{compound_id} cation {cation}")
        write_xyz(GEOM_DIR / f"{compound_id}_anion.xyz", an_atoms, an_coords,
                  comment=f"{compound_id} anion {anion}")
        pair_a, pair_c = combine_pair(cat_atoms, cat_coords, an_atoms, an_coords)
        write_xyz(GEOM_DIR / f"{compound_id}_pair.xyz", pair_a, pair_c,
                  comment=f"{compound_id} pair {smi}")

        records.append({
            "compound_id": compound_id,
            "name": f"lignoil_new_{i:03d}",
            "smiles": smi,
            "cation_smiles": cation,
            "anion_smiles": anion,
            "n_cation_atoms": len(cat_atoms),
            "n_anion_atoms": len(an_atoms),
            "n_pair_atoms": len(pair_a),
        })
        for frag in ("cation", "anion", "pair"):
            tasks.append(f"{compound_id}_{frag}")

    # Write outputs
    COMPOUND_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPOUND_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with open(TASK_LIST, "w") as f:
        for t in tasks:
            f.write(t + "\n")

    print(f"\nWrote {len(records)} compounds → {COMPOUND_CSV}")
    print(f"Wrote {len(tasks)} DFT tasks → {TASK_LIST}")
    print(f"Geometry files → {GEOM_DIR}")
    if failed:
        print(f"\n{len(failed)} SMILES failed geometry / fragment split:")
        for s in failed:
            print(f"  {s}")


if __name__ == "__main__":
    main()
