#!/usr/bin/env python3
"""Stage 2a — Render ion images for multiple low-energy conformers.

Wraps render_ion_images.generate_ion_conformer / render_ion_image but
parameterizes the ETKDG seed so we can sample N distinct conformers per ion.
MMFF energy is recorded so we can keep the N lowest-energy per ion.

Output layout:
    ion_images_multi/conf_{k}/{compound_id}_cation.png
    ion_images_multi/conf_{k}/{compound_id}_anion.png
    ion_images_multi/energies/{compound_id}.json   (list of (conf_id, mmff_energy))

Usage:
    # Generate 4 additional conformers (conf_1..conf_4; conf_0 is the existing run)
    python render_ion_images_multi.py --n_conformers 5 --start_id 0

    # Just one conformer with a specific seed
    python render_ion_images_multi.py --n_conformers 1 --start_id 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lignos/scripts"))

from render_ion_images import (  # noqa: E402
    render_ion_image,
    split_il_smiles,
)


def generate_ion_conformer_seeded(smiles, seed, n_points=512):
    """Same as generate_ion_conformer but with an explicit ETKDG seed.

    Returns (points, mmff_energy) so callers can keep the N lowest-energy
    conformers per ion.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid ion SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        fallback = AllChem.ETKDG()
        fallback.randomSeed = int(seed)
        AllChem.EmbedMolecule(mol, fallback)

    # Optimize and record energy
    try:
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, AllChem.MMFFGetMoleculeProperties(mol)
        )
        if ff is not None:
            ff.Minimize(maxIts=500)
            mmff_energy = float(ff.CalcEnergy())
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            mmff_energy = float("nan")
    except Exception:
        mmff_energy = float("nan")

    AllChem.ComputeGasteigerCharges(mol)

    conf = mol.GetConformer()
    positions = conf.GetPositions()
    charges = []
    for atom in mol.GetAtoms():
        c = float(atom.GetProp("_GasteigerCharge"))
        if np.isnan(c):
            c = 0.0
        charges.append(c)
    charges = np.array(charges)

    vdw_radii = {
        1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
        15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85,
    }

    surface_points, surface_normals, surface_charges = [], [], []
    rng = np.random.default_rng(seed)

    for i, atom in enumerate(mol.GetAtoms()):
        pos = positions[i]
        r = vdw_radii.get(atom.GetAtomicNum(), 1.70)
        n_atom_pts = max(8, n_points // len(positions))

        phi = rng.uniform(0, 2 * np.pi, n_atom_pts)
        cos_theta = rng.uniform(-1, 1, n_atom_pts)
        theta = np.arccos(cos_theta)

        sx = r * np.sin(theta) * np.cos(phi) + pos[0]
        sy = r * np.sin(theta) * np.sin(phi) + pos[1]
        sz = r * np.cos(theta) + pos[2]

        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)

        surface_points.append(np.stack([sx, sy, sz], axis=1))
        surface_normals.append(np.stack([nx, ny, nz], axis=1))
        surface_charges.append(np.full(n_atom_pts, charges[i]))

    surface_points = np.concatenate(surface_points, axis=0)
    surface_normals = np.concatenate(surface_normals, axis=0)
    surface_charges = np.concatenate(surface_charges, axis=0)

    if len(surface_points) > n_points:
        idx = rng.choice(len(surface_points), n_points, replace=False)
        surface_points = surface_points[idx]
        surface_normals = surface_normals[idx]
        surface_charges = surface_charges[idx]

    points = np.column_stack([surface_points, surface_normals, surface_charges])
    return points, mmff_energy


def process_compound_conformer(compound_id, smiles, conf_id, output_root, resolution=224):
    """Render one cation+anion pair for a single conformer id.

    The conf_id is the ETKDG seed as well as the output subdirectory name.
    Returns (paths, energies) tuple or None on failure.
    """
    out_dir = Path(output_root) / f"conf_{conf_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cation_path = out_dir / f"{compound_id}_cation.png"
    anion_path = out_dir / f"{compound_id}_anion.png"

    if cation_path.exists() and anion_path.exists():
        return (cation_path, anion_path), (None, None)

    try:
        cat_smi, an_smi = split_il_smiles(smiles)
    except ValueError as e:
        print(f"  SKIP {compound_id}: {e}")
        return None

    try:
        # Use a seed that is (conf_id * 10000 + hash(smiles) & 0xffff) so each
        # compound gets independent seeds across conformers.
        seed = int(conf_id) * 10_000 + (abs(hash(compound_id)) % 10_000)
        cat_pts, cat_e = generate_ion_conformer_seeded(cat_smi, seed, n_points=512)
        render_ion_image(cat_pts, cation_path, resolution)

        an_pts, an_e = generate_ion_conformer_seeded(an_smi, seed + 1, n_points=512)
        render_ion_image(an_pts, anion_path, resolution)

        return (cation_path, anion_path), (cat_e, an_e)

    except Exception as e:
        print(f"  ERROR {compound_id}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_conformers", type=int, default=5,
                    help="Total number of conformers to produce (starting at --start_id)")
    ap.add_argument("--start_id", type=int, default=1,
                    help="First conf_id to render (conf_0 is the existing deterministic run)")
    ap.add_argument("--compounds_csv", type=str,
                    default=str(PROJECT_ROOT / "data/pipeline/ilthermo_compounds.csv"))
    ap.add_argument("--compound_list", type=str, default=None,
                    help="Optional file with one compound_id per line")
    ap.add_argument("--output_root", type=str,
                    default=str(PROJECT_ROOT / "lignos/data/ion_images_multi"))
    ap.add_argument("--resolution", type=int, default=224)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.compounds_csv)

    if args.compound_list:
        with open(args.compound_list) as f:
            ids = {line.strip() for line in f if line.strip()}
        targets = [(r["compound_id"], r["smiles"])
                   for _, r in df.iterrows()
                   if r["compound_id"] in ids and pd.notna(r["smiles"])]
    else:
        targets = [(r["compound_id"], r["smiles"])
                   for _, r in df.iterrows()
                   if r.get("is_il", True) and pd.notna(r["smiles"])]

    energies_dir = Path(args.output_root) / "energies"
    energies_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(targets)} compounds × {args.n_conformers} conformers "
          f"(conf_{args.start_id}..conf_{args.start_id + args.n_conformers - 1})")

    for i, (cid, smi) in enumerate(targets):
        energies = {}
        energies_path = energies_dir / f"{cid}.json"
        if energies_path.exists():
            with open(energies_path) as f:
                energies = json.load(f)

        for k in range(args.start_id, args.start_id + args.n_conformers):
            print(f"  [{i+1}/{len(targets)}] {cid} conf_{k}")
            result = process_compound_conformer(cid, smi, k, args.output_root, args.resolution)
            if result is None:
                continue
            _, (cat_e, an_e) = result
            if cat_e is not None or an_e is not None:
                energies[str(k)] = {"cation": cat_e, "anion": an_e}

        if energies:
            with open(energies_path, "w") as f:
                json.dump(energies, f, indent=2)

    print("Done rendering conformers.")


if __name__ == "__main__":
    main()
