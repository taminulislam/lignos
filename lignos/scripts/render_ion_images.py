#!/usr/bin/env python3
"""Render separate COSMO images for cation and anion components.

Splits an ionic liquid SMILES into cation and anion fragments, generates
3D conformers, computes partial charges, and renders COSMO-style images.

Usage:
    python render_ion_images.py --compound_id AAQcOE
    python render_ion_images.py --compound_list missing_compounds.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def split_il_smiles(smiles):
    """Split an ionic liquid SMILES into cation and anion fragments.

    Handles both dot-separated (CATION.ANION) and charge-based splitting.

    Returns:
        (cation_smiles, anion_smiles)
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Get individual fragments
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

    if len(frags) < 2:
        raise ValueError(f"Cannot split into cation+anion: {smiles}")

    # Identify cation (positive charge) and anion (negative charge)
    cation, anion = None, None
    for frag in frags:
        charge = Chem.GetFormalCharge(frag)
        if charge > 0:
            cation = frag
        elif charge < 0:
            anion = frag
        elif charge == 0:
            # Neutral fragment: assign based on context
            if cation is None:
                cation = frag
            else:
                anion = frag

    if cation is None or anion is None:
        # Fallback: first fragment = cation, second = anion
        cation, anion = frags[0], frags[1]

    return Chem.MolToSmiles(cation), Chem.MolToSmiles(anion)


def generate_ion_conformer(smiles, n_points=512):
    """Generate 3D conformer and compute surface points for a single ion.

    Args:
        smiles: SMILES string for the ion
        n_points: number of surface points to sample

    Returns:
        points: (N, 7) array of x, y, z, nx, ny, nz, charge
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid ion SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    # Generate 3D conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        # Fallback
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    # Compute Gasteiger charges
    AllChem.ComputeGasteigerCharges(mol)

    # Get atom positions and charges
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    charges = []
    for atom in mol.GetAtoms():
        charge = float(atom.GetProp("_GasteigerCharge"))
        if np.isnan(charge):
            charge = 0.0
        charges.append(charge)
    charges = np.array(charges)

    # Generate surface points using van der Waals radii
    from rdkit.Chem import rdMolDescriptors
    vdw_radii = {
        1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
        15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85,
    }

    surface_points = []
    surface_normals = []
    surface_charges = []

    for i, atom in enumerate(mol.GetAtoms()):
        pos = positions[i]
        r = vdw_radii.get(atom.GetAtomicNum(), 1.70)

        # Sample points on sphere around atom
        n_atom_pts = max(8, n_points // len(positions))
        phi = np.random.uniform(0, 2 * np.pi, n_atom_pts)
        cos_theta = np.random.uniform(-1, 1, n_atom_pts)
        theta = np.arccos(cos_theta)

        sx = r * np.sin(theta) * np.cos(phi) + pos[0]
        sy = r * np.sin(theta) * np.sin(phi) + pos[1]
        sz = r * np.cos(theta) + pos[2]

        # Normal = direction from atom center
        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)

        surface_points.append(np.stack([sx, sy, sz], axis=1))
        surface_normals.append(np.stack([nx, ny, nz], axis=1))
        surface_charges.append(np.full(n_atom_pts, charges[i]))

    surface_points = np.concatenate(surface_points, axis=0)
    surface_normals = np.concatenate(surface_normals, axis=0)
    surface_charges = np.concatenate(surface_charges, axis=0)

    # Subsample to exactly n_points
    if len(surface_points) > n_points:
        idx = np.random.choice(len(surface_points), n_points, replace=False)
        surface_points = surface_points[idx]
        surface_normals = surface_normals[idx]
        surface_charges = surface_charges[idx]

    points = np.column_stack([
        surface_points, surface_normals, surface_charges
    ])  # (N, 7)

    return points


def render_ion_image(points, output_path, resolution=224, colormap="RdBu_r"):
    """Render a single ion's surface points as a COSMO-style image.

    Args:
        points: (N, 7) surface points with charges
        output_path: where to save the image
        resolution: image resolution
        colormap: matplotlib colormap
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    coords = points[:, :3]
    normals = points[:, 3:6]
    charges = points[:, 6]

    # Center and scale
    coords = coords - coords.mean(axis=0)
    scale = np.abs(coords).max()
    coords = coords / (scale + 1e-8)

    # Color by charge
    vmin, vmax = np.percentile(charges, [5, 95])
    if abs(vmax - vmin) < 1e-8:
        charge_norm = np.full_like(charges, 0.5)
    else:
        charge_norm = np.clip((charges - vmin) / (vmax - vmin), 0, 1)

    cmap = get_cmap(colormap)
    colors = cmap(charge_norm)

    # Simple diffuse lighting
    light_dir = np.array([0, 0, 1])
    diffuse = np.clip(np.dot(normals, light_dir), 0.2, 1.0)

    depth = coords[:, 2]
    order = np.argsort(depth)

    lit_colors = colors[order].copy()
    lit_colors[:, :3] *= diffuse[order, np.newaxis]

    fig = plt.figure(figsize=(resolution / 100, resolution / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        coords[order, 0], coords[order, 1], coords[order, 2],
        c=lit_colors, s=max(1, 30000 // len(coords)),
        alpha=0.9, edgecolors="none",
    )

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_position([0, 0, 1, 1])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight",
                pad_inches=0, facecolor="white")
    plt.close(fig)


def process_compound(compound_id, smiles, output_dir, resolution=224):
    """Generate cation and anion images for one compound.

    Args:
        compound_id: molecule identifier
        smiles: full IL SMILES
        output_dir: root output directory
        resolution: image resolution

    Returns:
        (cation_path, anion_path) or None on failure
    """
    output_dir = Path(output_dir)

    cation_path = output_dir / f"{compound_id}_cation.png"
    anion_path = output_dir / f"{compound_id}_anion.png"

    if cation_path.exists() and anion_path.exists():
        return cation_path, anion_path

    try:
        cat_smi, an_smi = split_il_smiles(smiles)
    except ValueError as e:
        print(f"  SKIP {compound_id}: {e}")
        return None

    try:
        cat_points = generate_ion_conformer(cat_smi, n_points=512)
        render_ion_image(cat_points, cation_path, resolution)

        an_points = generate_ion_conformer(an_smi, n_points=512)
        render_ion_image(an_points, anion_path, resolution)

        return cation_path, anion_path

    except Exception as e:
        print(f"  ERROR {compound_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Render cation/anion images")
    parser.add_argument("--compound_id", type=str)
    parser.add_argument("--compound_list", type=str)
    parser.add_argument("--compounds_csv", type=str,
                        default=str(PROJECT_ROOT / "data/pipeline/ilthermo_compounds.csv"))
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "lignos/data/ion_images"))
    parser.add_argument("--resolution", type=int, default=224)
    args = parser.parse_args()

    import pandas as pd
    compounds_df = pd.read_csv(args.compounds_csv)

    if args.compound_id:
        row = compounds_df[compounds_df["compound_id"] == args.compound_id]
        if row.empty:
            print(f"Compound {args.compound_id} not found in CSV")
            return
        targets = [(row.iloc[0]["compound_id"], row.iloc[0]["smiles"])]
    elif args.compound_list:
        with open(args.compound_list) as f:
            ids = {line.strip() for line in f if line.strip()}
        targets = [
            (row["compound_id"], row["smiles"])
            for _, row in compounds_df.iterrows()
            if row["compound_id"] in ids and pd.notna(row["smiles"])
        ]
    else:
        targets = [
            (row["compound_id"], row["smiles"])
            for _, row in compounds_df.iterrows()
            if row.get("is_il", True) and pd.notna(row["smiles"])
        ]

    print(f"Processing {len(targets)} compounds")
    success, fail = 0, 0

    for i, (cid, smi) in enumerate(targets):
        print(f"  [{i+1}/{len(targets)}] {cid}")
        result = process_compound(cid, smi, args.output_dir, args.resolution)
        if result:
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} success, {fail} failed")


if __name__ == "__main__":
    main()
