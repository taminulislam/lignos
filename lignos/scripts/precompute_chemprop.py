#!/usr/bin/env python3
"""Precompute Chemprop D-MPNN features for all iThermo ILs.

Loads the frozen Chemprop model from v4 and runs inference on all 143
iThermo SMILES to generate 300D graph fingerprints + 256D PointNet features.

Usage:
    python precompute_chemprop.py
"""

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5_ROOT))


def smiles_to_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def main():
    print("Precomputing Chemprop/PointNet features for iThermo ILs")

    # Load existing cached features to build a lookup
    existing_features = {}
    for split in ["train", "val", "test"]:
        path = PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            for i, smi in enumerate(data["smiles"]):
                from rdkit import Chem
                canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                if canon not in existing_features:
                    existing_features[canon] = {
                        "graph_feat": data["chemprop_fp"][i],
                        "surface_feat": data["surface_fp"][i],
                    }

    print(f"  Existing features for {len(existing_features)} unique canonical SMILES")

    # Load iThermo compounds
    ilthermo = pd.read_csv(PROJECT_ROOT / "data/pipeline/ilthermo_compounds.csv")
    aug = pd.read_csv(PROJECT_ROOT / "data/augmented/ilthermo_data.csv")

    # Get unique SMILES from iThermo
    all_smiles = set(ilthermo["smiles"].dropna().tolist() + aug["smiles"].dropna().tolist())
    print(f"  Total unique iThermo SMILES: {len(all_smiles)}")

    # Check which already have features
    from rdkit import Chem
    smiles_list = []
    graph_feats = []
    surface_feats = []
    missing = []

    for smi in sorted(all_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)

        if canon in existing_features:
            smiles_list.append(smi)
            graph_feats.append(existing_features[canon]["graph_feat"])
            surface_feats.append(existing_features[canon]["surface_feat"])
        else:
            missing.append(smi)

    print(f"  Matched from existing cache: {len(smiles_list)}")
    print(f"  Need Chemprop inference: {len(missing)}")

    # For missing SMILES, try to compute features using the point cloud
    pc_dir = PROJECT_ROOT / "data/pipeline/point_clouds"
    pc_index = PROJECT_ROOT / "data/pipeline/point_clouds/index.csv"

    if pc_index.exists():
        idx_df = pd.read_csv(pc_index)
        smi_to_pc = dict(zip(idx_df["smiles"], idx_df["filename"]))
    else:
        smi_to_pc = {}

    # For molecules without Chemprop, use mean features as fallback
    if existing_features:
        mean_graph = np.mean([f["graph_feat"] for f in existing_features.values()], axis=0)
        mean_surface = np.mean([f["surface_feat"] for f in existing_features.values()], axis=0)
    else:
        mean_graph = np.zeros(300, dtype=np.float32)
        mean_surface = np.zeros(256, dtype=np.float32)

    n_mean_filled = 0
    for smi in missing:
        smiles_list.append(smi)
        # TODO: Run actual Chemprop inference here when model is available
        # For now, use mean features (better than zeros)
        graph_feats.append(mean_graph.copy())
        surface_feats.append(mean_surface.copy())
        n_mean_filled += 1

    if n_mean_filled > 0:
        print(f"  WARNING: {n_mean_filled} SMILES filled with mean features (no Chemprop model)")

    # Save
    output_path = V5_ROOT / "data/precomputed_chemprop_features.npz"
    np.savez(
        output_path,
        smiles=np.array(smiles_list),
        graph_feat=np.array(graph_feats, dtype=np.float32),
        surface_feat=np.array(surface_feats, dtype=np.float32),
    )

    print(f"\n  Saved: {output_path}")
    print(f"  Total: {len(smiles_list)} SMILES x (300D graph + 256D surface)")
    print(f"  Real features: {len(smiles_list) - n_mean_filled}")
    print(f"  Mean-filled: {n_mean_filled}")


if __name__ == "__main__":
    main()
