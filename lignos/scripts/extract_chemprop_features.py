#!/usr/bin/env python3
"""Extract Chemprop D-MPNN 300D fingerprints for all iThermo SMILES.

Loads the trained Chemprop model and extracts the penultimate-layer
graph fingerprints (300D) for all unique iThermo SMILES.

Usage:
    python extract_chemprop_features.py
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


def extract_fingerprints(model, smiles_list, batch_size=64):
    """Extract 300D fingerprints from Chemprop's encoder.

    Args:
        model: loaded Chemprop MoleculeModel
        smiles_list: list of SMILES strings
        batch_size: batch size for inference

    Returns:
        fingerprints: (N, 300) numpy array
    """
    from chemprop.data import MoleculeDataset, MoleculeDatapoint
    from chemprop.data.utils import construct_molecule_batch

    model.eval()
    all_fps = []

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]

        # Create data points using chemprop API
        data_points = []
        valid_idx = []
        for j, smi in enumerate(batch_smiles):
            try:
                dp = MoleculeDatapoint(smiles=[smi])
                data_points.append(dp)
                valid_idx.append(j)
            except Exception:
                # Invalid SMILES, fill with zeros later
                pass

        if not data_points:
            all_fps.append(np.zeros((len(batch_smiles), 300), dtype=np.float32))
            continue

        dataset = MoleculeDataset(data_points)
        batch = construct_molecule_batch(dataset.data)

        with torch.no_grad():
            # Get encoder output (fingerprint) before the FFN
            encodings = model.encoder(batch)  # (B, hidden_size=300)
            fps = encodings.cpu().numpy()

        # Fill in results (handle skipped SMILES)
        result = np.zeros((len(batch_smiles), 300), dtype=np.float32)
        for k, idx in enumerate(valid_idx):
            result[idx] = fps[k]
        all_fps.append(result)

        if (i // batch_size) % 10 == 0:
            print(f"    Batch {i//batch_size}: processed {min(i+batch_size, len(smiles_list))}/{len(smiles_list)}")

    return np.concatenate(all_fps, axis=0)


def main():
    from chemprop.utils import load_checkpoint
    from rdkit import Chem

    print("Extracting Chemprop fingerprints for iThermo ILs")

    # Load model
    model_path = PROJECT_ROOT / "checkpoints/chemprop/fold_0/model_0/model.pt"
    model = load_checkpoint(str(model_path))
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Load existing cached features for reference
    existing = {}
    for split in ["train", "val", "test"]:
        data = np.load(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz",
                       allow_pickle=True)
        for i, smi in enumerate(data["smiles"]):
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            existing[canon] = {
                "graph": data["chemprop_fp"][i],
                "surface": data["surface_fp"][i],
                "split": split,
            }
    print(f"  Existing features: {len(existing)} ILs")

    # Get all unique SMILES from iThermo
    aug = pd.read_csv(PROJECT_ROOT / "data/augmented/ilthermo_data.csv")
    ilthermo = pd.read_csv(PROJECT_ROOT / "data/pipeline/ilthermo_compounds.csv")

    all_smiles = set()
    for smi in list(aug["smiles"].dropna()) + list(ilthermo["smiles"].dropna()):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            all_smiles.add(smi)

    unique_smiles = sorted(all_smiles)
    print(f"  Total unique SMILES: {len(unique_smiles)}")

    # Separate into: has features vs needs inference
    need_inference = []
    reuse_smiles = []
    reuse_graph = []
    reuse_surface = []

    for smi in unique_smiles:
        canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        if canon in existing:
            reuse_smiles.append(smi)
            reuse_graph.append(existing[canon]["graph"])
            reuse_surface.append(existing[canon]["surface"])
        else:
            need_inference.append(smi)

    print(f"  Reusing existing: {len(reuse_smiles)}")
    print(f"  Need Chemprop inference: {len(need_inference)}")

    # Run Chemprop inference
    if need_inference:
        print(f"  Running inference on {len(need_inference)} SMILES...")
        new_fps = extract_fingerprints(model, need_inference)
        print(f"  Extracted: {new_fps.shape}")

        # For surface features of new ILs: use point clouds if available,
        # otherwise use mean
        pc_dir = PROJECT_ROOT / "data/pipeline/point_clouds"
        pc_index = pc_dir / "index.csv"
        smi_to_pc = {}
        if pc_index.exists():
            idx_df = pd.read_csv(pc_index)
            smi_to_pc = dict(zip(idx_df["smiles"], idx_df["filename"]))

        mean_surface = np.mean(reuse_surface, axis=0) if reuse_surface else np.zeros(256)
        new_surface = []
        n_mean = 0
        for smi in need_inference:
            if smi in smi_to_pc:
                # TODO: run PointNet on point cloud for real features
                new_surface.append(mean_surface.copy())
                n_mean += 1
            else:
                new_surface.append(mean_surface.copy())
                n_mean += 1

        new_surface = np.array(new_surface, dtype=np.float32)
        if n_mean > 0:
            print(f"  WARNING: {n_mean} ILs use mean surface features (no PointNet)")
    else:
        new_fps = np.zeros((0, 300), dtype=np.float32)
        new_surface = np.zeros((0, 256), dtype=np.float32)

    # Combine
    all_smiles_out = reuse_smiles + need_inference
    all_graph = np.concatenate([
        np.array(reuse_graph, dtype=np.float32) if reuse_graph else np.zeros((0, 300)),
        new_fps.astype(np.float32),
    ])
    all_surface = np.concatenate([
        np.array(reuse_surface, dtype=np.float32) if reuse_surface else np.zeros((0, 256)),
        new_surface,
    ])

    # Also store which split each IL belongs to (for leakage prevention)
    splits = []
    for smi in all_smiles_out:
        canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        if canon in existing:
            splits.append(existing[canon]["split"])
        else:
            splits.append("new")

    # Save
    output = V5_ROOT / "data/precomputed_chemprop_features.npz"
    np.savez(
        output,
        smiles=np.array(all_smiles_out),
        graph_feat=all_graph,
        surface_feat=all_surface,
        splits=np.array(splits),
    )

    print(f"\n  Saved: {output}")
    print(f"  {len(all_smiles_out)} SMILES: "
          f"{len(reuse_smiles)} reused + {len(need_inference)} new Chemprop")

    # Summary by split
    for s in ["train", "val", "test", "new"]:
        n = sum(1 for x in splits if x == s)
        print(f"    {s}: {n} ILs")


if __name__ == "__main__":
    main()
