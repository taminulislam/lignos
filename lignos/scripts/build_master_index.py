#!/usr/bin/env python3
"""Build a master index mapping all compound identifiers and data paths.

Creates a unified CSV that maps:
  - compound_id (iThermo ID, e.g., ABXNZH)
  - smiles (canonical SMILES)
  - il_name (short name, e.g., EMIM NTf2)
  - pc_hash (point cloud filename hash)
  - has_point_cloud, has_cosmo_views, has_ion_images, has_sigma_map (booleans)
  - point_cloud_path, cosmo_frames_dir, cation_img_path, anion_img_path, sigma_map_path

Usage:
    python build_master_index.py
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent


def smiles_to_hash(smiles):
    return hashlib.md5(smiles.encode()).hexdigest()[:12]


def main():
    # Load sources
    ilthermo = pd.read_csv(PROJECT_ROOT / "data/pipeline/ilthermo_compounds.csv")
    pc_index_path = PROJECT_ROOT / "data/pipeline/point_clouds/index.csv"
    pc_index = pd.read_csv(pc_index_path) if pc_index_path.exists() else pd.DataFrame()

    # Also include training set compounds (28 ILs from il_data_raw)
    raw_path = PROJECT_ROOT / "data/processed/il_data_raw.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        training_smiles = raw[["smiles", "il_short_name"]].drop_duplicates("smiles")
    else:
        training_smiles = pd.DataFrame(columns=["smiles", "il_short_name"])

    # Build unified list from iThermo
    records = []
    for _, row in ilthermo.iterrows():
        if pd.isna(row.get("smiles")):
            continue
        records.append({
            "compound_id": row["compound_id"],
            "name": row.get("name", ""),
            "smiles": row["smiles"],
            "source": "ilthermo",
        })

    # Add training set compounds not already in iThermo
    ilthermo_smiles = set(ilthermo["smiles"].dropna())
    for _, row in training_smiles.iterrows():
        if row["smiles"] not in ilthermo_smiles:
            records.append({
                "compound_id": row.get("il_short_name", ""),
                "name": row.get("il_short_name", ""),
                "smiles": row["smiles"],
                "source": "training_set",
            })

    df = pd.DataFrame(records)
    print(f"Total compounds with SMILES: {len(df)}")

    # Compute hash IDs
    df["pc_hash"] = df["smiles"].apply(smiles_to_hash)

    # Map il_name from pc_index
    if not pc_index.empty:
        hash_to_name = dict(zip(
            pc_index["filename"].str.replace(".npz", ""),
            pc_index["il_name"]
        ))
        df["il_name"] = df["pc_hash"].map(hash_to_name).fillna(df["name"])
    else:
        df["il_name"] = df["name"]

    # Check data availability
    pc_dir = PROJECT_ROOT / "data/pipeline/point_clouds"
    v5_cosmo = V5_ROOT / "data/cosmo_images"
    v5_ions = V5_ROOT / "data/ion_images"
    v5_sigma = V5_ROOT / "data/sigma_maps"

    # Also check original cosmo_images
    orig_cosmo = PROJECT_ROOT / "data/pipeline/cosmo_images"

    df["has_point_cloud"] = df["pc_hash"].apply(
        lambda h: (pc_dir / f"{h}.npz").exists()
    )
    df["has_cosmo_views"] = df["pc_hash"].apply(
        lambda h: (v5_cosmo / f"{h}_frames").exists()
    ) | df["compound_id"].apply(
        lambda c: (orig_cosmo / f"{c}_frames").exists() if c else False
    )
    df["has_ion_images"] = df["compound_id"].apply(
        lambda c: (v5_ions / f"{c}_cation.png").exists() if c else False
    )
    df["has_sigma_map"] = df["pc_hash"].apply(
        lambda h: (v5_sigma / f"{h}.npz").exists()
    )

    # Build paths
    df["point_cloud_path"] = df.apply(
        lambda r: str(pc_dir / f"{r['pc_hash']}.npz") if r["has_point_cloud"] else "",
        axis=1
    )
    df["cosmo_frames_dir"] = df.apply(
        lambda r: (
            str(v5_cosmo / f"{r['pc_hash']}_frames")
            if (v5_cosmo / f"{r['pc_hash']}_frames").exists()
            else str(orig_cosmo / f"{r['compound_id']}_frames")
            if r["compound_id"] and (orig_cosmo / f"{r['compound_id']}_frames").exists()
            else ""
        ),
        axis=1
    )
    df["cation_img_path"] = df.apply(
        lambda r: str(v5_ions / f"{r['compound_id']}_cation.png") if r["has_ion_images"] else "",
        axis=1
    )
    df["anion_img_path"] = df.apply(
        lambda r: str(v5_ions / f"{r['compound_id']}_anion.png") if r["has_ion_images"] else "",
        axis=1
    )
    df["sigma_map_path"] = df.apply(
        lambda r: str(v5_sigma / f"{r['pc_hash']}.npz") if r["has_sigma_map"] else "",
        axis=1
    )

    # Completeness flags
    df["data_complete"] = (
        df["has_point_cloud"]
        & df["has_cosmo_views"]
        & df["has_ion_images"]
        & df["has_sigma_map"]
    )

    # Save
    output_path = V5_ROOT / "data" / "master_index.csv"
    df.to_csv(output_path, index=False)

    # Report
    print(f"\nData availability:")
    print(f"  Point clouds:   {df['has_point_cloud'].sum():3d} / {len(df)}")
    print(f"  COSMO views:    {df['has_cosmo_views'].sum():3d} / {len(df)}")
    print(f"  Ion images:     {df['has_ion_images'].sum():3d} / {len(df)}")
    print(f"  Sigma maps:     {df['has_sigma_map'].sum():3d} / {len(df)}")
    print(f"  Fully complete: {df['data_complete'].sum():3d} / {len(df)}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
