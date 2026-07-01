#!/usr/bin/env python3
"""Build expanded cached_{split}.npz files for the Combined(40D) pipeline.

Merges the original 28-IL v4 cached data with 585 new ILs from the expanded
ILThermo fetch. New ILs get:
  - Morgan fingerprint PCA'd to 40D as substitute for image features
  - thermo_feat = [T_norm, x_norm, 0...0] (only T and composition available)
  - v4_base = 0 (no v4 model predictions available)
  - Partial targets (gamma1, H_E only; rest NaN)

The original 28 ILs keep their FULL feature set (v4_base, image features,
thermo_feat, all 7 targets) so the test set comparison to the 0.8309 baseline
remains valid.

Usage:
    python build_expanded_dataset.py
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
V4 = PROJECT_ROOT / "cosmobridge_v4"

PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P",
         "density", "viscosity"]


def compute_morgan_fp(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)


def load_original_split(split):
    path = V4 / "data" / f"cached_{split}.npz"
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_new_il_rows(ilthermo_path):
    """Load expanded ILThermo data and build per-row feature/target arrays."""
    df = pd.read_csv(ilthermo_path)

    # Keep rows with at least one usable target (any of the 4 properties)
    has_gamma = df["gamma"].notna()
    has_he = df["H_E_kJmol"].notna()
    has_dens = df["density_gcm3"].notna()
    has_visc = df["viscosity_Pas"].notna()
    df = df[has_gamma | has_he | has_dens | has_visc].copy()

    # Cap rows per IL per property to keep dataset manageable
    MAX_ROWS_PER_IL_PROP = 30
    keep_idx = []
    for prop in df["property_type"].unique():
        sub = df[df["property_type"] == prop]
        for smi, grp in sub.groupby("il_smiles"):
            if len(grp) > MAX_ROWS_PER_IL_PROP:
                keep_idx.extend(grp.sample(MAX_ROWS_PER_IL_PROP, random_state=42).index)
            else:
                keep_idx.extend(grp.index)
    df = df.loc[sorted(set(keep_idx))].copy()
    print(f"After capping at {MAX_ROWS_PER_IL_PROP} rows/IL/prop: {len(df)} rows")

    print(f"New ILThermo rows with usable targets: {len(df)}")
    print(f"  Unique ILs: {df['il_smiles'].nunique()}")
    print(f"  With gamma: {df['gamma'].notna().sum()}")
    print(f"  With H_E: {df['H_E_kJmol'].notna().sum()}")
    print(f"  With density: {df['density_gcm3'].notna().sum()}")
    print(f"  With viscosity: {df['viscosity_Pas'].notna().sum()}")

    # Build targets (9D: 7 original + density + viscosity)
    targets = np.full((len(df), 9), np.nan, dtype=np.float32)
    targets[:, 0] = df["gamma"].values            # gamma1
    targets[:, 3] = df["H_E_kJmol"].values        # H_E
    targets[:, 7] = df["density_gcm3"].values      # density (g/cm³)
    targets[:, 8] = df["viscosity_Pas"].values      # viscosity (Pa·s)

    T = df["temperature"].values.astype(np.float32)
    x = df["x_water"].fillna(0.5).values.astype(np.float32)

    smiles = df["il_smiles"].values
    il_ids = df["il_name"].values

    return {
        "targets": targets,
        "temperature": T,
        "x_water": x,
        "smiles": smiles,
        "il_ids": il_ids,
    }


def main():
    print("=" * 60)
    print("Building expanded dataset")
    print("=" * 60)

    # Load original splits
    orig = {}
    for split in ["train", "val", "test"]:
        orig[split] = load_original_split(split)
        print(f"Original {split}: {len(orig[split]['smiles'])} samples, "
              f"{len(set(orig[split]['smiles']))} unique ILs")

    # Load new ILThermo data
    new_data = build_new_il_rows(V5 / "data" / "ilthermo_LignoIL.csv")

    # Filter out ILs already in original splits to avoid leakage
    orig_smiles = set()
    for split in ["train", "val", "test"]:
        for s in orig[split]["smiles"]:
            orig_smiles.add(s.decode() if isinstance(s, bytes) else s)
    # Also canonicalize for matching
    orig_canonical = set()
    for s in orig_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            orig_canonical.add(Chem.MolToSmiles(mol))

    new_smiles = new_data["smiles"]
    keep_mask = np.ones(len(new_smiles), dtype=bool)
    for i, s in enumerate(new_smiles):
        if s in orig_smiles:
            keep_mask[i] = False
            continue
        mol = Chem.MolFromSmiles(s)
        if mol and Chem.MolToSmiles(mol) in orig_canonical:
            keep_mask[i] = False

    n_overlap = (~keep_mask).sum()
    print(f"\nFiltered {n_overlap} rows overlapping with original splits")
    for k in new_data:
        new_data[k] = new_data[k][keep_mask]
    print(f"New rows after filtering: {len(new_data['smiles'])}")
    print(f"New unique ILs: {len(set(new_data['smiles']))}")

    # Compute Morgan fingerprints for ALL ILs (original + new)
    print("\nComputing Morgan fingerprints...")
    all_smiles_unique = list(set(
        [s.decode() if isinstance(s, bytes) else s
         for split in orig for s in orig[split]["smiles"]]
        + list(new_data["smiles"])
    ))
    fp_cache = {}
    for s in all_smiles_unique:
        fp_cache[s] = compute_morgan_fp(s)
    print(f"  Computed {len(fp_cache)} unique Morgan FPs")

    # Normalize targets for new data
    # Original 7 props: use original train stats
    # New props (density, viscosity): compute stats from new data
    orig_train_targets = orig["train"]["targets"]
    target_means = np.zeros(9, dtype=np.float32)
    target_stds = np.ones(9, dtype=np.float32)
    target_means[:7] = np.nanmean(orig_train_targets, axis=0)
    target_stds[:7] = np.nanstd(orig_train_targets, axis=0)
    # density and viscosity stats from new data
    for idx, col in [(7, "density_gcm3"), (8, "viscosity_Pas")]:
        vals = new_data["targets"][:, idx]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            target_means[idx] = valid.mean()
            target_stds[idx] = valid.std()
    target_stds[target_stds < 1e-8] = 1.0
    print(f"\nTarget stats (for normalization):")
    for i, p in enumerate(PROPS):
        print(f"  {p}: mean={target_means[i]:.4f}  std={target_stds[i]:.4f}")

    new_targets_normed = (new_data["targets"] - target_means) / target_stds

    # Normalize T and x for new data
    # Estimate T and x scaling from original thermo_feat
    # Column 0 likely correlates with T; use the original data range
    orig_thermo = orig["train"]["thermo_feat"]

    # Build thermo_feat for new data: [T_norm, x_norm, 0...0]
    T_scaler = StandardScaler()
    x_scaler = StandardScaler()
    T_scaler.fit(new_data["temperature"].reshape(-1, 1))
    x_scaler.fit(new_data["x_water"].reshape(-1, 1))
    new_thermo = np.zeros((len(new_data["smiles"]), 25), dtype=np.float32)
    new_thermo[:, 0] = T_scaler.transform(
        new_data["temperature"].reshape(-1, 1)
    ).ravel()
    new_thermo[:, 2] = x_scaler.transform(
        new_data["x_water"].reshape(-1, 1)
    ).ravel()

    # Split new data: add all to train (keep original val/test unchanged)
    n_new = len(new_data["smiles"])
    new_smiles_unique = list(set(new_data["smiles"]))
    np.random.seed(42)
    np.random.shuffle(new_smiles_unique)
    n_val_new = max(1, int(0.15 * len(new_smiles_unique)))
    val_new_set = set(new_smiles_unique[:n_val_new])
    train_new_set = set(new_smiles_unique[n_val_new:])

    train_mask = np.array([s in train_new_set for s in new_data["smiles"]])
    val_mask = np.array([s in val_new_set for s in new_data["smiles"]])

    print(f"\nNew data split:")
    print(f"  Train: {train_mask.sum()} rows ({len(train_new_set)} ILs)")
    print(f"  Val:   {val_mask.sum()} rows ({len(val_new_set)} ILs)")
    print(f"  Test:  0 rows (keeping original test unchanged)")

    # Build expanded cached files
    out_dir = V5 / "data" / "LignoIL"
    out_dir.mkdir(parents=True, exist_ok=True)

    def pad_targets_7to9(targets_7d, n_samples):
        """Pad original 7D targets to 9D with NaN for density/viscosity."""
        out = np.full((n_samples, 9), np.nan, dtype=np.float32)
        out[:, :7] = targets_7d
        return out

    for split, new_mask in [("train", train_mask), ("val", val_mask), ("test", None)]:
        o = orig[split]
        n_orig = len(o["smiles"])

        if new_mask is not None and new_mask.sum() > 0:
            n_add = new_mask.sum()

            # v4 base: zeros for new ILs (9D to match target width)
            v4_fusion = np.zeros((n_add, 9), dtype=np.float32)
            v4_chemprop = np.zeros((n_add, 9), dtype=np.float32)

            # Pad original v4 preds from 7D to 9D
            orig_fusion_9d = np.zeros((n_orig, 9), dtype=np.float32)
            orig_fusion_9d[:, :7] = o["preds_fusion"]
            orig_chemprop_9d = np.zeros((n_orig, 9), dtype=np.float32)
            orig_chemprop_9d[:, :7] = o["preds_chemprop"]

            new_fps = np.array([
                fp_cache[s] for s in new_data["smiles"][new_mask]
            ], dtype=np.float32)

            new_chemprop_fp = np.zeros((n_add, o["chemprop_fp"].shape[1]),
                                       dtype=np.float32)
            new_surface_fp = np.zeros((n_add, o["surface_fp"].shape[1]),
                                      dtype=np.float32)

            expanded = {
                "chemprop_fp": np.concatenate([o["chemprop_fp"], new_chemprop_fp]),
                "surface_fp": np.concatenate([o["surface_fp"], new_surface_fp]),
                "thermo_feat": np.concatenate([o["thermo_feat"],
                                               new_thermo[new_mask]]),
                "targets": np.concatenate([pad_targets_7to9(o["targets"], n_orig),
                                           new_targets_normed[new_mask]]),
                "preds_fusion": np.concatenate([orig_fusion_9d, v4_fusion]),
                "preds_chemprop": np.concatenate([orig_chemprop_9d,
                                                   v4_chemprop]),
                "smiles": np.concatenate([o["smiles"],
                                          new_data["smiles"][new_mask]]),
                "il_ids": np.concatenate([o["il_ids"],
                                          new_data["il_ids"][new_mask]]),
                "morgan_fp": np.concatenate([
                    np.array([fp_cache[s.decode() if isinstance(s, bytes) else s]
                              for s in o["smiles"]]),
                    new_fps,
                ]),
                "is_original": np.concatenate([
                    np.ones(n_orig, dtype=bool),
                    np.zeros(n_add, dtype=bool),
                ]),
            }
        else:
            # Test split or no new data — pad targets to 9D
            expanded = dict(o)
            expanded["targets"] = pad_targets_7to9(o["targets"], n_orig)
            orig_fusion_9d = np.zeros((n_orig, 9), dtype=np.float32)
            orig_fusion_9d[:, :7] = o["preds_fusion"]
            expanded["preds_fusion"] = orig_fusion_9d
            orig_chemprop_9d = np.zeros((n_orig, 9), dtype=np.float32)
            orig_chemprop_9d[:, :7] = o["preds_chemprop"]
            expanded["preds_chemprop"] = orig_chemprop_9d
            expanded["morgan_fp"] = np.array([
                fp_cache[s.decode() if isinstance(s, bytes) else s]
                for s in o["smiles"]
            ], dtype=np.float32)
            expanded["is_original"] = np.ones(n_orig, dtype=bool)

        out_path = out_dir / f"cached_{split}.npz"
        np.savez(out_path, **expanded)
        n_total = len(expanded["smiles"])
        n_new_in_split = n_total - n_orig if new_mask is not None else 0
        n_unique = len(set(
            s.decode() if isinstance(s, bytes) else s
            for s in expanded["smiles"]
        ))

        # Count non-NaN targets per property
        tgt = expanded["targets"]
        print(f"\n{split}: {n_total} samples ({n_orig} orig + {n_new_in_split} new), "
              f"{n_unique} unique ILs")
        for i, p in enumerate(PROPS):
            n_valid = (~np.isnan(tgt[:, i])).sum()
            print(f"  {p}: {n_valid}/{n_total} valid ({100*n_valid/n_total:.0f}%)")

    print(f"\nExpanded files saved to {out_dir}/")


if __name__ == "__main__":
    main()
