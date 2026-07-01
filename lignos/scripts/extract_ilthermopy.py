#!/usr/bin/env python3
"""Comprehensive extraction of IL+water data from ILThermo via ILThermoPy.

Downloads ALL binary IL+water datasets for:
  - Activity coefficients (gamma)
  - Excess enthalpy (H_E)

Filters for x1 ≈ 0.5 and derives gamma from activity when needed.

Output: lignos/data/ilthermopy_extracted.csv

Usage:
    python extract_ilthermopy.py
"""

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import ilthermopy as ilt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = V5_ROOT / "data" / "ilthermopy_extracted.csv"


def extract_activity_data():
    """Extract all IL+water activity coefficient datasets."""
    print("Searching for Activity datasets (IL+water, binary)...")
    results = ilt.Search(compound='water', n_compounds=2, prop='Activity')
    print(f"  Found {len(results)} datasets, {results['num_data_points'].sum()} total points")

    all_rows = []
    n_success = 0
    n_fail = 0

    for idx, row in results.iterrows():
        entry_id = row['id']
        try:
            entry = ilt.GetEntry(entry_id)
            df = entry.data

            if df is None or len(df) == 0:
                n_fail += 1
                continue

            header = entry.header
            components = entry.components

            # Identify which component is water and which is IL
            water_idx = None
            il_idx = None
            for i, comp in enumerate(components):
                if comp.smiles == 'O' or comp.name.lower() == 'water':
                    water_idx = i
                else:
                    il_idx = i

            if water_idx is None or il_idx is None:
                n_fail += 1
                continue

            il_smiles = components[il_idx].smiles
            il_name = components[il_idx].name
            il_id = components[il_idx].id

            # Parse columns from header
            # Common patterns:
            # V1 = mole fraction, V2 = temperature, V3 = pressure
            # V4 = activity or activity coefficient
            x_col = None
            T_col = None
            gamma_col = None
            activity_col = None

            for col, desc in header.items():
                desc_lower = desc.lower()
                if 'mole fraction' in desc_lower:
                    x_col = col
                    x_is_water = 'water' in desc_lower
                elif 'temperature' in desc_lower:
                    T_col = col
                elif 'activity coefficient' in desc_lower:
                    gamma_col = col
                elif 'activity' in desc_lower and 'coefficient' not in desc_lower:
                    activity_col = col

            if T_col is None:
                n_fail += 1
                continue

            for _, data_row in df.iterrows():
                T = data_row.get(T_col, np.nan)
                if pd.isna(T):
                    continue

                # Get composition
                if x_col:
                    x_water = data_row[x_col]
                    if not x_is_water:
                        x_water = 1 - data_row[x_col]
                    x1 = x_water  # x1 = mole fraction of water (solvent)
                else:
                    x1 = np.nan

                # Get gamma
                gamma_val = np.nan
                if gamma_col:
                    gamma_val = data_row.get(gamma_col, np.nan)
                elif activity_col and x_col:
                    # activity = x * gamma → gamma = activity / x
                    activity = data_row.get(activity_col, np.nan)
                    x = data_row.get(x_col, np.nan)
                    if x > 0.001:
                        gamma_val = activity / x

                all_rows.append({
                    'entry_id': entry_id,
                    'il_smiles': il_smiles,
                    'il_name': il_name,
                    'il_compound_id': il_id,
                    'temperature': T,
                    'x1_water': x1,
                    'gamma_water': gamma_val,
                    'property_type': 'activity',
                    'reference': row.get('reference', ''),
                })

            n_success += 1
            if n_success % 25 == 0:
                print(f"  Processed {n_success}/{len(results)} datasets...")

            time.sleep(0.2)  # Rate limiting

        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  ERROR {entry_id}: {e}")

    print(f"  Activity: {n_success} success, {n_fail} failed, {len(all_rows)} data points")
    return all_rows


def extract_excess_enthalpy():
    """Extract all IL+water excess enthalpy datasets."""
    print("\nSearching for Excess enthalpy datasets (IL+water, binary)...")
    results = ilt.Search(compound='water', n_compounds=2, prop='Excess enthalpy')
    print(f"  Found {len(results)} datasets")

    all_rows = []
    n_success = 0
    n_fail = 0

    for idx, row in results.iterrows():
        entry_id = row['id']
        try:
            entry = ilt.GetEntry(entry_id)
            df = entry.data

            if df is None or len(df) == 0:
                n_fail += 1
                continue

            header = entry.header
            components = entry.components

            water_idx = None
            il_idx = None
            for i, comp in enumerate(components):
                if comp.smiles == 'O' or comp.name.lower() == 'water':
                    water_idx = i
                else:
                    il_idx = i

            if il_idx is None:
                n_fail += 1
                continue

            il_smiles = components[il_idx].smiles
            il_name = components[il_idx].name
            il_id = components[il_idx].id

            x_col, T_col, he_col = None, None, None
            for col, desc in header.items():
                desc_lower = desc.lower()
                if 'mole fraction' in desc_lower:
                    x_col = col
                    x_is_water = 'water' in desc_lower
                elif 'temperature' in desc_lower:
                    T_col = col
                elif 'excess enthalpy' in desc_lower or 'enthalpy' in desc_lower:
                    he_col = col

            for _, data_row in df.iterrows():
                T = data_row.get(T_col, np.nan) if T_col else np.nan
                he = data_row.get(he_col, np.nan) if he_col else np.nan

                x1 = np.nan
                if x_col:
                    x_water = data_row[x_col]
                    if not x_is_water:
                        x_water = 1 - data_row[x_col]
                    x1 = x_water

                all_rows.append({
                    'entry_id': entry_id,
                    'il_smiles': il_smiles,
                    'il_name': il_name,
                    'il_compound_id': il_id,
                    'temperature': T,
                    'x1_water': x1,
                    'H_E': he,
                    'property_type': 'excess_enthalpy',
                    'reference': row.get('reference', ''),
                })

            n_success += 1
            if n_success % 10 == 0:
                print(f"  Processed {n_success}/{len(results)} datasets...")

            time.sleep(0.2)

        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  ERROR {entry_id}: {e}")

    print(f"  H_E: {n_success} success, {n_fail} failed, {len(all_rows)} data points")
    return all_rows


def main():
    print("=" * 60)
    print("ILThermoPy: Comprehensive IL+Water Data Extraction")
    print("=" * 60)

    # Extract both property types
    activity_rows = extract_activity_data()
    he_rows = extract_excess_enthalpy()

    # Combine
    all_data = pd.DataFrame(activity_rows + he_rows)
    print(f"\nTotal extracted: {len(all_data)} data points")

    # Statistics
    print(f"\nBreakdown:")
    print(f"  Activity: {len(activity_rows)} points")
    print(f"  H_E: {len(he_rows)} points")
    print(f"  Unique ILs: {all_data['il_smiles'].nunique()}")
    print(f"  Temperature range: {all_data['temperature'].min():.1f} - {all_data['temperature'].max():.1f} K")

    if 'x1_water' in all_data.columns:
        x1 = all_data['x1_water'].dropna()
        print(f"  x1_water range: {x1.min():.3f} - {x1.max():.3f}")

        # Filter for x1 ≈ 0.5
        near_05 = all_data[(all_data['x1_water'] >= 0.45) & (all_data['x1_water'] <= 0.55)]
        print(f"\n  Filtered for x1 ≈ 0.5 (±0.05): {len(near_05)} points from {near_05['il_smiles'].nunique()} ILs")

    # Save full dataset
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    all_data.to_csv(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")
    print(f"  Total rows: {len(all_data)}")

    # Also save filtered x1≈0.5 version
    if len(near_05) > 0:
        filtered_path = OUTPUT.parent / "ilthermopy_x05_filtered.csv"
        near_05.to_csv(filtered_path, index=False)
        print(f"  Filtered (x1≈0.5): {filtered_path} ({len(near_05)} rows)")


if __name__ == "__main__":
    main()
