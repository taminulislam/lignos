#!/usr/bin/env python3
"""Expanded ILThermo data fetch — pulls Activity, Excess enthalpy, Density, and
Viscosity for IL+water binary mixtures. Outputs a single CSV with all data
points, property type labels, and per-sample condition info (T, x, P).

This replaces the narrower extract_ilthermopy.py which only pulled Activity
and Excess enthalpy.

Output:
    lignos/data/ilthermo_expanded.csv
    lignos/data/ilthermo_expanded_summary.csv  (per-IL property availability)

Usage:
    python fetch_ilthermo_expanded.py
    python fetch_ilthermo_expanded.py --props Activity "Excess enthalpy" Density
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ilthermopy as ilt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = PROJECT_ROOT / "lignos"
OUTPUT = V5_ROOT / "data" / "ilthermo_expanded.csv"

DEFAULT_PROPS = ["Activity", "Excess enthalpy", "Density", "Viscosity"]


def extract_property(prop_name, compound="water", ncmp=2):
    """Fetch all datasets for a property type and extract data points."""
    print(f"\n{'='*60}")
    print(f"Fetching: {prop_name} (ncmp={ncmp}, compound={compound})")
    print(f"{'='*60}")

    if compound:
        results = ilt.Search(compound=compound, n_compounds=ncmp, prop=prop_name)
    else:
        results = ilt.Search(n_compounds=ncmp, prop=prop_name)

    print(f"  Found {len(results)} datasets, {results['num_data_points'].sum()} total points")

    all_rows = []
    n_success = 0
    n_fail = 0

    for idx, row in results.iterrows():
        entry_id = row["id"]
        try:
            entry = ilt.GetEntry(entry_id)
            df = entry.data
            if df is None or len(df) == 0:
                n_fail += 1
                continue

            header = entry.header
            components = entry.components

            water_idx = il_idx = None
            for i, comp in enumerate(components):
                if comp.smiles == "O" or "water" in comp.name.lower():
                    water_idx = i
                else:
                    il_idx = i

            if il_idx is None:
                n_fail += 1
                continue

            il_smiles = components[il_idx].smiles
            il_name = components[il_idx].name
            il_id = components[il_idx].id

            # Parse header to identify columns and units
            cols = {}
            units = {}
            for col, desc in header.items():
                dl = desc.lower()
                # Skip error/uncertainty columns
                if "error" in dl or "uncertainty" in dl:
                    continue
                if "mole fraction" in dl:
                    cols["x"] = col
                    cols["x_is_water"] = "water" in dl
                elif "weight fraction" in dl:
                    cols["w"] = col
                    cols["w_is_water"] = "water" in dl
                elif "molality" in dl:
                    cols["molality"] = col
                elif "temperature" in dl:
                    cols["T"] = col
                elif "activity coefficient" in dl:
                    cols["gamma"] = col
                elif "activity" in dl and "coefficient" not in dl:
                    cols["activity"] = col
                elif ("excess enthalpy" in dl or
                      ("enthalpy" in dl and "excess" in dl)):
                    cols["H_E"] = col
                    units["H_E"] = "kJ/mol" if "kj" in dl else "J/mol"
                elif ("specific density" in dl or
                      ("density" in dl and "molar volume" not in dl
                       and "volume" not in dl)):
                    cols["density"] = col
                    if "kg/m" in dl or "kg" in dl:
                        units["density"] = "kg/m3"
                    else:
                        units["density"] = "g/cm3"
                elif "viscosity" in dl:
                    cols["viscosity"] = col
                    if "mpa" in dl or "cp" in dl:
                        units["viscosity"] = "mPa.s"
                    else:
                        units["viscosity"] = "Pa.s"

            for _, data_row in df.iterrows():
                T = data_row.get(cols.get("T"), np.nan)
                if pd.isna(T):
                    continue

                # --- Composition (mole fraction of water) ---
                x_water = np.nan
                if "x" in cols:
                    x_val = data_row.get(cols["x"], np.nan)
                    if pd.notna(x_val):
                        x_water = x_val if cols.get("x_is_water") else 1.0 - x_val
                elif "w" in cols:
                    w_val = data_row.get(cols["w"], np.nan)
                    if pd.notna(w_val):
                        w_water = w_val if cols.get("w_is_water") else 1.0 - w_val
                        # Approximate: assume IL MW ~ 250 g/mol, water = 18.015
                        mw_water = 18.015
                        mw_il = 250.0
                        if 0 < w_water < 1:
                            n_w = w_water / mw_water
                            n_il = (1 - w_water) / mw_il
                            x_water = n_w / (n_w + n_il)

                # --- Activity coefficient (gamma) ---
                gamma = np.nan
                if "gamma" in cols:
                    gamma = data_row.get(cols["gamma"], np.nan)
                elif "activity" in cols:
                    act = data_row.get(cols["activity"], np.nan)
                    if pd.notna(act) and pd.notna(x_water) and x_water > 0.001:
                        gamma = act / x_water

                # --- Excess enthalpy (normalize to kJ/mol) ---
                H_E_val = np.nan
                if "H_E" in cols:
                    raw = data_row.get(cols["H_E"], np.nan)
                    if pd.notna(raw):
                        if units.get("H_E") == "J/mol":
                            H_E_val = raw / 1000.0
                        else:
                            H_E_val = raw

                # --- Density (normalize to g/cm³) ---
                density_val = np.nan
                if "density" in cols:
                    raw = data_row.get(cols["density"], np.nan)
                    if pd.notna(raw):
                        if units.get("density") == "kg/m3":
                            density_val = raw / 1000.0
                        else:
                            density_val = raw
                        if density_val <= 0 or density_val > 2.5:
                            density_val = np.nan

                # --- Viscosity (normalize to Pa·s) ---
                viscosity_val = np.nan
                if "viscosity" in cols:
                    raw = data_row.get(cols["viscosity"], np.nan)
                    if pd.notna(raw):
                        if units.get("viscosity") == "mPa.s":
                            viscosity_val = raw / 1000.0
                        else:
                            viscosity_val = raw
                        if viscosity_val <= 0 or viscosity_val > 10.0:
                            viscosity_val = np.nan

                r = {
                    "entry_id": entry_id,
                    "il_smiles": il_smiles,
                    "il_name": il_name,
                    "il_compound_id": il_id,
                    "temperature": T,
                    "x_water": x_water,
                    "property_type": prop_name,
                    "reference": row.get("reference", ""),
                    "gamma": gamma if prop_name == "Activity" else np.nan,
                    "H_E_kJmol": H_E_val,
                    "density_gcm3": density_val,
                    "viscosity_Pas": viscosity_val,
                }
                all_rows.append(r)

            n_success += 1
            if n_success % 50 == 0:
                print(f"  Processed {n_success}/{len(results)} datasets "
                      f"({len(all_rows)} rows)...")
            time.sleep(0.15)

        except Exception as e:
            n_fail += 1
            if n_fail <= 10:
                print(f"  ERROR {entry_id}: {type(e).__name__}: {e}")
            time.sleep(0.3)

    print(f"  Done: {n_success} success, {n_fail} failed, {len(all_rows)} data points")
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", nargs="+", default=DEFAULT_PROPS,
                    help="Property types to fetch")
    ap.add_argument("--output", type=str, default=str(OUTPUT))
    args = ap.parse_args()

    all_data = []
    for prop in args.props:
        rows = extract_property(prop)
        all_data.extend(rows)

    df = pd.DataFrame(all_data)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total data points: {len(df)}")
    print(f"Unique ILs (SMILES): {df['il_smiles'].nunique()}")
    print()

    for prop in args.props:
        sub = df[df["property_type"] == prop]
        print(f"  {prop:<20}: {len(sub):>6} points, {sub['il_smiles'].nunique():>4} ILs")

    # Property availability per IL
    summary_rows = []
    for smiles, grp in df.groupby("il_smiles"):
        r = {"il_smiles": smiles, "il_name": grp["il_name"].iloc[0]}
        for prop in args.props:
            sub = grp[grp["property_type"] == prop]
            r[f"n_{prop.lower().replace(' ', '_')}"] = len(sub)
        r["n_total"] = len(grp)
        r["n_props"] = sum(1 for prop in args.props
                          if len(grp[grp["property_type"] == prop]) > 0)
        summary_rows.append(r)
    summary = pd.DataFrame(summary_rows).sort_values("n_props", ascending=False)

    print(f"\nILs by property coverage:")
    for n in range(1, len(args.props) + 1):
        count = (summary["n_props"] >= n).sum()
        if count > 0:
            print(f"  >= {n} properties: {count} ILs")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows)")

    summary_path = out_path.parent / "ilthermo_expanded_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path} ({len(summary)} ILs)")


if __name__ == "__main__":
    main()
