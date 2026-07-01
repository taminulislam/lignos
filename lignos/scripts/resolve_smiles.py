#!/usr/bin/env python3
"""Resolve missing SMILES for iThermo compounds using PubChem REST API.

Looks up SMILES from IUPAC names via the PubChem PUG REST interface.
Falls back to NCI/CACTUS resolver if PubChem fails.

Usage:
    python resolve_smiles.py --dry_run          # Preview what will be looked up
    python resolve_smiles.py                    # Run lookup and update CSV
    python resolve_smiles.py --output resolved.csv  # Save to separate file
"""

import argparse
import sys
import time
import json
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent


def lookup_pubchem(name, retries=2):
    """Look up SMILES from compound name via PubChem PUG REST API.

    Args:
        name: IUPAC or common name of the compound
        retries: number of retry attempts

    Returns:
        SMILES string or None
    """
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(name)}/property/IsomericSMILES,CanonicalSMILES/JSON"
    )

    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props:
                    p = props[0]
                    return (p.get("IsomericSMILES")
                            or p.get("CanonicalSMILES")
                            or p.get("SMILES"))
            elif resp.status_code == 404:
                return None  # Not found, don't retry
            elif resp.status_code == 503:
                # Rate limited or server busy
                time.sleep(2 ** attempt)
                continue
        except (requests.RequestException, json.JSONDecodeError):
            if attempt < retries:
                time.sleep(1)
                continue
    return None


def lookup_cactus(name, retries=1):
    """Fallback: look up SMILES via NCI/CACTUS chemical identifier resolver.

    Args:
        name: compound name
        retries: retry attempts

    Returns:
        SMILES string or None
    """
    url = (
        f"https://cactus.nci.nih.gov/chemical/structure/"
        f"{requests.utils.quote(name)}/smiles"
    )

    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                smiles = resp.text.strip()
                if smiles and not smiles.startswith("<!"):  # Not HTML error
                    return smiles
            return None
        except requests.RequestException:
            if attempt < retries:
                time.sleep(1)
                continue
    return None


def validate_smiles(smiles):
    """Validate SMILES using RDKit.

    Returns:
        Canonical SMILES or None if invalid.
    """
    from rdkit import Chem
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def resolve_il_name(name):
    """Try multiple strategies to resolve an IL name to SMILES.

    Strategy order:
    1. Full name -> PubChem
    2. Full name -> CACTUS
    3. Split into cation/anion names, look up each separately
    4. Clean up common naming variations and retry

    Returns:
        SMILES string or None
    """
    # Strategy 1: Direct PubChem lookup
    smiles = lookup_pubchem(name)
    if validate_smiles(smiles):
        return validate_smiles(smiles)

    # Strategy 1b: Try with cleaned brackets (very common in IL names)
    cleaned = name.replace("[(", "(").replace(")]", ")")
    if cleaned != name:
        smiles = lookup_pubchem(cleaned)
        if validate_smiles(smiles):
            return validate_smiles(smiles)

    # Strategy 2: Direct CACTUS lookup
    smiles = lookup_cactus(name)
    if validate_smiles(smiles):
        return validate_smiles(smiles)

    # Strategy 3: Split cation/anion and look up separately
    # Common IL naming: "CATION ANION" or "CATION + ANION"
    for sep in [" bis", " tris", " tetrakis"]:
        if sep in name.lower():
            # Don't split on these -- they're part of anion names
            break
    else:
        # Try splitting on common patterns
        parts = try_split_il_name(name)
        if parts:
            cat_name, an_name = parts
            cat_smi = lookup_pubchem(cat_name) or lookup_cactus(cat_name)
            an_smi = lookup_pubchem(an_name) or lookup_cactus(an_name)

            if cat_smi and an_smi:
                combined = f"{cat_smi}.{an_smi}"
                if validate_smiles(combined):
                    return validate_smiles(combined)

    # Strategy 4: Name variations
    variations = generate_name_variations(name)
    for var in variations:
        smiles = lookup_pubchem(var)
        if validate_smiles(smiles):
            return validate_smiles(smiles)

    return None


def try_split_il_name(name):
    """Try to split an IL name into cation and anion components.

    Common patterns:
    - "1-ethyl-3-methylimidazolium tetrafluoroborate"
    - "N-butylpyridinium chloride"

    Returns:
        (cation_name, anion_name) or None
    """
    # Common anion keywords to split on
    anion_keywords = [
        "bis[(trifluoromethyl)sulfonyl]imide",
        "bis((trifluoromethyl)sulfonyl)imide",
        "bis(trifluoromethylsulfonyl)imide",
        "trifluoromethanesulfonate",
        "trifluoroacetate",
        "tetrafluoroborate",
        "hexafluorophosphate",
        "dicyanamide",
        "cyanocyanamide",
        "thiocyanate",
        "acetate",
        "chloride",
        "bromide",
        "iodide",
        "nitrate",
        "sulfate",
        "phosphate",
        "tosylate",
        "triflate",
        "formate",
        "lactate",
        "propanoate",
        "butanoate",
    ]

    name_lower = name.lower()
    for anion in anion_keywords:
        if anion in name_lower:
            idx = name_lower.rfind(anion)
            cation_part = name[:idx].strip().rstrip(" ")
            anion_part = name[idx:].strip()
            if cation_part and anion_part:
                return (cation_part, anion_part)

    return None


def generate_name_variations(name):
    """Generate common naming variations for retry."""
    variations = []

    # Replace square brackets with parentheses (common IL naming issue)
    v = name.replace("[(", "((").replace(")]", "))")
    if v != name:
        variations.append(v)

    v = name.replace("[", "(").replace("]", ")")
    if v != name:
        variations.append(v)

    # Common anion name normalizations
    anion_rewrites = {
        "bis[(trifluoromethyl)sulfonyl]imide": "bis(trifluoromethylsulfonyl)imide",
        "bis((trifluoromethyl)sulfonyl)imide": "bis(trifluoromethylsulfonyl)imide",
        "bis[(trifluoromethyl)sulfonyl]amide": "bis(trifluoromethylsulfonyl)amide",
        "trifluorotris(perfluoroethyl)phosphate(V)": "tris(pentafluoroethyl)trifluorophosphate",
        "trifluorotris(perfluoroethyl)phosphate": "tris(pentafluoroethyl)trifluorophosphate",
    }
    for old, new in anion_rewrites.items():
        if old.lower() in name.lower():
            idx = name.lower().find(old.lower())
            rewritten = name[:idx] + new + name[idx + len(old):]
            variations.append(rewritten)

    # Try "imide" <-> "amide" swap (both are used for NTf2)
    if "imide" in name.lower():
        variations.append(name.replace("imide", "amide").replace("Imide", "Amide"))
    if "amide" in name.lower():
        variations.append(name.replace("amide", "imide").replace("Amide", "Imide"))

    # Remove stereochemistry indicators
    for prefix in ["(+)-", "(-)-", "(±)-", "(R)-", "(S)-"]:
        if name.startswith(prefix):
            variations.append(name[len(prefix):])

    # Remove "-1H-" prefix on imidazolium (some databases omit it)
    if "-1H-" in name:
        variations.append(name.replace("-1H-", "-"))

    # Common abbreviation expansions
    abbrevs = {
        "BMIM": "1-butyl-3-methylimidazolium",
        "EMIM": "1-ethyl-3-methylimidazolium",
        "HMIM": "1-hexyl-3-methylimidazolium",
        "OMIM": "1-octyl-3-methylimidazolium",
        "AMIM": "1-allyl-3-methylimidazolium",
        "NTf2": "bis(trifluoromethylsulfonyl)imide",
        "TFSI": "bis(trifluoromethylsulfonyl)imide",
        "BF4": "tetrafluoroborate",
        "PF6": "hexafluorophosphate",
        "OAc": "acetate",
        "OTf": "trifluoromethanesulfonate",
    }
    for abbr, full in abbrevs.items():
        if abbr in name:
            variations.append(name.replace(abbr, full))

    return variations


def main():
    parser = argparse.ArgumentParser(description="Resolve missing SMILES")
    parser.add_argument("--compounds_csv", type=str,
                        default=str(PROJECT_ROOT / "data/pipeline/ilthermo_compounds.csv"))
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV (default: update in-place)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Just show what would be looked up")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max compounds to process")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between API calls (seconds)")
    args = parser.parse_args()

    df = pd.read_csv(args.compounds_csv)
    missing = df[df["smiles"].isna()].copy()

    print(f"Total compounds: {len(df)}")
    print(f"Missing SMILES: {len(missing)}")

    if args.dry_run:
        print(f"\nWould look up {len(missing)} compounds:")
        for _, row in missing.head(20).iterrows():
            print(f"  {row['compound_id']}: {row['name']}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        return

    if args.limit:
        missing = missing.head(args.limit)
        print(f"Processing first {args.limit} compounds")

    # Resolve SMILES
    resolved = 0
    failed = []

    for idx, (_, row) in enumerate(missing.iterrows()):
        cid = row["compound_id"]
        name = row["name"]

        print(f"  [{idx+1}/{len(missing)}] {cid}: {name[:60]}...", end=" ")

        smiles = resolve_il_name(name)

        if smiles:
            df.loc[df["compound_id"] == cid, "smiles"] = smiles
            resolved += 1
            print(f"-> {smiles[:50]}")
        else:
            failed.append((cid, name))
            print("-> FAILED")

        time.sleep(args.delay)  # Rate limiting

    print(f"\nResults: {resolved} resolved, {len(failed)} failed")

    # Save
    output_path = args.output or args.compounds_csv
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    # Save failed list for manual resolution
    if failed:
        failed_path = V5_ROOT / "scripts" / "failed_smiles_lookup.txt"
        with open(failed_path, "w") as f:
            for cid, name in failed:
                f.write(f"{cid}\t{name}\n")
        print(f"Failed lookups saved to: {failed_path}")

    # Now generate missing_compounds.txt (compounds with SMILES but no data)
    df_updated = pd.read_csv(output_path)
    existing_pc = {p.stem for p in (PROJECT_ROOT / "data/pipeline/point_clouds").glob("*.npz")}
    existing_img = {
        f.stem.split("_")[0]
        for f in (PROJECT_ROOT / "data/pipeline/cosmo_images").glob("*_cosmo.png")
    }
    all_existing = existing_pc | existing_img

    actionable = df_updated[
        (df_updated["smiles"].notna()) & (~df_updated["compound_id"].isin(all_existing))
    ]

    missing_path = V5_ROOT / "scripts" / "missing_compounds.txt"
    actionable["compound_id"].to_csv(missing_path, index=False, header=False)
    print(f"\nmissing_compounds.txt: {len(actionable)} compounds ready for generation")


if __name__ == "__main__":
    main()
