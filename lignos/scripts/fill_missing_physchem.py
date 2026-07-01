#!/usr/bin/env python3
"""Fill physchem (12-D) for the 24 LignoIL ILs missing from il_physchem_features.csv.

Three fill tiers per column:
  1. RDKit-computable from SMILES: clogp, cat_MW, cat_C, cat_N, an_MW, an_O, an_HB
  2. ILThermo-expanded lookup: viscosity_298K (rows at 293-303K median, when present)
  3. Median-of-existing-28 imputation: kt_alpha, kt_beta, pKa, conductivity
     (plus viscosity_298K when ILThermo has no match)

Writes:
  - il_physchem_features.csv (overwrites; all 52 ILs present)
  - il_physchem_features_fill_audit.csv (per-IL, per-column source tag)

After running, re-run extend_cache_with_unified_lignin.py (or equivalent) to
rebuild cached_{train,val,test}.npz with the expanded physchem map.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
UNI_DIR = V5 / "data" / "LignoIL_unified"
PHYS_CSV = UNI_DIR / "il_physchem_features.csv"
UNI_CSV = UNI_DIR / "unified_lignin.csv"
ILTHERMO_CSV = V5 / "data" / "ilthermo_expanded.csv"
AUDIT_CSV = UNI_DIR / "il_physchem_features_fill_audit.csv"

PHYSCHEM_ORDER = [
    "smiles", "kt_alpha", "kt_beta", "clogp", "viscosity_298K", "pKa",
    "conductivity", "cat_MW", "cat_C", "cat_N", "an_MW", "an_O", "an_HB",
]
RDKIT_COLS = ["clogp", "cat_MW", "cat_C", "cat_N", "an_MW", "an_O", "an_HB"]
EMPIRICAL_COLS = ["kt_alpha", "kt_beta", "viscosity_298K", "pKa", "conductivity"]


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


def split_il(smi):
    """Return (cation_smiles, anion_smiles) by formal charge sign."""
    cation = anion = None
    for f in smi.split("."):
        m = Chem.MolFromSmiles(f)
        if m is None:
            continue
        q = sum(a.GetFormalCharge() for a in m.GetAtoms())
        if q > 0 and cation is None:
            cation = f
        elif q < 0 and anion is None:
            anion = f
    return cation, anion


def rdkit_frag_features(smi):
    """Return dict {MW, C, N, O, HB_donors} for a fragment SMILES."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    mh = Chem.AddHs(m)
    mw = Descriptors.MolWt(m)
    c = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "C")
    n = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "N")
    o = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "O")
    # HB donors: count -OH, -NH, -NH2 (heavy atoms with attached H on N or O).
    hb = 0
    for a in mh.GetAtoms():
        if a.GetSymbol() in ("N", "O"):
            for nb in a.GetNeighbors():
                if nb.GetSymbol() == "H":
                    hb += 1
                    break
    return {"MW": float(mw), "C": float(c), "N": float(n), "O": float(o), "HB": float(hb)}


def main():
    print("Loading existing physchem table...")
    phys = pd.read_csv(PHYS_CSV)
    phys["smi_canon"] = phys["smiles"].map(canon)
    existing_canon = set(phys["smi_canon"].dropna())
    print(f"  existing rows: {len(phys)} (unique canon SMILES: {len(existing_canon)})")

    print("Loading unified lignin SMILES...")
    uni = pd.read_csv(UNI_CSV)
    uni_canon = sorted({canon(s) for s in uni["smiles"].dropna() if canon(s)})
    missing = [s for s in uni_canon if s not in existing_canon]
    print(f"  unified unique SMILES: {len(uni_canon)}")
    print(f"  missing from physchem: {len(missing)}")

    # Medians of existing 28 for empirical columns (used as imputation fallback)
    medians = {col: float(phys[col].median()) for col in EMPIRICAL_COLS}
    print("Medians (used for empirical-column imputation where no ILThermo / literature hit):")
    for c, v in medians.items():
        print(f"  {c:15s} = {v:.4f}")

    # Preload ILThermo expanded for viscosity lookup
    print("Indexing ILThermo-expanded viscosity...")
    ilt = pd.read_csv(ILTHERMO_CSV)
    ilt["smi_canon"] = ilt["il_smiles"].map(canon)
    ilt_visc_298 = (
        ilt[(ilt["property_type"] == "Viscosity") & ilt["viscosity"].notna()
            & ilt["temperature"].between(293, 303)]
        .groupby("smi_canon")["viscosity"].median()
    )
    print(f"  ILThermo viscosity (293-303K) median available for {len(ilt_visc_298)} SMILES")

    # Build new rows for each missing SMILES
    new_rows = []
    audit_rows = []
    for smi in missing:
        cat_smi, an_smi = split_il(smi)
        if cat_smi is None or an_smi is None:
            print(f"  SKIP {smi}: could not split into cation/anion")
            continue
        cat = rdkit_frag_features(cat_smi)
        an = rdkit_frag_features(an_smi)
        m_whole = Chem.MolFromSmiles(smi)
        clogp = float(Crippen.MolLogP(m_whole)) if m_whole else medians.get("clogp", 0.0)

        # viscosity: prefer ILThermo median at ~298 K, else median-of-28
        visc_source = "median_28"
        if smi in ilt_visc_298.index:
            visc = float(ilt_visc_298.loc[smi])
            visc_source = "ilthermo_median"
        else:
            visc = medians["viscosity_298K"]

        row = {
            "smiles": smi,
            "kt_alpha": medians["kt_alpha"],
            "kt_beta": medians["kt_beta"],
            "clogp": clogp,
            "viscosity_298K": visc,
            "pKa": medians["pKa"],
            "conductivity": medians["conductivity"],
            "cat_MW": cat["MW"],
            "cat_C": cat["C"],
            "cat_N": cat["N"],
            "an_MW": an["MW"],
            "an_O": an["O"],
            "an_HB": an["HB"],
        }
        new_rows.append(row)

        # Audit: per-column source tag
        audit = {"smiles": smi,
                 "kt_alpha": "median_28", "kt_beta": "median_28",
                 "clogp": "rdkit", "viscosity_298K": visc_source,
                 "pKa": "median_28", "conductivity": "median_28",
                 "cat_MW": "rdkit", "cat_C": "rdkit", "cat_N": "rdkit",
                 "an_MW": "rdkit", "an_O": "rdkit", "an_HB": "rdkit"}
        audit_rows.append(audit)

    print(f"Built {len(new_rows)} new physchem rows.")

    # Merge and write
    phys_out = pd.concat(
        [phys.drop(columns=["smi_canon"]), pd.DataFrame(new_rows)],
        ignore_index=True,
    )
    phys_out = phys_out[PHYSCHEM_ORDER]
    phys_out.to_csv(PHYS_CSV, index=False)

    # Existing rows have all source = "measured" (originals from phys_prop Excel)
    audit_existing = [{"smiles": s, **{c: "measured" for c in PHYSCHEM_ORDER if c != "smiles"}}
                      for s in phys["smi_canon"].dropna()]
    audit_full = pd.DataFrame(audit_existing + audit_rows)
    audit_full = audit_full[PHYSCHEM_ORDER]
    audit_full.to_csv(AUDIT_CSV, index=False)

    # Verify coverage
    out_canon = set(canon(s) for s in phys_out["smiles"])
    unified_covered = sum(1 for s in uni_canon if s in out_canon)
    print()
    print(f"Wrote: {PHYS_CSV} ({len(phys_out)} rows)")
    print(f"Wrote: {AUDIT_CSV} (source tags per row/col)")
    print(f"Coverage: {unified_covered}/{len(uni_canon)} unified ILs have physchem now.")
    if unified_covered != len(uni_canon):
        missing_still = [s for s in uni_canon if s not in out_canon]
        print(f"STILL MISSING: {len(missing_still)}")
        for s in missing_still[:5]:
            print(f"  {s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
