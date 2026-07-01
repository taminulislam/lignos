#!/usr/bin/env python3
"""Build the unified LignoIL dataset.

Sources
-------
1) `data/LignoIL/Activity coeff and Excess Enthalpy for Imaging.xlsx`
   - Sheet `Datasheet`   : 28 ILs × 8 T, 7 core thermo properties (already cached).
   - Sheet `Physicochemical Properties` : 97 lignin-solubility rows with 15 physchem
     features (Kamlet-Taft α/β, CLogP, viscosity, pKa, conductivity, MW/atom counts).
2) `data/LignoIL/baran2024_lignin_data.csv` : 142 lignin-yield rows, biomass
   composition fields, IL short-names only.

Outputs (written to `data/LignoIL_unified/`)
-------------------------------------------
- `unified_lignin.csv`       — union of the two lignin tables, deduped by
                               (SMILES, biomass_source, T, time_min, solid_loading_pct),
                               with a `source` column {phys_prop, baran, both}.
- `il_physchem_features.csv` — 28 ILs × 12 physchem descriptors (IL-level).
- `cached_{train,val,test}.npz` — copies of the current cache with two new keys:
    * `physchem_feat`  (N, 12)   — IL-level descriptors joined by SMILES
    * `has_physchem`   (N,)      — bool, False where the IL is not in the physchem table.

Notes
-----
- Does NOT rewrite `targets`, `thermo_feat`, or any existing cache field — the
  existing 25D `thermo_feat` path keeps working unchanged. To use the new
  features, the model should concatenate `[thermo_feat, physchem_feat]`.
- Baran IL names (`[Ch] [Lys]`, `[Bmim] [OAc]`, …) are resolved to canonical
  SMILES via a normalized-name map built from the Phys Prop sheet. Unmatched
  rows are written to `unified_lignin.csv` with `smiles=NaN` and tagged in
  `unresolved_baran.csv` for manual review.
"""
from __future__ import annotations
import re, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
SRC = V5 / "data" / "LignoIL"
OUT = V5 / "data" / "LignoIL_unified"
OUT.mkdir(parents=True, exist_ok=True)

XLSX = SRC / "Activity coeff and Excess Enthalpy for Imaging.xlsx"
BARAN = SRC / "baran2024_lignin_data.csv"

PHYSCHEM_COLS = [
    "Kamlet Taft (α)", "Kamlet Taft (β)", "CLogP",
    "Viscosity (η/ mPa·s) @298.15K", "pKa", "Conductivity (S/m)",
    "cation molecular weight", "cation carbons count", "cation nitrogen count",
    "anion molecular weight", "anion oxygen count", "anion hydrogen bonding",
]
PHYSCHEM_SHORT = [
    "kt_alpha", "kt_beta", "clogp", "viscosity_298K", "pKa", "conductivity",
    "cat_MW", "cat_C", "cat_N", "an_MW", "an_O", "an_HB",
]
assert len(PHYSCHEM_COLS) == len(PHYSCHEM_SHORT) == 12


# ---------------------------------------------------------------------------
# Name normalization & SMILES resolution
# ---------------------------------------------------------------------------
def canon_smiles(smi):
    if not isinstance(smi, str):
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


_NAME_ALIASES = {
    # cation aliases → canonical token
    "amim": "amim", "bmim": "bmim", "emim": "emim", "hmim": "hmim",
    "mmim": "mmim", "pmim": "pmim", "omim": "omim", "bdmim": "bdmim",
    "ch": "ch", "choline": "ch",
    # anion aliases
    "oac": "oac", "meco2": "oac", "acetate": "oac",
    "cf3so3": "otf", "tfo": "otf", "otf": "otf",
    "lac": "lac", "lactate": "lac",
    "cl": "cl", "br": "br", "i": "i", "f": "f",
    "hso4": "hso4", "meso4": "meso4", "mesu": "meso4",
    "ntf2": "ntf2", "tf2n": "ntf2",
    "bf4": "bf4", "pf6": "pf6",
    "lys": "lys", "gly": "gly", "ala": "ala", "ser": "ser",
    "thr": "thr", "met": "met", "pro": "pro", "glc": "glc",
    "for": "for", "bz": "bz", "but": "but", "hex": "hex",
}

# Supplementary IL shorthand → canonical SMILES for ILs absent from Phys Prop.
# Rescues unresolved Baran rows so they retain a SMILES (physchem join will be
# has_physchem=False for these since they're not in the physchem table).
_CHOLINE = "C[N+](C)(C)CCO"
_DMEA = "C[NH+](C)CCO"
_SUPPLEMENTARY_SMILES = {
    "emim dmp": "CC[n+]1ccn(C)c1.COP(=O)([O-])OC",
    "mmim meso4": "Cn1cc[n+](C)c1.COS(=O)(=O)[O-]",
    # "Mim" in Baran shorthand means 3H-methylimidazolium (protonated).
    "mim hso4": "C[n+]1cc[nH]c1.OS(=O)(=O)[O-]",
    # CF3SO3 normalizes to "otf" via _NAME_ALIASES; use post-normalization key.
    "bmim otf": "CCCCn1cc[n+](C)c1.O=S(=O)([O-])C(F)(F)F",
    "c4h8so3hmim hso4": "OS(=O)(=O)CCCCn1cc[n+](C)c1.OS(=O)(=O)[O-]",
    "c2h4coohmim cl": "OC(=O)CCn1cc[n+](C)c1.[Cl-]",
    # DMEA (dimethylethanolammonium) salts
    "dmea oac": f"{_DMEA}.CC(=O)[O-]",
    "dmea for": f"{_DMEA}.[O-]C=O",
    "dmea c4h5o4": f"{_DMEA}.OC(=O)CCC(=O)[O-]",
    # Choline amino-acid / carboxylate ILs
    "ch gly": f"{_CHOLINE}.NCC(=O)[O-]",
    "ch ala": f"{_CHOLINE}.C[C@@H](N)C(=O)[O-]",
    "ch ser": f"{_CHOLINE}.N[C@@H](CO)C(=O)[O-]",
    "ch thr": f"{_CHOLINE}.C[C@@H](O)[C@H](N)C(=O)[O-]",
    "ch pro": f"{_CHOLINE}.O=C([O-])C1CCCN1",
    "ch met": f"{_CHOLINE}.CSCC[C@H](N)C(=O)[O-]",
    "ch phe": f"{_CHOLINE}.N[C@@H](Cc1ccccc1)C(=O)[O-]",
    "ch for": f"{_CHOLINE}.[O-]C=O",
    "ch bz":  f"{_CHOLINE}.O=C([O-])c1ccccc1",
    "ch but": f"{_CHOLINE}.CCCC(=O)[O-]",
    "ch hex": f"{_CHOLINE}.CCCCCC(=O)[O-]",
    "ch oct": f"{_CHOLINE}.CCCCCCCC(=O)[O-]",
    # "[Ch] [i-Oct]" splits on '-' to "i oct" — match post-normalization.
    "ch i oct": f"{_CHOLINE}.CC(C)CCCCC(=O)[O-]",
    "ch piv": f"{_CHOLINE}.CC(C)(C)C(=O)[O-]",
    "ch tfa": f"{_CHOLINE}.O=C([O-])C(F)(F)F",
    "ch nic": f"{_CHOLINE}.O=C([O-])c1cccnc1",
}


def normalize_il_name(name):
    """Lowercase, strip brackets/spaces/slashes, split cation/anion tokens, map aliases."""
    if not isinstance(name, str):
        return None
    s = name.lower().strip()
    s = s.replace("[", " ").replace("]", " ")
    s = re.sub(r"/.*$", "", s)  # drop "/H2O", "/DMSO" etc
    tokens = [t for t in re.split(r"\s+|-", s) if t]
    tokens = [_NAME_ALIASES.get(t, t) for t in tokens]
    return " ".join(tokens)


def build_name_to_smiles(pp):
    """Map normalized IL short-name → canonical SMILES, using Phys Prop + supplementary dict."""
    mp = {}
    for _, r in pp.iterrows():
        smi = canon_smiles(r.get("SMILES"))
        if not smi:
            continue
        for key_col in ["Ionic Liquid"]:
            raw = r.get(key_col)
            k = normalize_il_name(raw)
            if k and k not in mp:
                mp[k] = smi
        # Try "cation anion" composition too
        cat = str(r.get("Cation") or "").lower()
        ani = str(r.get("Anion") or "").lower()
        for anon_token in [ani, _NAME_ALIASES.get(ani, ani)]:
            key = normalize_il_name(f"{cat} {anon_token}")
            if key and key not in mp:
                mp[key] = smi
    # Merge supplementary map (canonicalized). Supplementary entries add ILs not
    # present in Phys Prop; physchem features will be unavailable for them.
    for k, raw_smi in _SUPPLEMENTARY_SMILES.items():
        canon = canon_smiles(raw_smi)
        if canon and k not in mp:
            mp[k] = canon
    return mp


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
def load_phys_prop():
    df = pd.read_excel(XLSX, sheet_name="Physicochemical Properties")
    df.columns = [c.strip().rstrip("\xa0").strip() for c in df.columns]
    df["smiles"] = df["SMILES"].map(canon_smiles)
    # Tag IL-mixture rows (e.g. "[amim][Cl]/DMSO", "[BMIM][OAc]/H2O"). Their
    # physchem values refer to the mixture, not the pure IL, so must be excluded
    # when building the IL-level physchem table.
    df["is_mixture"] = df["Ionic Liquid"].astype(str).str.contains("/")
    return df


def load_baran():
    df = pd.read_csv(BARAN)
    df = df.dropna(subset=["il_name", "yield_pct"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Extract the two lignin tables in a common schema
# ---------------------------------------------------------------------------
COMMON_COLS = [
    "smiles", "biomass_source", "temperature_C", "time_min",
    "solid_loading_pct", "il_conc_in_solvent_pct",
    "lignin_value_pct", "measurement_type",
    "cellulose_pct", "hemicellulose_pct", "lignin_in_biomass_pct",
    "source", "il_name_raw",
]


def extract_phys_prop_lignin(pp):
    """Phys Prop rows measure solubility wt% (Lignin Solubility %)."""
    out = pd.DataFrame({
        "smiles": pp["smiles"],
        "biomass_source": pp["Biomass Source"],
        "temperature_C": pd.to_numeric(pp["Temperature (°C)"], errors="coerce"),
        "time_min": pd.to_numeric(pp["Time (min)"], errors="coerce"),
        "solid_loading_pct": pd.to_numeric(pp["Solid Loading (%)"], errors="coerce"),
        "il_conc_in_solvent_pct": np.nan,
        "lignin_value_pct": pd.to_numeric(pp["Lignin Solubility (%)"], errors="coerce"),
        "measurement_type": "solubility_wt_pct",
        "cellulose_pct": np.nan,
        "hemicellulose_pct": np.nan,
        "lignin_in_biomass_pct": np.nan,
        "source": "phys_prop",
        "il_name_raw": pp["Ionic Liquid"],
    })
    return out[COMMON_COLS]


def extract_baran_lignin(baran, name_map):
    """Baran rows measure extraction yield % (yield_pct). il_conc is IL
    weight fraction in the solvent (IL + co-solvent), NOT biomass solid loading.
    """
    norm = baran["il_name"].map(normalize_il_name)
    smiles = norm.map(name_map.get)
    unresolved = baran[smiles.isna()].copy()
    unresolved["normalized_name"] = norm[smiles.isna()]
    unresolved.to_csv(OUT / "unresolved_baran.csv", index=False)
    out = pd.DataFrame({
        "smiles": smiles,
        "biomass_source": baran["biomass_type"],
        "temperature_C": pd.to_numeric(baran["temp_C"], errors="coerce"),
        "time_min": pd.to_numeric(baran["time_h"], errors="coerce") * 60.0,
        "solid_loading_pct": np.nan,                # Baran does not report biomass solid loading
        "il_conc_in_solvent_pct": pd.to_numeric(baran["il_conc"], errors="coerce") * 100.0,
        "lignin_value_pct": pd.to_numeric(baran["yield_pct"], errors="coerce"),
        "measurement_type": "extraction_yield_pct",
        "cellulose_pct": pd.to_numeric(baran["perc_cellulose"], errors="coerce"),
        "hemicellulose_pct": pd.to_numeric(baran["perc_hemicellulose"], errors="coerce"),
        "lignin_in_biomass_pct": pd.to_numeric(baran["perc_lignins"], errors="coerce"),
        "source": "baran",
        "il_name_raw": baran["il_name"],
    })
    return out[COMMON_COLS], unresolved


def unify(pp_lig, baran_lig):
    """Union + dedup. Dropping rows with null SMILES — they cannot be trained on.
    Dedup is only meaningful between rows of the same measurement_type; we never
    merge a solubility row with an extraction-yield row even if process conditions match.
    """
    joined = pd.concat([pp_lig, baran_lig], axis=0, ignore_index=True)
    n_before_drop = len(joined)
    joined = joined.dropna(subset=["smiles"]).reset_index(drop=True)
    n_null = n_before_drop - len(joined)
    print(f"  Dropped {n_null} rows with null SMILES (unresolved Baran IL names).")

    dup_key = ["smiles", "biomass_source", "temperature_C", "time_min",
               "solid_loading_pct", "measurement_type"]
    merged_rows = []
    kept_mask = np.ones(len(joined), dtype=bool)
    for _, grp in joined.dropna(subset=["smiles", "biomass_source"]).groupby(dup_key, dropna=False):
        if len(grp) <= 1:
            continue
        first_idx = grp.index[0]
        merged = grp.iloc[0].copy()
        for col in grp.columns:
            if pd.isna(merged[col]):
                for i in range(1, len(grp)):
                    v = grp.iloc[i][col]
                    if not pd.isna(v):
                        merged[col] = v
                        break
        merged["source"] = "both"
        merged_rows.append((first_idx, merged))
        for i in grp.index[1:]:
            kept_mask[i] = False
    for idx, row in merged_rows:
        joined.loc[idx] = row
    deduped = joined[kept_mask].reset_index(drop=True)
    return deduped


# ---------------------------------------------------------------------------
# Physchem feature table (IL-level)
# ---------------------------------------------------------------------------
def _count_atoms_in_cation(smi_salt):
    """Heuristic: split salt SMILES on '.', find the cation fragment (contains [n+]
    or [N+]), and count nitrogen atoms. Used to impute cat_N when missing."""
    if not isinstance(smi_salt, str):
        return None
    frags = smi_salt.split(".")
    cat = None
    for f in frags:
        if "[n+]" in f or "[N+]" in f or "[NH+]" in f or "[NH2+]" in f:
            cat = f
            break
    if cat is None:
        return None
    mol = Chem.MolFromSmiles(cat)
    if mol is None:
        return None
    return int(sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N"))


def build_physchem_table(pp):
    """Build a 28-IL physchem table. CRITICAL: exclude IL-mixture rows
    ("/H2O", "/DMSO") because their values refer to the mixture, not the pure IL.
    """
    pure = pp[(~pp["is_mixture"]) & pp["smiles"].notna()]
    n_excluded = pp["is_mixture"].sum()
    print(f"  Physchem table: excluded {n_excluded} IL-mixture rows before aggregation.")
    rows = []
    inconsistencies = 0
    for smi, grp in pure.groupby("smiles"):
        rec = {"smiles": smi}
        for long_name, short in zip(PHYSCHEM_COLS, PHYSCHEM_SHORT):
            vals = pd.to_numeric(grp[long_name], errors="coerce").dropna().unique()
            if len(vals) == 0:
                rec[short] = np.nan
            elif len(vals) == 1:
                rec[short] = float(vals[0])
            else:
                # Still inconsistent even among pure-IL rows — take the median
                # and count it as a diagnostic.
                rec[short] = float(np.median(vals))
                inconsistencies += 1
        # Impute cat_N from SMILES if still missing
        if pd.isna(rec["cat_N"]):
            n = _count_atoms_in_cation(smi)
            if n is not None:
                rec["cat_N"] = float(n)
        rows.append(rec)
    print(f"  Physchem table: {inconsistencies} (IL, feature) pairs still had "
          f"multiple values among pure rows — median used.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regenerate cached npz with physchem_feat joined by SMILES
# ---------------------------------------------------------------------------
def regenerate_cache(physchem_df):
    phys_map = {row["smiles"]: np.array(
        [row[c] for c in PHYSCHEM_SHORT], dtype=np.float32
    ) for _, row in physchem_df.iterrows()}

    stats = {}
    for split in ["train", "val", "test"]:
        src_path = SRC / f"cached_{split}.npz"
        dst_path = OUT / f"cached_{split}.npz"
        src = np.load(src_path, allow_pickle=True)
        data = {k: src[k] for k in src.files}

        smiles = np.array([s.decode() if isinstance(s, bytes) else s for s in data["smiles"]])
        N = len(smiles)
        physchem_feat = np.zeros((N, len(PHYSCHEM_SHORT)), dtype=np.float32)
        has_physchem = np.zeros(N, dtype=bool)
        matched_smiles = set()
        # Cached SMILES are stored raw; canonicalize on lookup so ordering / HSO4 tautomers match.
        for i, smi in enumerate(smiles):
            canon = canon_smiles(smi) or smi
            vec = phys_map.get(canon)
            if vec is None:
                vec = phys_map.get(smi)
            if vec is not None:
                physchem_feat[i] = np.where(np.isnan(vec), 0.0, vec)
                has_physchem[i] = True
                matched_smiles.add(canon)

        data["physchem_feat"] = physchem_feat
        data["has_physchem"] = has_physchem
        np.savez(dst_path, **data)
        stats[split] = {
            "rows": N,
            "rows_with_physchem": int(has_physchem.sum()),
            "unique_smiles_matched": len(matched_smiles),
            "unique_smiles_in_split": int(len(set(smiles))),
        }
        print(f"  {split:5s}: {has_physchem.sum():>5d}/{N} rows matched  "
              f"(IL coverage: {len(matched_smiles)}/{len(set(smiles))})")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Build unified LignoIL dataset")
    print("=" * 60)

    pp = load_phys_prop()
    baran = load_baran()
    print(f"Phys Prop: {len(pp)} rows, {pp['smiles'].nunique()} unique SMILES")
    print(f"Baran    : {len(baran)} rows, {baran['il_name'].nunique()} unique IL names")

    name_map = build_name_to_smiles(pp)
    print(f"Built IL-name → SMILES map: {len(name_map)} entries")

    pp_lig = extract_phys_prop_lignin(pp)
    baran_lig, unresolved = extract_baran_lignin(baran, name_map)
    print(f"Baran rows resolved to SMILES: "
          f"{baran_lig['smiles'].notna().sum()}/{len(baran_lig)}  "
          f"(unresolved → unresolved_baran.csv)")

    unified = unify(pp_lig, baran_lig)
    n_pp = (unified["source"] == "phys_prop").sum()
    n_bar = (unified["source"] == "baran").sum()
    n_both = (unified["source"] == "both").sum()
    print(f"\nUnified lignin table: {len(unified)} rows "
          f"(phys_prop={n_pp}, baran={n_bar}, both={n_both})")
    print(f"  Unique SMILES : {unified['smiles'].nunique()}")
    print(f"  Unique biomass: {unified['biomass_source'].nunique()}")
    print(f"  measurement_type counts:")
    print(unified["measurement_type"].value_counts().to_string())

    unified_path = OUT / "unified_lignin.csv"
    unified.to_csv(unified_path, index=False)
    print(f"  -> {unified_path}")

    physchem = build_physchem_table(pp)
    physchem_path = OUT / "il_physchem_features.csv"
    physchem.to_csv(physchem_path, index=False)
    print(f"\nPhyschem feature table: {len(physchem)} ILs × {len(PHYSCHEM_SHORT)} features")
    print(f"  -> {physchem_path}")

    print(f"\nRegenerating cached npz with physchem_feat joined by SMILES...")
    stats = regenerate_cache(physchem)

    summary = {
        "inputs": {
            "phys_prop_rows": int(len(pp)),
            "baran_rows": int(len(baran)),
        },
        "unified_lignin": {
            "rows": int(len(unified)),
            "phys_prop_only": int(n_pp),
            "baran_only": int(n_bar),
            "overlap_merged": int(n_both),
            "unique_smiles": int(unified["smiles"].nunique()),
            "unique_biomass": int(unified["biomass_source"].nunique()),
        },
        "physchem_il_table": {
            "n_ils": int(len(physchem)),
            "features": PHYSCHEM_SHORT,
        },
        "cache_coverage": stats,
    }
    with open(OUT / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {OUT / 'build_summary.json'}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
