"""Day-1 deliverable: build the IL catalog for virtual screening.

Produces a CSV of enumerated (cation, anion) IL SMILES, programmatically
generated from:
  - Cation families: imidazolium, pyrrolidinium, pyridinium, piperidinium,
    ammonium (quaternary), phosphonium, cholinium, with alkyl substituent
    variants (methyl, ethyl, propyl, butyl, pentyl, hexyl, heptyl, octyl,
    decyl, dodecyl, tetradecyl) + common functional variants
  - Anion catalog: halides, sulfonates, sulfates, carboxylates (aliphatic +
    aromatic), 20 amino-acid anions, fluorinated anions, phosphates

Output:
  data/virtual_screen/il_catalog.csv with columns:
      cation_family, cation_name, anion_family, anion_name,
      cation_smiles, anion_smiles, il_smiles, canonical_smiles

Uses il_name_resolver.py + RDKit for SMILES assembly and canonicalization.
"""
from __future__ import annotations
import csv, itertools, sys
from pathlib import Path

from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts" / "lit_scrape"))
from il_name_resolver import ALKYL_MAP, resolve_cation, resolve_anion

OUT_CSV = PROJECT_ROOT / "data" / "virtual_screen" / "il_catalog.csv"


# ==========================================================================
# Cation generators — each yields (family, name, smiles) tuples
# ==========================================================================
# Common alkyl substituents (short + medium) — covers C1..C12, tetradecyl
ALKYL_SHORT = ["methyl", "ethyl", "propyl", "butyl", "pentyl", "hexyl",
                "heptyl", "octyl", "decyl", "dodecyl"]
ALKYL_LONG = ALKYL_SHORT + ["tetradecyl", "hexadecyl"]


def gen_imidazolium():
    """1-R-3-methyl-imidazolium with R varying, plus 1,2,3-trimethyl & allyl variants."""
    out = []
    for r in ALKYL_SHORT:
        name = f"1-{r}-3-methylimidazolium"
        smi = resolve_cation(name)
        if smi:
            out.append(("imidazolium", name, smi))
    # 2-methyl variants (1,2-dimethyl-3-R-imidazolium)
    for r in ["butyl", "hexyl", "octyl"]:
        name = f"1-{r}-2,3-dimethylimidazolium"
        smi = resolve_cation(name)
        if smi:
            out.append(("imidazolium", name, smi))
    # Allyl variants
    for r in ["methyl", "butyl"]:
        name = f"1-allyl-3-{r}imidazolium"
        cat = f"C=CC[n+]1ccn({ALKYL_MAP[r]})c1"
        m = Chem.MolFromSmiles(cat)
        if m: out.append(("imidazolium", name, Chem.MolToSmiles(m)))
    return out


def gen_pyrrolidinium():
    """1,1-dialkyl-pyrrolidinium."""
    out = []
    for r1 in ["methyl", "ethyl", "butyl", "hexyl", "octyl"]:
        for r2 in ["methyl", "ethyl", "butyl"]:
            if r1 == r2: continue
            name = f"1-{r1}-1-{r2}pyrrolidinium"
            smi = resolve_cation(name)
            if smi:
                out.append(("pyrrolidinium", name, smi))
    return out


def gen_piperidinium():
    out = []
    for r1 in ["methyl", "ethyl", "butyl", "hexyl"]:
        for r2 in ["methyl", "ethyl"]:
            if r1 == r2: continue
            name = f"1-{r1}-1-{r2}piperidinium"
            smi = resolve_cation(name)
            if smi:
                out.append(("piperidinium", name, smi))
    return out


def gen_pyridinium():
    out = []
    for r in ALKYL_SHORT[:8]:
        name = f"1-{r}pyridinium"
        smi = resolve_cation(name)
        if smi:
            out.append(("pyridinium", name, smi))
    return out


def gen_ammonium():
    """Tetraalkyl and tri-alkyl(methyl) ammoniums."""
    out = []
    # Tetra-same
    for r in ["methyl", "ethyl", "propyl", "butyl", "hexyl", "octyl"]:
        name = f"tetra{r}ammonium"
        smi = resolve_cation(name)
        if smi: out.append(("ammonium", name, smi))
    # Tri-alkyl-methyl (mixed) — e.g., trihexyl(methyl)ammonium
    for r in ["butyl", "hexyl", "octyl", "decyl"]:
        for r2 in ["methyl", "ethyl"]:
            if r == r2: continue
            name = f"tri{r}({r2})ammonium"
            # Manual SMILES — no OPSIN for this form
            r1 = ALKYL_MAP[r]; r2_smi = ALKYL_MAP[r2]
            smi = f"{r1}[N+]({r1})({r1}){r2_smi}"
            m = Chem.MolFromSmiles(smi)
            if m: out.append(("ammonium", name, Chem.MolToSmiles(m)))
    return out


def gen_phosphonium():
    """Tetraalkyl and mixed alkyl phosphoniums."""
    out = []
    for r in ["methyl", "ethyl", "butyl", "hexyl", "octyl"]:
        name = f"tetra{r}phosphonium"
        smi = resolve_cation(name)
        if smi: out.append(("phosphonium", name, smi))
    # Trihexyl(tetradecyl)phosphonium — CYPHOS 101 family
    for pair in [("hexyl", "tetradecyl"), ("butyl", "hexadecyl"), ("butyl", "decyl")]:
        r, r2 = pair
        name = f"tri{r}({r2})phosphonium"
        smi = resolve_cation(name)
        if smi: out.append(("phosphonium", name, smi))
    return out


def gen_cholinium():
    """Cholinium + simple analogs."""
    out = [("cholinium", "cholinium", "C[N+](C)(C)CCO"),
           ("cholinium", "(2-hydroxypropyl)trimethylammonium", "C[N+](C)(C)CC(C)O"),
           ("cholinium", "ethanolammonium", "[NH3+]CCO"),
           ("cholinium", "dimethylethanolammonium", "C[NH+](C)CCO")]
    return out


# ==========================================================================
# Anion catalog (with literal SMILES)
# ==========================================================================
ANION_CATALOG = {
    # halides
    "halide": [
        ("chloride", "[Cl-]"),
        ("bromide", "[Br-]"),
        ("iodide", "[I-]"),
        ("fluoride", "[F-]"),
    ],
    # fluorinated anions
    "fluorinated": [
        ("tetrafluoroborate", "F[B-](F)(F)F"),
        ("hexafluorophosphate", "F[P-](F)(F)(F)(F)F"),
        ("bis(trifluoromethylsulfonyl)imide", "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F"),
        ("trifluoromethanesulfonate", "O=S(=O)([O-])C(F)(F)F"),
        ("trifluoroacetate", "O=C([O-])C(F)(F)F"),
        ("tris(pentafluoroethyl)trifluorophosphate",
            "F[P-](F)(F)(C(F)(F)C(F)(F)F)(C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F"),
    ],
    # sulfonate
    "sulfonate": [
        ("methanesulfonate", "CS(=O)(=O)[O-]"),
        ("ethanesulfonate", "CCS(=O)(=O)[O-]"),
        ("p-toluenesulfonate", "Cc1ccc(S(=O)(=O)[O-])cc1"),
        ("benzenesulfonate", "O=S(=O)([O-])c1ccccc1"),
    ],
    # sulfate
    "sulfate": [
        ("methyl sulfate", "COS(=O)(=O)[O-]"),
        ("ethyl sulfate", "CCOS(=O)(=O)[O-]"),
        ("hydrogen sulfate", "OS(=O)(=O)[O-]"),
    ],
    # aliphatic carboxylates
    "carboxylate_aliphatic": [
        ("formate", "O=C[O-]"),
        ("acetate", "CC(=O)[O-]"),
        ("propionate", "CCC(=O)[O-]"),
        ("butyrate", "CCCC(=O)[O-]"),
        ("hexanoate", "CCCCCC(=O)[O-]"),
        ("octanoate", "CCCCCCCC(=O)[O-]"),
        ("decanoate", "CCCCCCCCCC(=O)[O-]"),
        ("pivalate", "CC(C)(C)C(=O)[O-]"),
        ("lactate", "CC(O)C(=O)[O-]"),
        ("glycolate", "OCC(=O)[O-]"),
        ("pyruvate", "CC(=O)C(=O)[O-]"),
    ],
    # aromatic carboxylates
    "carboxylate_aromatic": [
        ("benzoate", "O=C([O-])c1ccccc1"),
        ("salicylate", "O=C([O-])c1ccccc1O"),
        ("nicotinate", "O=C([O-])c1ccncc1"),
    ],
    # phosphate / phosphonate
    "phosphate": [
        ("dimethyl phosphate", "COP(=O)([O-])OC"),
        ("diethyl phosphate", "CCOP(=O)([O-])OCC"),
        ("dibutyl phosphate", "CCCCOP(=O)([O-])OCCCC"),
    ],
    # pseudohalides
    "pseudohalide": [
        ("dicyanamide", "N#C[N-]C#N"),
        ("thiocyanate", "N#C[S-]"),
        ("tricyanomethanide", "N#C[C-](C#N)C#N"),
        ("nitrate", "[O-][N+](=O)[O-]"),
    ],
    # 20 proteinogenic amino-acid anions (deprotonated -COOH)
    "amino_acid": [
        ("glycinate", "[O-]C(=O)CN"),
        ("alaninate", "C[C@@H](N)C(=O)[O-]"),
        ("valinate", "CC(C)[C@@H](N)C(=O)[O-]"),
        ("leucinate", "CC(C)C[C@@H](N)C(=O)[O-]"),
        ("isoleucinate", "CC[C@H](C)[C@@H](N)C(=O)[O-]"),
        ("serinate", "OC[C@@H](N)C(=O)[O-]"),
        ("threoninate", "C[C@@H](O)[C@H](N)C(=O)[O-]"),
        ("cysteinate", "SC[C@@H](N)C(=O)[O-]"),
        ("methioninate", "CSCC[C@H](N)C(=O)[O-]"),
        ("prolinate", "O=C([O-])[C@@H]1CCCN1"),
        ("phenylalaninate", "N[C@@H](Cc1ccccc1)C(=O)[O-]"),
        ("tyrosinate", "N[C@@H](Cc1ccc(O)cc1)C(=O)[O-]"),
        ("tryptophanate", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)[O-]"),
        ("histidinate", "N[C@@H](Cc1c[nH]cn1)C(=O)[O-]"),
        ("lysinate", "NCCCC[C@H](N)C(=O)[O-]"),
        ("arginateate", "N[C@@H](CCCNC(=N)N)C(=O)[O-]"),
        ("aspartate", "[O-]C(=O)C[C@H](N)C(=O)[O-]"),
        ("glutamate", "[O-]C(=O)CC[C@H](N)C(=O)[O-]"),
        ("asparaginate", "NC(=O)C[C@@H](N)C(=O)[O-]"),
        ("glutaminate", "NC(=O)CC[C@@H](N)C(=O)[O-]"),
    ],
    # misc carboxylates common in biomass-extraction ILs
    "misc": [
        ("succinate (monoanion)", "O=C([O-])CCC(=O)O"),
        ("maleate (monoanion)", "O=C([O-])/C=C\\C(=O)O"),
        ("oxalate (monoanion)", "O=C([O-])C(=O)O"),
        ("gluconate", "OCC(O)C(O)C(O)C(O)C(=O)[O-]"),
        ("citrate (monoanion)", "O=C([O-])CC(O)(CC(=O)O)C(=O)O"),
    ],
}


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Gather cations
    cations = []
    for gen in (gen_imidazolium, gen_pyrrolidinium, gen_piperidinium,
                 gen_pyridinium, gen_ammonium, gen_phosphonium, gen_cholinium):
        family_out = gen()
        cations.extend(family_out)
    # Dedup cations by canonical SMILES
    seen_cat = {}
    for fam, name, smi in cations:
        c = canon(smi)
        if c and c not in seen_cat:
            seen_cat[c] = (fam, name, c)
    cations_dedup = list(seen_cat.values())
    print(f"Unique cations: {len(cations_dedup)}")

    # Gather anions
    anions = []
    for fam, entries in ANION_CATALOG.items():
        for name, smi in entries:
            c = canon(smi)
            if c: anions.append((fam, name, c))
    # Dedup
    seen_an = {}
    for fam, name, smi in anions:
        if smi not in seen_an:
            seen_an[smi] = (fam, name, smi)
    anions_dedup = list(seen_an.values())
    print(f"Unique anions: {len(anions_dedup)}")

    # Enumerate cation × anion
    total = 0
    seen_il = set()
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cation_family", "cation_name", "cation_smiles",
                     "anion_family", "anion_name", "anion_smiles",
                     "il_smiles", "canonical_smiles"])
        for (cf, cn, cs), (af, an, asmi) in itertools.product(cations_dedup, anions_dedup):
            il = f"{cs}.{asmi}"
            cil = canon(il)
            if cil is None or cil in seen_il:
                continue
            seen_il.add(cil)
            w.writerow([cf, cn, cs, af, an, asmi, il, cil])
            total += 1

    print(f"\nTotal unique IL pairs: {total}")
    print(f"Catalog written → {OUT_CSV}")

    # Print cation-family × anion-family breakdown
    print("\nCation family sizes:")
    from collections import Counter
    cat_family_ct = Counter(cf for cf, _, _ in cations_dedup)
    for f, n in sorted(cat_family_ct.items(), key=lambda x: -x[1]):
        print(f"  {f:25s}: {n}")
    print("\nAnion family sizes:")
    an_family_ct = Counter(af for af, _, _ in anions_dedup)
    for f, n in sorted(an_family_ct.items(), key=lambda x: -x[1]):
        print(f"  {f:25s}: {n}")


if __name__ == "__main__":
    main()
