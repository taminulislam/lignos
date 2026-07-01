"""Helper: resolve IUPAC-style systematic IL names to canonical SMILES.

Covers patterns like:
  "1-butyl-3-methylimidazolium"           → CCCC[n+]1ccn(C)c1
  "1-ethyl-3-methylimidazolium"           → CC[n+]1ccn(C)c1
  "1-butyl-1-methylpyrrolidinium"         → CCCC[N+]1(C)CCCC1
  "tetrafluoroborate"                     → F[B-](F)(F)F
  "bis(trifluoromethylsulfonyl)imide"     → O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F

Works with standalone strings OR cation+anion pairs in a single sentence.
"""
from __future__ import annotations
import re
from typing import Optional

from rdkit import Chem

try:
    from py2opsin import py2opsin as _opsin
    _HAS_OPSIN = True
except Exception:  # Java missing or lib not installed
    _opsin = None
    _HAS_OPSIN = False


ALKYL_MAP = {
    "methyl": "C", "ethyl": "CC", "propyl": "CCC", "butyl": "CCCC",
    "pentyl": "CCCCC", "hexyl": "CCCCCC", "heptyl": "CCCCCCC",
    "octyl": "CCCCCCCC", "nonyl": "CCCCCCCCC", "decyl": "CCCCCCCCCC",
    "dodecyl": "CCCCCCCCCCCC", "tetradecyl": "CCCCCCCCCCCCCC",
    "hexadecyl": "CCCCCCCCCCCCCCCC", "allyl": "C=CC",
}

CATION_TEMPLATES = {
    "imidazolium":   "{R1}[n+]1cc{n_sub}c1",   # positions 1 and 3: R1 and R3
    "pyrrolidinium": "{R1}[N+]1({R2})CCCC1",   # 1,1-substituted
    "piperidinium":  "{R1}[N+]1({R2})CCCCC1",
    "pyridinium":    "{R1}[n+]1ccccc1",
    "morpholinium":  "{R1}[N+]1({R2})CCOCC1",
    "ammonium":      "{R1}[N+]({R2})({R3}){R4}",
    "phosphonium":   "{R1}[P+]({R2})({R3}){R4}",
}

ANION_MAP = {
    "chloride": "[Cl-]", "bromide": "[Br-]", "iodide": "[I-]", "fluoride": "[F-]",
    "tetrafluoroborate": "F[B-](F)(F)F",
    "hexafluorophosphate": "F[P-](F)(F)(F)(F)F",
    "bis(trifluoromethylsulfonyl)imide": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "bis(trifluoromethanesulfonyl)imide": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "triflate": "O=S(=O)([O-])C(F)(F)F",
    "trifluoromethanesulfonate": "O=S(=O)([O-])C(F)(F)F",
    "trifluoromethylsulfonate": "O=S(=O)([O-])C(F)(F)F",
    "dicyanamide": "N#C[N-]C#N",
    "tricyanomethanide": "N#C[C-](C#N)C#N",
    "thiocyanate": "N#C[S-]",
    "methylsulfate": "COS(=O)(=O)[O-]",
    "methyl sulfate": "COS(=O)(=O)[O-]",
    "ethylsulfate": "CCOS(=O)(=O)[O-]",
    "ethyl sulfate": "CCOS(=O)(=O)[O-]",
    "hydrogensulfate": "O=S(=O)([O-])O",
    "hydrogen sulfate": "O=S(=O)([O-])O",
    "acetate": "CC(=O)[O-]",
    "formate": "O=C[O-]",
    "nitrate": "[O-][N+](=O)[O-]",
    "perchlorate": "[O-]Cl(=O)(=O)=O",
    "azide": "[N-]=[N+]=[N-]",
    "tris(pentafluoroethyl)trifluorophosphate": "F[P-](F)(F)(C(F)(F)F)(C(F)(F)F)C(F)(F)F",
    "tosylate": "Cc1ccc(S(=O)(=O)[O-])cc1",
    "tolulenesulfonate": "Cc1ccc(S(=O)(=O)[O-])cc1",
    "p-toluenesulfonate": "Cc1ccc(S(=O)(=O)[O-])cc1",
    "dimethylphosphate": "COP(=O)([O-])OC",
    "diethylphosphate": "CCOP(=O)([O-])OCC",
    "glycinate": "NCC(=O)[O-]",
    "alaninate": "C[C@@H](N)C(=O)[O-]",
    "prolinate": "O=C([O-])C1CCCN1",
    "phenylalaninate": "N[C@@H](Cc1ccccc1)C(=O)[O-]",
}


def _canon(smi: Optional[str]) -> Optional[str]:
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def resolve_cation(phrase: str) -> Optional[str]:
    """Return cation SMILES from a systematic name phrase, or None."""
    p = phrase.lower()
    # imidazolium: e.g., "1-butyl-3-methylimidazolium"
    m = re.search(r"1-(\w+)-3-(\w+)imidazolium", p)
    if m and m.group(1) in ALKYL_MAP and m.group(2) in ALKYL_MAP:
        r1 = ALKYL_MAP[m.group(1)]
        r3 = ALKYL_MAP[m.group(2)]
        return f"{r1}[n+]1ccn({r3})c1"
    m = re.search(r"1-(\w+)-2,3-dimethylimidazolium", p)
    if m and m.group(1) in ALKYL_MAP:
        r1 = ALKYL_MAP[m.group(1)]
        return f"{r1}[n+]1ccn(C)c1C"
    # pyrrolidinium: "1-butyl-1-methylpyrrolidinium"
    m = re.search(r"1-(\w+)-1-(\w+)pyrrolidinium", p)
    if m and m.group(1) in ALKYL_MAP and m.group(2) in ALKYL_MAP:
        r1 = ALKYL_MAP[m.group(1)]
        r2 = ALKYL_MAP[m.group(2)]
        return f"{r1}[N+]1({r2})CCCC1"
    # piperidinium
    m = re.search(r"1-(\w+)-1-(\w+)piperidinium", p)
    if m and m.group(1) in ALKYL_MAP and m.group(2) in ALKYL_MAP:
        r1 = ALKYL_MAP[m.group(1)]
        r2 = ALKYL_MAP[m.group(2)]
        return f"{r1}[N+]1({r2})CCCCC1"
    # N-alkylpyridinium
    m = re.search(r"1-(\w+)pyridinium", p)
    if m and m.group(1) in ALKYL_MAP:
        return f"{ALKYL_MAP[m.group(1)]}[n+]1ccccc1"
    # tetraalkylammonium: "tetrabutylammonium"
    m = re.search(r"tetra(\w+)ammonium", p)
    if m and m.group(1) in ALKYL_MAP:
        r = ALKYL_MAP[m.group(1)]
        return f"{r}[N+]({r})({r}){r}"
    # choline / cholinium
    if "choline" in p or "cholinium" in p or ("2-hydroxyethyl" in p and "trimethyl" in p):
        return "C[N+](C)(C)CCO"
    # trialkyl(alkyl)phosphonium — e.g. trihexyl(tetradecyl)phosphonium
    m = re.search(r"tri(\w+?)\s*\(\s*(\w+?)\s*\)phosphonium", p)
    if m and m.group(1) in ALKYL_MAP and m.group(2) in ALKYL_MAP:
        r1, r2 = ALKYL_MAP[m.group(1)], ALKYL_MAP[m.group(2)]
        return f"{r1}[P+]({r1})({r1}){r2}"
    # trialkyl(alkyl)ammonium
    m = re.search(r"tri(\w+?)\s*\(\s*(\w+?)\s*\)ammonium", p)
    if m and m.group(1) in ALKYL_MAP and m.group(2) in ALKYL_MAP:
        r1, r2 = ALKYL_MAP[m.group(1)], ALKYL_MAP[m.group(2)]
        return f"{r1}[N+]({r1})({r1}){r2}"
    # trialkylalkylphosphonium (no parentheses) — e.g. trihexyltetradecylphosphonium.
    # MUST check this before tetraalkylphosphonium because "tetra" also appears inside
    # "tetradecyl" and would be spuriously matched by the tetra-regex.
    _ALKYLS = "methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl|dodecyl|tetradecyl|hexadecyl"
    m = re.search(rf"tri({_ALKYLS})({_ALKYLS})phosphonium", p)
    if m:
        r1, r2 = ALKYL_MAP[m.group(1)], ALKYL_MAP[m.group(2)]
        return f"{r1}[P+]({r1})({r1}){r2}"
    # tetraalkylphosphonium — e.g. tetrabutylphosphonium, tetraethylphosphonium.
    # Use word boundary so we don't match inside "trihexyltetradecyl..."
    m = re.search(r"\btetra(\w+)phosphonium", p)
    if m and m.group(1) in ALKYL_MAP:
        r = ALKYL_MAP[m.group(1)]
        return f"{r}[P+]({r})({r}){r}"
    return None


def resolve_anion(phrase: str) -> Optional[str]:
    """Return anion SMILES from a systematic name phrase, or None."""
    p = phrase.lower()
    for name, smi in ANION_MAP.items():
        if name in p:
            return smi
    return None


def _opsin_salt(phrase: str) -> Optional[str]:
    """Ask OPSIN to resolve the whole phrase. Only return if it yields a
    cation.anion salt (net charge 0 and at least one positive + one negative
    fragment). Guards against OPSIN returning only the cation SMILES."""
    if not _HAS_OPSIN or not phrase:
        return None
    try:
        smi = _opsin(phrase, output_format="SMILES")
    except Exception:
        return None
    if not smi or not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) < 2:
        return None
    pos = neg = False
    for f in frags:
        q = sum(a.GetFormalCharge() for a in f.GetAtoms())
        if q > 0: pos = True
        elif q < 0: neg = True
    if not (pos and neg):
        return None
    return _canon(smi)


def resolve_il(phrase: str) -> Optional[str]:
    """Full IL SMILES (cation.anion canonicalized) from a phrase.

    Strategy: rule-based cation+anion resolve first (fast, no Java). If either
    fails, try OPSIN on the whole phrase (handles systematic names the rules
    miss, e.g., 'trihexyl(tetradecyl)phosphonium dicyanamide')."""
    cat = resolve_cation(phrase)
    an = resolve_anion(phrase)
    if cat and an:
        return _canon(f"{cat}.{an}")
    # OPSIN fallback on the whole phrase
    salt = _opsin_salt(phrase)
    if salt:
        return salt
    # Mixed: rules found one side, OPSIN can fill the other
    if cat and not an and _HAS_OPSIN:
        for tail in re.findall(r"\b[a-z][a-z\-()\s]{3,40}(?:ate|ide|ite|onium|phosphate|sulfonate)\b", phrase.lower()):
            smi = _opsin(tail.strip(), output_format="SMILES")
            if not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m and sum(a.GetFormalCharge() for a in m.GetAtoms()) < 0:
                return _canon(f"{cat}.{smi}")
    if an and not cat and _HAS_OPSIN:
        for head in re.findall(r"\b[a-z0-9,()\-\s]{3,60}(?:ium)\b", phrase.lower()):
            smi = _opsin(head.strip(), output_format="SMILES")
            if not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m and sum(a.GetFormalCharge() for a in m.GetAtoms()) > 0:
                return _canon(f"{smi}.{an}")
    return None


if __name__ == "__main__":
    # Self-test
    tests = [
        ("1-butyl-3-methylimidazolium tetrafluoroborate",
         "F[B-](F)(F)F.CCCC[n+]1ccn(C)c1"),
        ("1-ethyl-3-methylimidazolium bis(trifluoromethylsulfonyl)imide",
         "CC[n+]1ccn(C)c1.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F"),
        ("1-butyl-1-methylpyrrolidinium triflate",
         "CCCC[N+]1(C)CCCC1.O=S(=O)([O-])C(F)(F)F"),
        ("tetrabutylammonium bromide",
         "CCCC[N+](CCCC)(CCCC)CCCC.[Br-]"),
        ("cholinium acetate",
         "C[N+](C)(C)CCO.CC(=O)[O-]"),
        ("trihexyl(tetradecyl)phosphonium bis(trifluoromethylsulfonyl)imide",
         "CCCCCC[P+](CCCCCC)(CCCCCC)CCCCCCCCCCCCCC.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F"),
        ("trihexyltetradecylphosphonium dicyanamide",
         "CCCCCC[P+](CCCCCC)(CCCCCC)CCCCCCCCCCCCCC.N#C[N-]C#N"),
    ]
    ok = 0
    for name, expected in tests:
        got = resolve_il(name)
        canon_expected = _canon(expected)
        passed = got == canon_expected
        ok += int(passed)
        print(f"  {'OK' if passed else 'XX'}  {name}\n      got:  {got}\n      want: {canon_expected}")
    print(f"\n{ok}/{len(tests)} tests passed.")
