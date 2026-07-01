"""Option Y — Stage 4: normalize extracted tables into long-format rows
matching our training schema and dedup against existing cache.

This stage is semi-automated: it identifies column semantics via header keyword
matching, canonicalizes IL names via a seed dictionary (SMILES), and emits:
  - `high_confidence.csv` — rows the script is confident about (auto-insertable)
  - `review_queue.csv`    — rows that need a human eyeball (partial match, multi-IL
                             headers, unit ambiguity, unrecognized SMILES)
  - `unmapped_ils.csv`    — IL names we could not resolve; the user can extend
                             the seed dictionary and re-run.

Run after 03_extract_tables.py.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import pandas as pd
from rdkit import Chem

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from il_name_resolver import resolve_il as _resolve_systematic

import os as _os
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TABLES_SUB = _os.environ.get("LIT_TABLES_DIR", "tables")
TABLES_DIR = PROJECT_ROOT / "data" / "lit_scrape" / _TABLES_SUB
OUT_DIR = PROJECT_ROOT / "data" / "lit_scrape"
_SUFFIX = "" if _TABLES_SUB == "tables" else f"_{_TABLES_SUB.replace('tables_', '')}"
HIGH_CONF = OUT_DIR / f"high_confidence{_SUFFIX}.csv"
REVIEW = OUT_DIR / f"review_queue{_SUFFIX}.csv"
UNMAPPED = OUT_DIR / f"unmapped_ils{_SUFFIX}.csv"
IL_DICT_JSON = PROJECT_ROOT / "lignos" / "scripts" / "lit_scrape" / "il_name_dict.json"

# Seed dictionary — common IL cation/anion abbreviations → canonical SMILES.
# User can extend this file to cover more names.
DEFAULT_IL_DICT = {
    # cations
    "[EMIM]": "CC[n+]1ccn(C)c1",
    "[BMIM]": "CCCC[n+]1ccn(C)c1",
    "[HMIM]": "CCCCCC[n+]1ccn(C)c1",
    "[OMIM]": "CCCCCCCC[n+]1ccn(C)c1",
    "[DMIM]": "Cn1cc[n+](C)c1",
    "[MMIM]": "Cn1cc[n+](C)c1",
    "[BMPYRR]": "CCCC[N+]1(C)CCCC1",
    "[BMPYR]":  "CCCC[N+]1(C)CCCC1",
    "[C4mim]":  "CCCC[n+]1ccn(C)c1",
    "[C2mim]":  "CC[n+]1ccn(C)c1",
    "[C6mim]":  "CCCCCC[n+]1ccn(C)c1",
    "[C8mim]":  "CCCCCCCC[n+]1ccn(C)c1",
    # anions
    "[Cl]":  "[Cl-]",
    "[Br]":  "[Br-]",
    "[BF4]": "F[B-](F)(F)F",
    "[PF6]": "F[P-](F)(F)(F)(F)F",
    "[NTf2]": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "[Tf2N]": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
    "[OTf]":  "O=S(=O)([O-])C(F)(F)F",
    "[TfO]":  "O=S(=O)([O-])C(F)(F)F",
    "[DCA]":  "N#C[N-]C#N",
    "[MeSO4]": "COS(=O)(=O)[O-]",
    "[OAc]":  "CC(=O)[O-]",
    "[NO3]":  "[O-][N+](=O)[O-]",
    "[SCN]":  "N#C[S-]",
    "[FAP]":  "F[P-](F)(F)(C(F)(F)F)(C(F)(F)F)C(F)(F)F",
}

# Column keyword → target schema mapping. Order matters — most specific first.
COLUMN_MAP = {
    "gamma_inf": r"γ\s*[∞\^]|gamma\s*\^?\s*∞|gamma\s*inf|infinite\s*dilution|\bln\s*γ\s*[∞\^]",
    "gamma1":    r"\bγ\s*1\b|gamma\s*1|γ\s*12",
    "gamma2":    r"\bγ\s*2\b|gamma\s*2|γ\s*21",
    "H_vap":     r"Δ\s*vap\s*H|\bH\s*vap\b|enthalpy\s*of\s*vaporization|Delta\s*vap\s*H",
    # G_E = excess Gibbs energy of MIXING. Explicitly reject "ΔG*E" / "G*E"
    # which denotes excess Gibbs of ACTIVATION of viscous flow (Eyring) — that
    # is a viscosity-model parameter, NOT the thermodynamic target.
    "G_E":       r"\bG\s*\^\s*E\b|\bG\s*E\s*/|\bexcess\s*molar\s*Gibbs|excess\s*Gibbs(?!\s+energy\s+of\s+activation)",
    # Same caveat for H_E (mixing) vs H*E (activation)
    "H_E":       r"\bH\s*\^\s*E\b|\bH\s*E\s*/|excess\s*molar\s*enthalpy|excess\s*enthalpy(?!\s+of\s+activation)",
    "G_mix":     r"G\s*mix|Gibbs\s*.*\s*mixing|ΔG\s*mix",
    "P":         r"vapor\s*pressure|\bP\b\s*/\s*(k?Pa|bar|mbar|atm)|pressure\s*/\s*(k?Pa|bar)",
    "T":         r"\bT\s*/\s*K\b|\bT\s*\(\s*K\s*\)|\btemperature\b",
    "x1":        r"\bx\s*1\b|mole\s*fraction.*\s*1|\bx_1\b",
    "x2":        r"\bx\s*2\b|mole\s*fraction.*\s*2|\bx_2\b",
    "activity":  r"\bactivity\b|\ba_1\b|\ba_2\b",
}

# Explicit negative patterns — headers that look like a target but denote a
# different physical quantity. If any column header matches one of these,
# REJECT the whole table (prevents subtle mis-extraction).
NEGATIVE_PATTERNS = [
    r"ΔG\s*\*\s*E",       # excess Gibbs of activation of viscous flow
    r"ΔH\s*\*\s*E",
    r"\bG\s*\*\s*E\b",
    r"\bH\s*\*\s*E\b",
    r"δg\s*\*\s*e",
    r"δh\s*\*\s*e",
    r"activation\s+energy",
    r"Grunberg[-\s]Nissan",  # viscosity mixing rule, not thermodynamics
    r"Kendall[-\s]Monroe",
    r"Frenkel(?:\s+correlation)?",
    r"Hind(?:\s+correlation)?",
]
NEGATIVE_RE = re.compile("|".join(NEGATIVE_PATTERNS), re.IGNORECASE)

NUM_RE_BODY = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*$")


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


def load_il_dict():
    if IL_DICT_JSON.exists():
        return json.loads(IL_DICT_JSON.read_text())
    IL_DICT_JSON.write_text(json.dumps(DEFAULT_IL_DICT, indent=2))
    return DEFAULT_IL_DICT


def _lookup_il(token: str, il_dict: dict):
    """Case-insensitive + whitespace-normalized dict lookup for [TOKEN]."""
    if token in il_dict:
        return il_dict[token]
    # whitespace-normalized and case-insensitive
    norm = re.sub(r"\s+", "", token).upper()
    for k, v in il_dict.items():
        if re.sub(r"\s+", "", k).upper() == norm:
            return v
    # common subscripts: Tf2N/NTf2/TfN/NTf all mean bistriflimide
    if norm in ("[TFN]", "[NTF]", "[TF2N]", "[NTF2]", "[TFSI]"):
        return il_dict.get("[NTf2]")
    # imidazolium with alkyl count inside: e.g. "[C mim]" = C1mim = MMIM? Skip.
    return None


def _is_citation_token(token: str) -> bool:
    """Filter out citation brackets like [11,15,17], [9–20], [1]."""
    inner = token.strip("[]").replace(" ", "").replace("\n", "")
    if not inner:
        return True
    # All digits/separators/dashes/en-dashes → citation
    return bool(re.fullmatch(r"[\d,\-–]+", inner))


def resolve_il_name(name: str, il_dict: dict):
    """Scan a string for IL cation/anion abbrevs. Returns canonical SMILES or None."""
    cation = anion = None
    hits = re.findall(r"\[[^\]]+\]", name)
    for h in hits:
        if _is_citation_token(h):
            continue
        smi = _lookup_il(h, il_dict)
        if smi is None:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        q = sum(a.GetFormalCharge() for a in m.GetAtoms())
        if q > 0 and cation is None:
            cation = smi
        elif q < 0 and anion is None:
            anion = smi
    if cation and anion:
        return canon(f"{cation}.{anion}")
    return None


def match_columns(header_row):
    """Return dict col_index → target_key."""
    out = {}
    for i, cell in enumerate(header_row):
        if not cell:
            continue
        s = str(cell).lower()
        for target, pat in COLUMN_MAP.items():
            if re.search(pat, s, re.IGNORECASE):
                out[i] = target
                break
    return out


def _count_numeric(row):
    n = 0
    for v in row:
        try:
            float(str(v or "").strip())
            n += 1
        except Exception:
            pass
    return n


def build_header(raw, max_header_rows: int = 4):
    """Find the best header row range and return (col_map, header_row_count).

    Camelot splits multi-line headers across N rows and leaves blank cells in
    row 0 where a unit line falls to row 1. Strategy:
      1. Find the first row with ≥50% numeric cells — that's the first data row.
         Everything above is header.
      2. Merge header rows column-wise and run match_columns.
      3. Retry with 1..N rows merged and pick the merge that yields the most
         columns, ties broken by earliest split (fewer header rows)."""
    n = len(raw)
    if n == 0:
        return {}, 0

    # find first data row (≥50% numeric cells)
    first_data = None
    for r in range(min(max_header_rows + 1, n)):
        row = list(raw.iloc[r])
        if raw.shape[1] > 0 and _count_numeric(row) / raw.shape[1] >= 0.5:
            first_data = r
            break
    if first_data is None:
        first_data = min(max_header_rows, n - 1)
    first_data = max(1, first_data)  # at least 1 header row

    best = ({}, 1)
    for hrows in range(1, min(first_data, max_header_rows) + 1):
        merged = [
            " ".join(str(raw.iloc[r, c] or "") for r in range(hrows)).strip()
            for c in range(raw.shape[1])
        ]
        cm = match_columns(merged)
        # prefer more matched columns; ties → fewer header rows
        if len(cm) > len(best[0]) or (len(cm) == len(best[0]) and hrows < best[1]):
            best = (cm, hrows)
    return best[0], best[1]


# Fix 1 — caption-aware T extraction.
# When the table body has no T column, parse a temperature from the caption/snippet.
_CAPTION_T_PATTERNS = [
    # "at T = 303.15 K", "T = 303.15 K", "T/K = 303.15"
    r"\bT\s*(?:/\s*K)?\s*=\s*([0-9]{2,3}(?:\.[0-9]+)?)\s*K?\b",
    # "at 298.15 K", "at a temperature of 303 K"
    r"\bat\s+(?:a?\s*temperature\s+of\s+)?([0-9]{2,3}(?:\.[0-9]+)?)\s*K\b",
    # "temperature of 298 K"
    r"\btemperature\s+of\s+([0-9]{2,3}(?:\.[0-9]+)?)\s*K\b",
    # "303.15 K" in isolation near start of caption
    r"^[^.]*?\b([0-9]{2,3}\.[0-9]{1,3})\s*K\b",
]


def extract_caption_T(snippet: str):
    """Return a single T (float, K) from snippet, or None.

    Only returns when there is exactly ONE plausible temperature in the caption.
    If the snippet mentions multiple temperatures (e.g., "298 K, 308 K, 318 K"
    or "at different temperatures"), returns None — those tables need a T
    column in the body and should not be blanket-assigned a single T."""
    if not snippet:
        return None
    s = snippet
    # Hard no: "different temperatures" / "various T" / "as a function of T"
    if re.search(r"different\s+temperatures|various\s+temperatures|function\s+of\s+temperature|temperature\s+range", s, re.I):
        return None
    # Collect all plausible Ts (250–700 K)
    cands = set()
    for pat in _CAPTION_T_PATTERNS:
        for m in re.finditer(pat, s, re.I | re.M):
            try:
                t = float(m.group(1))
            except (ValueError, IndexError):
                continue
            if 250.0 <= t <= 700.0:
                cands.add(round(t, 3))
    # Only accept a unique caption-T
    if len(cands) == 1:
        return next(iter(cands))
    return None


def find_il_candidates_in_context(meta_path: Path, il_dict):
    """Look for IL names in the page text snippet captured with the table."""
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text())
    text = (meta.get("page_text_snippet") or "") + " " + " ".join(meta.get("keywords_found") or [])
    ils = []
    for name in il_dict:
        if name in text:
            smi = resolve_il_name(name + ("[Cl]" if "[" not in name[1:] else ""), il_dict)
            # Better: pair name with any other [X] mention on the page
    # Try pairing all [A]/[B] tokens present
    tokens = [t for t in set(re.findall(r"\[[^\]]+\]", text)) if not _is_citation_token(t)]
    cat = an = None
    cat_name = an_name = None
    for t in tokens:
        smi = _lookup_il(t, il_dict)
        if smi is None:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        q = sum(a.GetFormalCharge() for a in m.GetAtoms())
        if q > 0 and cat is None:
            cat, cat_name = smi, t
        elif q < 0 and an is None:
            an, an_name = smi, t
    if cat and an:
        return [{"smiles": canon(f"{cat}.{an}"), "cation": cat_name, "anion": an_name}]
    # Fallback: try systematic-name resolver on the raw text
    sys_smi = _resolve_systematic(text)
    if sys_smi:
        return [{"smiles": sys_smi, "cation": "systematic", "anion": "systematic"}]
    return []


def process_one_table(csv_path: Path, il_dict):
    try:
        raw = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False)
    except Exception as e:
        return [], [], []
    if len(raw) < 2:
        return [], [], []

    # Fix 3 — robust multi-line header detection (tries 1..max_header_rows merges)
    col_map, hdr_rows = build_header(raw, max_header_rows=4)
    body = raw.iloc[hdr_rows:]

    if not col_map:
        return [], [], []  # no recognizable columns

    # Reject tables whose merged header contains any negative pattern
    # (activation-energy columns, viscosity mixing rules, etc.)
    merged_header = " | ".join(
        " ".join(str(raw.iloc[r, c] or "") for r in range(hdr_rows)).strip()
        for c in range(raw.shape[1])
    )
    if NEGATIVE_RE.search(merged_header):
        return [], [], []

    # Fix 1 — caption-aware T extraction when T column absent
    meta_path = csv_path.with_suffix(".meta.json")
    caption_T = None
    if "T" not in col_map.values():
        if meta_path.exists():
            snippet = (json.loads(meta_path.read_text()).get("page_text_snippet") or "")
            caption_T = extract_caption_T(snippet)
        if caption_T is None:
            return [], [], []  # no T — cannot build a usable row

    # Must have at least one target column beyond T (otherwise we're adding
    # T+SMILES-only rows, which carry no training signal).
    TARGET_KEYS = {"gamma1", "gamma2", "gamma_inf", "H_E", "H_vap",
                   "G_E", "G_mix", "P", "activity"}
    if not (set(col_map.values()) & TARGET_KEYS):
        return [], [], []

    # IL context from meta
    il_cands = find_il_candidates_in_context(meta_path, il_dict)

    # Solute column + per-row solute SMILES (from 03_extract solute_col meta)
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    solute_col = meta.get("solute_col")
    solute_values = meta.get("solute_values") or []
    solute_cache: dict[str, str] = {}

    def resolve_solute(name: str):
        if not name:
            return None
        key = name.lower().strip()
        if key in solute_cache:
            return solute_cache[key]
        try:
            from py2opsin import py2opsin as _op
            smi = _op(name, output_format="SMILES")
        except Exception:
            smi = None
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m is not None and sum(a.GetFormalCharge() for a in m.GetAtoms()) == 0:
                can = Chem.MolToSmiles(m)
                solute_cache[key] = can
                return can
        solute_cache[key] = None
        return None

    hi, rev, unm = [], [], []
    doi = csv_path.parent.name  # directory name = doi_safe

    # Plausible physical ranges — reject rows with values that can't be real
    RANGE_CHECKS = {
        "T":         (250.0, 700.0),      # K
        "x1":        (0.0, 1.0),
        "x2":        (0.0, 1.0),
        "gamma1":    (0.01, 100.0),
        "gamma2":    (0.01, 100.0),
        "gamma_inf": (0.01, 1000.0),
        "G_E":       (-50000, 50000),     # J/mol
        "H_E":       (-50000, 50000),
        "G_mix":     (-50000, 50000),
        "H_vap":     (0, 300000),         # J/mol
        "P":         (0, 1e9),             # Pa
        "activity":  (0, 100.0),
    }

    for _, row in body.iterrows():
        rec = {"source_doi": doi,
               "source_csv": str(csv_path.relative_to(PROJECT_ROOT)),
               "T_source": "column" if caption_T is None else "caption"}
        # Solute identity per row (if 03 detected a solute column)
        if solute_col is not None and solute_col < len(row):
            solute_name = str(row.iloc[solute_col] or "").strip()
            if solute_name and not NUM_RE_BODY.match(solute_name):
                rec["solute_name"] = solute_name
                smi = resolve_solute(solute_name)
                if smi:
                    rec["solute_smiles"] = smi
        valid = True
        for col_idx, key in col_map.items():
            val = str(row.iloc[col_idx] or "").strip()
            try:
                num = float(val)
            except ValueError:
                rec[key] = None
                # gamma/H_E/H_vap targets are OK to be None per-row; T is only
                # required if this table uses a T column.
                if key == "T" and caption_T is None:
                    valid = False
                    break
                continue
            # Range check
            range_lo, range_hi = RANGE_CHECKS.get(key, (float("-inf"), float("inf")))
            if not (range_lo <= num <= range_hi):
                valid = False
                break
            rec[key] = num
        if caption_T is not None and "T" not in rec:
            rec["T"] = caption_T
        if not valid:
            continue
        # drop rows where no target has a numeric value
        if not any(rec.get(k) is not None for k in TARGET_KEYS):
            continue
        if il_cands:
            rec["il_smiles"] = il_cands[0]["smiles"]
            hi.append(rec)
        else:
            rev.append(rec)

    # Deduplicate identical rows within a single table (camelot often
    # re-emits the same table body multiple times across stream/lattice flavors
    # — see review_queue_camelot.csv with 27 duplicates from one source table).
    def _dedup_rows(rows):
        seen = set()
        out = []
        for r in rows:
            key = tuple(round(v, 4) if isinstance(v, float) else v
                        for k, v in sorted(r.items()) if k not in ("source_csv",))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out
    hi = _dedup_rows(hi)
    rev = _dedup_rows(rev)

    if not il_cands:
        # also record the unmapped names for dictionary extension
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            text = meta.get("page_text_snippet") or ""
            for t in set(re.findall(r"\[[^\]]+\]", text)):
                if _is_citation_token(t):
                    continue
                if _lookup_il(t, il_dict) is None:
                    unm.append({"doi": doi, "unmapped_token": t, "snippet": text[:200]})

    return hi, rev, unm


def main():
    il_dict = load_il_dict()
    print(f"IL dictionary: {len(il_dict)} entries → {IL_DICT_JSON}")

    all_hi, all_rev, all_unm = [], [], []
    n_tables = 0
    for csv_path in sorted(TABLES_DIR.rglob("*.csv")):
        if ".meta." in csv_path.name:
            continue
        n_tables += 1
        hi, rev, unm = process_one_table(csv_path, il_dict)
        all_hi.extend(hi); all_rev.extend(rev); all_unm.extend(unm)

    def write_csv(path, rows):
        if not rows:
            path.write_text("")
            return
        cols = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})

    write_csv(HIGH_CONF, all_hi)
    write_csv(REVIEW, all_rev)
    # Deduplicate unmapped
    seen = {}
    for u in all_unm:
        seen.setdefault(u["unmapped_token"], u)
    write_csv(UNMAPPED, list(seen.values()))

    print(f"Tables processed: {n_tables}")
    print(f"  High-confidence rows: {len(all_hi)} → {HIGH_CONF}")
    print(f"  Review queue rows:    {len(all_rev)} → {REVIEW}")
    print(f"  Unmapped IL tokens:   {len(seen)} → {UNMAPPED}")
    if seen:
        print("Top unmapped tokens (extend il_name_dict.json):")
        for t in list(seen.keys())[:10]:
            print(f"  {t}")


if __name__ == "__main__":
    main()
