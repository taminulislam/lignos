"""Option Y — Stage 3: extract candidate data tables from fetched PDFs.

Uses pdfplumber (pure-Python, no Ghostscript/Java). Scans each PDF page for
tables; for each table, captures a snippet of surrounding text (caption) for
downstream parsing.

Heuristic filter for "looks like thermodynamic data":
  - Has a header row containing at least one of: T, K, x, gamma, G^E, G_E,
    H^E, HE, P, kPa, Pa, activity, coefficient, mol, fraction, mole
  - Has at least 3 rows and 2 numeric columns.

Outputs:
    data/lit_scrape/tables/<doi_safe>/table_<page>_<idx>.csv — raw table
    data/lit_scrape/tables/<doi_safe>/table_<page>_<idx>.meta.json — caption + page info
    data/lit_scrape/tables_index.csv — master index (doi, page, n_rows, n_cols, keywords_found, snippet)

Run after 02_fetch_pdfs.py.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "lit_scrape" / "pdfs"
OUT_DIR = PROJECT_ROOT / "data" / "lit_scrape" / "tables"
INDEX_CSV = PROJECT_ROOT / "data" / "lit_scrape" / "tables_index.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Case-insensitive keyword set indicating "thermodynamic-relevant" table.
KEYWORDS = [
    r"\bT\s*/\s*K\b", r"\bT\s*\(\s*K\s*\)", r"temperature",
    r"\bx[12]?\b", r"mole\s*fraction", r"mass\s*fraction",
    r"gamma", r"γ", r"activity\s*coefficient",
    r"G\s*[\^\*]\s*E", r"excess\s*(Gibbs|enthalpy|entropy)",
    r"H\s*[\^\*]\s*E", r"H\s*vap", r"enthalpy\s*of\s*vaporization",
    r"G\s*mix", r"Gibbs\s*energy\s*of\s*mixing",
    r"vapor\s*pressure", r"\bP\b\s*/\s*kPa", r"\bP\b\s*/\s*Pa",
    r"ionic\s*liquid", r"\[[A-Za-z0-9]+\]",  # [EMIM] etc.
]
KW_RE = re.compile("|".join(KEYWORDS), re.IGNORECASE)

NUM_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*$")


def is_numeric_cell(s):
    return bool(NUM_RE.match(str(s or "")))


# --- Solute-name detection ---------------------------------------------------
# Activity-coefficient and γ∞ papers typically have a row per solute. The
# leftmost column in the body carries the solute name (e.g., "hexane",
# "benzene"). Capture that column header + body as solute candidates so 04
# can pair (IL, solute, T) → γ∞ and activity-coef rows become trainable.

# Common solute-column headers seen in J.Chem.Eng.Data / Fluid Phase Equilib.
_SOLUTE_HEADER_PATTERNS = [
    r"\bsolute\b", r"\bsolvent\b", r"\bcompound\b", r"\bsubstance\b",
    r"\borganic\b", r"\bchemical\b", r"\bname\b",
    r"\bhydrocarbon\b", r"\balcohol\b", r"\bamine\b",
]
_SOLUTE_HEADER_RE = re.compile("|".join(_SOLUTE_HEADER_PATTERNS), re.IGNORECASE)


def find_solute_column(tbl, header_rows: int = 2) -> int | None:
    """Return the index of the column that looks like a solute-name list.

    A column qualifies as the solute column when ALL of:
      - its header cell (merged across header_rows rows) matches a known
        solute-type keyword OR the column is the leftmost non-T column, AND
      - ≥60% of body cells are non-numeric strings of length 2..40, AND
      - the column has at least 3 distinct body values.
    Returns None if nothing qualifies.
    """
    if not tbl or len(tbl) < header_rows + 3:
        return None
    ncols = max(len(r) for r in tbl)
    body = tbl[header_rows:]
    # Build per-column stats
    cand = None
    for c in range(ncols):
        header_cell = " ".join(str((tbl[r][c] if c < len(tbl[r]) else "") or "")
                               for r in range(header_rows)).strip().lower()
        body_vals = [str((row[c] if c < len(row) else "") or "").strip()
                     for row in body]
        body_vals = [v for v in body_vals if v]
        if len(body_vals) < 3:
            continue
        non_num = [v for v in body_vals if not is_numeric_cell(v)
                   and 2 <= len(v) <= 40]
        frac_non_num = len(non_num) / max(len(body_vals), 1)
        distinct = len(set(non_num))
        header_match = bool(_SOLUTE_HEADER_RE.search(header_cell))
        # A column is a strong solute candidate if header matches, or if it's
        # the leftmost column and the body is overwhelmingly non-numeric.
        is_leftmost_textual = (c == 0 and frac_non_num >= 0.8 and distinct >= 3)
        if header_match and frac_non_num >= 0.6 and distinct >= 3:
            return c
        if is_leftmost_textual and cand is None:
            cand = c
    return cand


def process_pdf(pdf_path: Path, out_pdf_dir: Path):
    rows_out = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    continue
                for tbl_idx, tbl in enumerate(tables):
                    if not tbl or len(tbl) < 3:
                        continue
                    flat = " ".join(" ".join(str(c or "") for c in row) for row in tbl[:2])
                    hits = KW_RE.findall(flat)
                    # Count numeric cells
                    total_cells = sum(len(row) for row in tbl)
                    numeric_cells = sum(is_numeric_cell(c) for row in tbl for c in row)
                    if not hits and numeric_cells < 0.3 * total_cells:
                        continue
                    if len(tbl[0]) < 2:
                        continue
                    # Save
                    csv_path = out_pdf_dir / f"table_p{page_idx:03d}_i{tbl_idx}.csv"
                    with open(csv_path, "w", newline="") as f:
                        w = csv.writer(f)
                        for row in tbl:
                            w.writerow([c or "" for c in row])
                    # Detect solute column (leftmost non-numeric body column)
                    solute_col = find_solute_column(tbl, header_rows=2)
                    solute_values = []
                    if solute_col is not None:
                        for row in tbl[2:]:
                            if solute_col < len(row):
                                v = str(row[solute_col] or "").strip()
                                if v and not is_numeric_cell(v):
                                    solute_values.append(v)
                    # Save meta
                    meta = {
                        "pdf": pdf_path.name,
                        "page": page_idx + 1,
                        "table_idx": tbl_idx,
                        "n_rows": len(tbl),
                        "n_cols": len(tbl[0]) if tbl else 0,
                        "numeric_frac": round(numeric_cells / max(total_cells, 1), 3),
                        "keywords_found": list(set(hits)),
                        "solute_col": solute_col,
                        "solute_values": solute_values,
                        "page_text_snippet": (text[:800] if text else ""),
                    }
                    (csv_path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
                    rows_out.append({
                        "pdf": pdf_path.name,
                        "page": page_idx + 1,
                        "table_idx": tbl_idx,
                        "n_rows": meta["n_rows"],
                        "n_cols": meta["n_cols"],
                        "numeric_frac": meta["numeric_frac"],
                        "keywords": ";".join(meta["keywords_found"]),
                        "csv_path": str(csv_path.relative_to(PROJECT_ROOT)),
                    })
    except Exception as e:
        print(f"  {pdf_path.name}: ERROR {e}")
    return rows_out


def main(args):
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.max:
        pdfs = pdfs[: args.max]
    print(f"Processing {len(pdfs)} PDFs from {PDF_DIR}")

    all_rows = []
    for i, pdf_path in enumerate(pdfs):
        doi_safe = pdf_path.stem
        out_pdf_dir = OUT_DIR / doi_safe
        out_pdf_dir.mkdir(parents=True, exist_ok=True)
        rows = process_pdf(pdf_path, out_pdf_dir)
        all_rows.extend(rows)
        if (i + 1) % 10 == 0 or i == len(pdfs) - 1:
            print(f"  [{i+1}/{len(pdfs)}] {pdf_path.name}: {len(rows)} candidate tables")

    # Master index
    if all_rows:
        cols = list(all_rows[0].keys())
        with open(INDEX_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(all_rows)
    print(f"\n{len(all_rows)} candidate tables across {len(pdfs)} PDFs.")
    print(f"Index: {INDEX_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()
    main(args)
