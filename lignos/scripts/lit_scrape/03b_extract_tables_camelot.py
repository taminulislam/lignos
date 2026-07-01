"""Option Y — Stage 3b: alternative table extraction using camelot-py.

camelot handles merged cells + multi-line headers better than pdfplumber.
Needs Ghostscript on PATH (install via `conda install -c conda-forge ghostscript`).

Writes the same output format as 03_extract_tables.py so stage 4 (normalize)
works unchanged. Output directory is `tables_camelot/` to keep the two
extractors' results side by side for comparison.

Usage:
    python 03b_extract_tables_camelot.py [--max N] [--flavor lattice|stream|both]
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path

# Ghostscript path must be in PATH before importing camelot
os.environ["PATH"] = "/u/kahmed2/miniconda3/bin:" + os.environ.get("PATH", "")

import pdfplumber  # still used for page text snippets
import camelot

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "lit_scrape" / "pdfs"
OUT_DIR = PROJECT_ROOT / "data" / "lit_scrape" / "tables_camelot"
INDEX_CSV = PROJECT_ROOT / "data" / "lit_scrape" / "tables_camelot_index.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import re

# Reuse the solute-column detector from 03_extract_tables.py (sibling module).
import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "_lit_03", Path(__file__).resolve().parent / "03_extract_tables.py")
_lit03 = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_lit03)
find_solute_column = _lit03.find_solute_column

KEYWORDS = [
    r"\bT\s*/\s*K\b", r"\bT\s*\(\s*K\s*\)", r"temperature",
    r"\bx[12]?\b", r"mole\s*fraction", r"mass\s*fraction",
    r"gamma", r"γ", r"activity\s*coefficient",
    r"G\s*[\^\*]\s*E", r"excess\s*(Gibbs|enthalpy|entropy)",
    r"H\s*[\^\*]\s*E", r"H\s*vap", r"enthalpy\s*of\s*vaporization",
    r"G\s*mix", r"Gibbs\s*energy\s*of\s*mixing",
    r"vapor\s*pressure", r"\bP\b\s*/\s*kPa", r"\bP\b\s*/\s*Pa",
    r"ionic\s*liquid", r"\[[A-Za-z0-9]+\]",
]
KW_RE = re.compile("|".join(KEYWORDS), re.IGNORECASE)
NUM_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*$")


def is_numeric_cell(s):
    return bool(NUM_RE.match(str(s or "")))


def extract_page_text(pdf_path, page_num):
    """Get page text via pdfplumber for context snippet."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            return pdf.pages[page_num - 1].extract_text() or ""
    except Exception:
        return ""


def process_pdf(pdf_path: Path, out_pdf_dir: Path, flavor: str):
    """Extract via camelot with given flavor(s). Returns list of table dicts."""
    flavors = ["lattice", "stream"] if flavor == "both" else [flavor]
    rows_out = []
    for fl in flavors:
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor=fl, suppress_stdout=True)
        except Exception as e:
            print(f"    {pdf_path.name} flavor={fl}: ERROR {e}")
            continue
        for tbl_idx, t in enumerate(tables):
            df = t.df
            if len(df) < 3 or df.shape[1] < 2:
                continue
            flat = " ".join(" ".join(str(c or "") for c in df.iloc[:2, :].values.flatten()))
            hits = KW_RE.findall(flat)
            total = df.shape[0] * df.shape[1]
            num_cells = sum(1 for c in df.values.flatten() if is_numeric_cell(c))
            if not hits and num_cells < 0.3 * total:
                continue
            page_num = int(t.page)
            csv_path = out_pdf_dir / f"{fl}_p{page_num:03d}_i{tbl_idx}.csv"
            df.to_csv(csv_path, index=False, header=False)
            # Solute column + body values
            tbl_list = df.values.tolist()
            solute_col = find_solute_column(tbl_list, header_rows=2)
            solute_values = []
            if solute_col is not None:
                for row in tbl_list[2:]:
                    if solute_col < len(row):
                        v = str(row[solute_col] or "").strip()
                        if v and not is_numeric_cell(v):
                            solute_values.append(v)
            meta = {
                "pdf": pdf_path.name,
                "flavor": fl,
                "page": page_num,
                "table_idx": tbl_idx,
                "n_rows": len(df),
                "n_cols": df.shape[1],
                "numeric_frac": round(num_cells / max(total, 1), 3),
                "keywords_found": list(set(hits)),
                "parsing_accuracy": float(t.parsing_report.get("accuracy", 0)),
                "whitespace": float(t.parsing_report.get("whitespace", 0)),
                "solute_col": solute_col,
                "solute_values": solute_values,
                "page_text_snippet": extract_page_text(pdf_path, page_num)[:800],
            }
            (csv_path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
            rows_out.append({
                "pdf": pdf_path.name,
                "flavor": fl,
                "page": page_num,
                "table_idx": tbl_idx,
                "n_rows": meta["n_rows"],
                "n_cols": meta["n_cols"],
                "numeric_frac": meta["numeric_frac"],
                "accuracy": meta["parsing_accuracy"],
                "keywords": ";".join(meta["keywords_found"]),
                "csv_path": str(csv_path.relative_to(PROJECT_ROOT)),
            })
    return rows_out


def main(args):
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.max:
        pdfs = pdfs[: args.max]
    print(f"camelot extraction on {len(pdfs)} PDFs, flavor={args.flavor}")
    all_rows = []
    for i, pdf_path in enumerate(pdfs):
        doi_safe = pdf_path.stem
        out_pdf_dir = OUT_DIR / doi_safe
        out_pdf_dir.mkdir(parents=True, exist_ok=True)
        rows = process_pdf(pdf_path, out_pdf_dir, args.flavor)
        all_rows.extend(rows)
        print(f"  [{i+1}/{len(pdfs)}] {pdf_path.name}: {len(rows)} candidate tables")

    if all_rows:
        cols = list(all_rows[0].keys())
        with open(INDEX_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(all_rows)
    print(f"\n{len(all_rows)} candidate tables across {len(pdfs)} PDFs.")
    print(f"Index: {INDEX_CSV}")
    print(f"Tables root: {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--flavor", choices=["lattice", "stream", "both"], default="both")
    args = ap.parse_args()
    main(args)
