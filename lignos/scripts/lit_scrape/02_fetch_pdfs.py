"""Option Y — Stage 2: fetch open-access PDFs for candidate papers via Unpaywall.

Unpaywall API (free): https://api.unpaywall.org/v2/{doi}?email={email}
Returns best OA URL if available. We only attempt download when the paper is
flagged is_oa=True in the discovery CSV.

Usage:
    python 02_fetch_pdfs.py [--max 50] [--input candidate_papers.csv]

Output:
    data/lit_scrape/pdfs/<doi_sanitized>.pdf — downloaded files
    data/lit_scrape/fetch_log.csv — per-row outcome (ok/skipped/failed + reason)

Network-heavy; CPU-only; polite throttle. Safe to re-run (skips already fetched).
"""
from __future__ import annotations
import argparse, csv, os, re, time
from pathlib import Path
from urllib.parse import quote

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_IN_CSV = PROJECT_ROOT / "data" / "lit_scrape" / "candidate_papers.csv"
PDF_DIR = PROJECT_ROOT / "data" / "lit_scrape" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

EMAIL = os.environ.get("UNPAYWALL_EMAIL", "your-email@example.com")
TIMEOUT = 30


def sanitize_doi(doi: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", doi)


def fetch_unpaywall(doi: str):
    """Call Unpaywall. Returns dict {is_oa, oa_url, oa_status, license}."""
    url = f"https://api.unpaywall.org/v2/{quote(doi)}?email={quote(EMAIL)}"
    r = requests.get(url, timeout=TIMEOUT)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}"}
    data = r.json()
    best = data.get("best_oa_location") or {}
    return {
        "is_oa": data.get("is_oa", False),
        "oa_url": best.get("url_for_pdf") or best.get("url") or "",
        "oa_status": data.get("oa_status", ""),
        "license": best.get("license", ""),
        "all_oa_urls": [
            loc.get("url_for_pdf") or loc.get("url")
            for loc in (data.get("oa_locations") or [])
            if loc.get("url_for_pdf") or loc.get("url")
        ],
    }


def fetch_semantic_scholar(doi: str):
    """Semantic Scholar fallback. Returns openAccessPdf URL if available."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi)}?fields=openAccessPdf"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        oa = data.get("openAccessPdf") or {}
        return oa.get("url")
    except Exception:
        return None


def download_pdf(url: str, out_path: Path) -> str:
    """Return 'ok' or error reason."""
    try:
        r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    except Exception as e:
        return f"request_failed: {e}"
    if r.status_code != 200:
        return f"http_{r.status_code}"
    ct = r.headers.get("content-type", "").lower()
    if "pdf" not in ct and not r.content[:4].startswith(b"%PDF"):
        return f"not_pdf (ct={ct})"
    out_path.write_bytes(r.content)
    return "ok"


def main(args):
    in_csv = Path(args.input) if args.input else DEFAULT_IN_CSV
    if not in_csv.is_absolute():
        in_csv = PROJECT_ROOT / in_csv
    log_csv = Path(args.log) if args.log else in_csv.parent / (
        f"fetch_log_{in_csv.stem}.csv" if in_csv.stem != "candidate_papers" else "fetch_log.csv"
    )
    if not log_csv.is_absolute():
        log_csv = PROJECT_ROOT / log_csv
    rows = list(csv.DictReader(open(in_csv)))
    print(f"Input: {in_csv}")
    print(f"Log:   {log_csv}")
    print(f"Loaded {len(rows)} candidates; {sum(1 for r in rows if r.get('is_oa')=='True')} OA-flagged.")
    log = []
    n_ok = n_skip = n_fail = 0
    for i, row in enumerate(rows[: args.max]):
        doi = row["doi"]
        if not doi:
            continue
        safe = sanitize_doi(doi)
        pdf_path = PDF_DIR / f"{safe}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 1024:
            log.append({**row, "fetch_result": "already_present"})
            n_skip += 1
            continue

        # Cross-check via Unpaywall even if discovery said is_oa=True
        try:
            up = fetch_unpaywall(doi)
        except Exception as e:
            log.append({**row, "fetch_result": f"unpaywall_error: {e}"})
            n_fail += 1
            time.sleep(0.3)
            continue
        if "error" in up:
            log.append({**row, "fetch_result": up["error"]})
            n_fail += 1
            time.sleep(0.3)
            continue
        # Build ordered list of candidate URLs: Unpaywall best, then alternates,
        # then Semantic Scholar fallback.
        candidate_urls = []
        if up["oa_url"]:
            candidate_urls.append(up["oa_url"])
        for u in up.get("all_oa_urls", []):
            if u and u not in candidate_urls:
                candidate_urls.append(u)
        ss_url = fetch_semantic_scholar(doi)
        if ss_url and ss_url not in candidate_urls:
            candidate_urls.append(ss_url)

        if not candidate_urls:
            log.append({**row, "fetch_result": "not_open_access"})
            n_skip += 1
            time.sleep(0.2)
            continue

        result = None
        for url in candidate_urls:
            result = download_pdf(url, pdf_path)
            if result == "ok":
                break
        log.append({**row, "fetch_result": result or "all_urls_failed",
                    "actual_oa_url": candidate_urls[0],
                    "n_urls_tried": len(candidate_urls)})
        if result == "ok":
            n_ok += 1
        else:
            n_fail += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{min(args.max, len(rows))}] ok={n_ok} skip={n_skip} fail={n_fail}")
        time.sleep(0.25)  # polite

    # Write log
    if log:
        cols = sorted({k for r in log for k in r.keys()})
        with open(log_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in log:
                w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nDone. ok={n_ok} skipped={n_skip} failed={n_fail}")
    print(f"PDFs: {PDF_DIR}  Log: {log_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--input", type=str, default=None,
                    help="Input candidate CSV (default: data/lit_scrape/candidate_papers.csv). "
                         "Relative paths are resolved from the project root.")
    ap.add_argument("--log", type=str, default=None,
                    help="Fetch-log output path (default: fetch_log_<input-stem>.csv beside the input).")
    args = ap.parse_args()
    main(args)
