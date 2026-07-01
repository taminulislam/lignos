#!/bin/bash
# End-to-end driver for the literature-scrape pipeline (Option Y).
# Runs: 01 discover → 02 fetch → 03 extract → 04 normalize
# Idempotent — safe to re-run after extending the IL name dictionary.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$PROJECT_ROOT"

MAX_PER_PROP="${MAX_PER_PROP:-500}"   # candidates per target property (stage 1)
MAX_FETCH="${MAX_FETCH:-200}"         # PDFs to attempt to download (stage 2)

echo "============================================"
echo "Option Y: literature scrape pipeline"
echo "  MAX_PER_PROP=$MAX_PER_PROP  MAX_FETCH=$MAX_FETCH"
echo "============================================"

echo ""; echo "--- Stage 1: discover candidates (OpenAlex) ---"
python lignos/scripts/lit_scrape/01_discover_papers.py \
    --max-per-prop "$MAX_PER_PROP"

echo ""; echo "--- Stage 2: fetch OA PDFs (Unpaywall) ---"
python lignos/scripts/lit_scrape/02_fetch_pdfs.py --max "$MAX_FETCH"

echo ""; echo "--- Stage 3: extract candidate tables (pdfplumber) ---"
python lignos/scripts/lit_scrape/03_extract_tables.py

echo ""; echo "--- Stage 4: normalize + dedup ---"
python lignos/scripts/lit_scrape/04_normalize.py

echo ""; echo "============================================"
echo "Done. Review artifacts:"
echo "  data/lit_scrape/candidate_papers.csv   — all candidates ranked"
echo "  data/lit_scrape/fetch_log.csv          — download outcome per DOI"
echo "  data/lit_scrape/tables_index.csv       — extracted tables"
echo "  data/lit_scrape/high_confidence.csv    — auto-insertable rows"
echo "  data/lit_scrape/review_queue.csv       — rows needing human review"
echo "  data/lit_scrape/unmapped_ils.csv       — IL names to add to il_name_dict.json"
echo "============================================"
