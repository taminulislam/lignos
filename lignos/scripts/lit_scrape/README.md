# Option Y — literature scrape pipeline

Four-stage automation to pull thermodynamic data for ionic liquids from the
open literature, targeted at the four pinned properties holding core7 R² below
0.90: `gamma2`, `G^E`, `G_mix`, `H_vap`.

## Stages

| Stage | Script | Input | Output |
|---|---|---|---|
| 1. Discover | `01_discover_papers.py` | OpenAlex API | `data/lit_scrape/candidate_papers.csv` (ranked) |
| 2. Fetch    | `02_fetch_pdfs.py`      | Unpaywall API | `data/lit_scrape/pdfs/*.pdf` + `fetch_log.csv` |
| 3. Extract  | `03_extract_tables.py`  | PDFs (pdfplumber) | `data/lit_scrape/tables/<doi>/*.csv` + `tables_index.csv` |
| 4. Normalize| `04_normalize.py`       | tables + `il_name_dict.json` | `high_confidence.csv`, `review_queue.csv`, `unmapped_ils.csv` |

End-to-end: `bash run_all.sh`. All stages are idempotent.

## Required tuning by the user

The most important artifact the user needs to edit is **`il_name_dict.json`**
in this directory. It maps IL cation/anion abbreviations (e.g. `[EMIM]`,
`[BMPYRR]`, `[NTf2]`) to their canonical SMILES. After each run, inspect
`unmapped_ils.csv` — add any token that appears often enough to be worth
resolving, then re-run stage 4.

Seed dictionary covers: EMIM, BMIM, HMIM, OMIM, DMIM, MMIM, BMPYRR, BMPYR,
C4mim, C2mim, C6mim, C8mim, Cl, Br, BF4, PF6, NTf2, Tf2N, OTf, TfO, DCA,
MeSO4, OAc, NO3, SCN, FAP. Every additional token you add expands coverage.

## Current state of the pipeline

- Discovery is reliable (OpenAlex returns ranked candidates with `is_oa` flag).
- Fetch success rate ≈ 40% because many "OA" papers fail when Unpaywall's URL
  is behind a Cloudflare challenge or requires JavaScript. Failures logged.
- Extraction quality is **mediocre** — `pdfplumber` handles flat tables but
  struggles with multi-line headers, merged cells, and figure-embedded tables.
  Expect ~20% of extracted tables to be directly normalizable; the rest go to
  `review_queue.csv` for human review.
- Normalization is **seed-limited** — the dictionary starts with ~26 entries.
  Every re-run of stage 4 (after dictionary extension) converts more rows to
  high-confidence.

## Known gaps / next upgrades

- Add **camelot-py** (needs Ghostscript) for better table extraction quality.
- Add **name-to-SMILES** resolver via OPSIN (systematic IL names like
  "1-butyl-3-methylimidazolium tetrafluoroborate") as fallback when abbrev
  lookup fails.
- Add **ChemDataExtractor** for figure-caption-based property discovery.
- Integrate **DDB open-access tables** (Dortmund Data Bank's free subset) as
  a second-stage data source.

## Running on a cluster

Stages 1–2 are network-bound; run on a login/head node. Stages 3–4 are
CPU-bound and can run on the login node or any SLURM CPU job.
