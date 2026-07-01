"""Option Y — Stage 1: paper discovery via OpenAlex.

Query OpenAlex for ionic-liquid papers that measure the four pinned properties
we need (gamma_2, G^E, G_mix, H_vap). Output a ranked CSV of candidate papers
with DOI, open-access status, and an "IL coverage" estimate from the abstract.

Usage:
    python 01_discover_papers.py [--max 500] [--from-year 2005]
    # writes: data/lit_scrape/candidate_papers.csv

This is a FREE API — OpenAlex requires no key (but benefits from email).
"""
from __future__ import annotations
import argparse, csv, os, re, time
from pathlib import Path
from urllib.parse import quote

import pyalex
from pyalex import Works

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_CSV = PROJECT_ROOT / "data" / "lit_scrape" / "candidate_papers.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Polite: identify ourselves (any email, no verification)
pyalex.config.email = os.environ.get("OPENALEX_EMAIL", "your-email@example.com")

# Search targets — each tuple = (property label, OpenAlex search string).
# The search terms are phrased to maximize precision: "ionic liquid" + property
# + (activity coefficient | excess enthalpy | ...). We'll rank by estimated
# IL coverage in the abstract later.
SEARCHES = [
    ("gamma2",  '"ionic liquid" AND "activity coefficient" AND (infinite OR dilution OR finite)'),
    ("G_E",     '"ionic liquid" AND ("excess Gibbs" OR "G^E" OR "Gibbs energy of mixing")'),
    ("G_mix",   '"ionic liquid" AND ("Gibbs energy of mixing" OR "mixing enthalpy")'),
    ("H_vap",   '"ionic liquid" AND ("vaporization enthalpy" OR "enthalpy of vaporization" OR "vapor pressure")'),
]

# Rough IL-name vocabulary for abstract-based coverage estimate
IL_PATTERNS = [
    r"\[EMIM\]", r"\[BMIM\]", r"\[HMIM\]", r"\[OMIM\]", r"\[DMIM\]",
    r"\[MMIM\]", r"\[EMMIM\]", r"\[BMPIP\]", r"\[BMPYR\]", r"\[BMPY\]",
    r"\[AMIM\]", r"\[CnMIM\]", r"\[BMPYRR\]", r"\[N[0-9,]+\]",
    r"imidazolium", r"pyridinium", r"pyrrolidinium", r"ammonium", r"phosphonium",
    r"\[NTf2\]", r"\[BF4\]", r"\[PF6\]", r"\[DCA\]", r"\[OTf\]",
    r"\[TfO\]", r"\[NO3\]", r"\[OAc\]", r"\[MeSO4\]", r"\[Cl\]",
]
IL_RE = re.compile("|".join(IL_PATTERNS), re.IGNORECASE)


def estimate_il_coverage(abstract_inverted_index):
    """OpenAlex returns abstracts as inverted indices. Reconstruct and count
    IL-name mentions as a proxy for how many ILs the paper touches."""
    if not abstract_inverted_index:
        return 0
    words = {}
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            words[pos] = word
    abstract = " ".join(words[i] for i in sorted(words.keys()))
    return len(IL_RE.findall(abstract))


def run(args):
    all_rows = []
    seen_dois = set()
    for prop_label, query in SEARCHES:
        print(f"[{prop_label}] querying OpenAlex: {query}")
        w = (Works()
             .search(query)
             .filter(from_publication_date=f"{args.from_year}-01-01")
             .filter(type="article")
             .sort(relevance_score="desc"))
        got = 0
        for page in w.paginate(per_page=50, n_max=args.max_per_prop):
          for paper in page:
            doi = (paper.get("doi") or "").replace("https://doi.org/", "")
            if not doi or doi in seen_dois:
                continue
            seen_dois.add(doi)
            ai = paper.get("abstract_inverted_index") or {}
            il_hits = estimate_il_coverage(ai)
            if il_hits == 0:
                continue  # not actually about ILs
            oa = paper.get("open_access") or {}
            all_rows.append({
                "doi": doi,
                "year": paper.get("publication_year"),
                "title": (paper.get("title") or "").replace("\n", " ").strip(),
                "venue": ((paper.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
                "is_oa": bool(oa.get("is_oa")),
                "oa_url": oa.get("oa_url") or "",
                "cited_by": paper.get("cited_by_count", 0),
                "relevance_score": paper.get("relevance_score", 0),
                "il_mentions_est": il_hits,
                "target_prop": prop_label,
            })
            got += 1
        print(f"  kept {got} candidates (total unique so far: {len(all_rows)})")
        time.sleep(0.2)  # polite throttle

    # Rank: open-access first, then more IL mentions, then higher citation
    all_rows.sort(key=lambda r: (-int(r["is_oa"]), -r["il_mentions_est"], -r["cited_by"]))
    with open(OUT_CSV, "w", newline="") as f:
        cols = ["doi", "year", "title", "venue", "is_oa", "oa_url",
                "cited_by", "relevance_score", "il_mentions_est", "target_prop"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    n_oa = sum(1 for r in all_rows if r["is_oa"])
    print(f"\nWrote {OUT_CSV} — {len(all_rows)} unique candidates, {n_oa} open-access.")
    print(f"Top 10 by rank:")
    for r in all_rows[:10]:
        tag = "OA" if r["is_oa"] else "--"
        print(f"  [{tag}] {r['year']} {r['target_prop']:7s} IL≈{r['il_mentions_est']:>2d}  {r['title'][:80]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2005)
    ap.add_argument("--max-per-prop", type=int, default=500)
    args = ap.parse_args()
    run(args)
