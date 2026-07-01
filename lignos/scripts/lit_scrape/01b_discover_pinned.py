"""Option Y — Stage 1 (narrow variant): targeted discovery for pinned props.

The broad 01_discover_papers.py returns ~350 candidates across 4 props, but the
actual yield of usable tables was lopsided: 9 G^E + 16 γ∞ tables extracted,
ZERO for H_vap / G_mix / γ2. This script re-runs discovery with narrower,
per-property queries that:

  1. Put the property phrase in quotes and combine with ionic-liquid variants.
  2. Prefer journals that publish primary thermo data (J. Chem. Eng. Data,
     J. Chem. Thermodynamics, Fluid Phase Equilib., Thermochim. Acta, Ind.
     Eng. Chem. Res., etc).
  3. Drop review / simulation-only papers (title contains "review", "MD", "DFT",
     "simulation" without "experimental"|"measurement").
  4. Require "experimental" OR "measurement" OR "measured" in the abstract
     to filter out purely theoretical / screening papers.
  5. Emit per-prop CSVs plus a merged one, so the user can fetch each bucket
     independently.

Usage:
    python 01b_discover_pinned.py [--max-per-prop 400] [--from-year 2000]
    # writes: data/lit_scrape/candidate_papers_narrow.csv
    #         data/lit_scrape/candidate_papers_<prop>.csv  (one per prop)
"""
from __future__ import annotations
import argparse, csv, os, re, time
from pathlib import Path

import pyalex
from pyalex import Works

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "lit_scrape"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MERGED = OUT_DIR / "candidate_papers_narrow.csv"

pyalex.config.email = os.environ.get("OPENALEX_EMAIL", "your-email@example.com")

# Narrower search terms — property-specific phrases require exact matches.
# Each entry: (label, query, must_abstract_regex, drop_title_regex)
SEARCHES = [
    ("gamma2",
     '"ionic liquid" AND "activity coefficient" AND ("finite dilution" OR "gamma 2" OR "γ2" OR "mole fraction" OR "binary mixture")',
     r"activity\s+coefficient|γ\s*2|gamma\s*2|excess\s+molar",
     r"review|simulation|molecular\s+dynamics|^md\s"),
    ("gamma_inf",
     '"ionic liquid" AND ("infinite dilution" OR "γ∞" OR "activity coefficient at infinite")',
     r"infinite\s+dilution|γ\s*∞|gamma.*infinite",
     r"review|simulation|molecular\s+dynamics"),
    ("G_E",
     '"ionic liquid" AND ("excess Gibbs" OR "excess molar Gibbs" OR "G^E" OR "G E /J")',
     r"excess\s+Gibbs|excess\s+molar|G\s*\^?\s*E|binary\s+mixture",
     r"^review|simulation"),
    ("G_mix",
     '"ionic liquid" AND ("Gibbs energy of mixing" OR "mixing Gibbs" OR "ΔG mix" OR "enthalpy of mixing")',
     r"Gibbs\s+energy|mixing\s+enthalpy|ΔG\s*mix|ΔH\s*mix",
     r"^review|simulation-only"),
    ("H_vap",
     '"ionic liquid" AND ("vaporization enthalpy" OR "enthalpy of vaporization" OR "ΔvapH" OR "Δ vap H")',
     r"vaporization\s+enthalpy|enthalpy\s+of\s+vaporization|Δ\s*vap\s*H|ΔvapH",
     r"^review|polymer|nanofluid"),
]

# Journals we strongly prefer — these publish primary thermochemical data for ILs.
# OpenAlex uses "source.display_name" fields.
PRIMARY_JOURNALS = [
    "Journal of Chemical & Engineering Data",
    "Journal of Chemical Engineering Data",
    "The Journal of Chemical Thermodynamics",
    "Journal of Chemical Thermodynamics",
    "Fluid Phase Equilibria",
    "Thermochimica Acta",
    "Industrial & Engineering Chemistry Research",
    "Journal of Molecular Liquids",
    "Journal of Solution Chemistry",
    "Physical Chemistry Chemical Physics",
    "Green Chemistry",
]
JOURNAL_SET_LC = {j.lower() for j in PRIMARY_JOURNALS}

# IL vocabulary for coverage estimate (richer than 01_discover_papers)
IL_PATTERNS = [
    r"\[EMIM\]", r"\[BMIM\]", r"\[HMIM\]", r"\[OMIM\]", r"\[DMIM\]", r"\[MMIM\]",
    r"\[EMMIM\]", r"\[BMPIP\]", r"\[BMPYR\]", r"\[BMPY\]", r"\[AMIM\]",
    r"\[C[0-9]+mim\]", r"\[C[0-9]+MIM\]", r"\[BMPYRR\]", r"\[N[0-9,]+\]",
    r"imidazolium", r"pyridinium", r"pyrrolidinium", r"ammonium", r"phosphonium",
    r"\[NTf2\]", r"\[Tf2N\]", r"\[BF4\]", r"\[PF6\]", r"\[DCA\]", r"\[OTf\]",
    r"\[TfO\]", r"\[NO3\]", r"\[OAc\]", r"\[MeSO4\]", r"\[Cl\]", r"\[Br\]",
    r"\[SCN\]", r"\[FAP\]", r"\[TCM\]",
]
IL_RE = re.compile("|".join(IL_PATTERNS), re.IGNORECASE)


def reconstruct_abstract(ai):
    if not ai:
        return ""
    words = {}
    for w, positions in ai.items():
        for p in positions:
            words[p] = w
    return " ".join(words[i] for i in sorted(words.keys()))


def run(args):
    all_rows = []
    seen = set()
    per_prop_rows = {s[0]: [] for s in SEARCHES}

    for label, query, must_abs, drop_title in SEARCHES:
        print(f"\n=== [{label}] {query} ===")
        must_re = re.compile(must_abs, re.IGNORECASE)
        drop_re = re.compile(drop_title, re.IGNORECASE)
        w = (Works()
             .search(query)
             .filter(from_publication_date=f"{args.from_year}-01-01")
             .filter(type="article")
             .sort(relevance_score="desc"))
        n_seen = n_kept = n_title_drop = n_abs_drop = n_no_il = 0
        for page in w.paginate(per_page=50, n_max=args.max_per_prop):
            for paper in page:
                n_seen += 1
                doi = (paper.get("doi") or "").replace("https://doi.org/", "")
                if not doi:
                    continue
                title = (paper.get("title") or "").strip()
                if drop_re.search(title):
                    n_title_drop += 1
                    continue
                abstract = reconstruct_abstract(paper.get("abstract_inverted_index"))
                if not must_re.search(abstract):
                    n_abs_drop += 1
                    continue
                il_hits = len(IL_RE.findall(abstract))
                if il_hits == 0:
                    n_no_il += 1
                    continue
                venue = ((paper.get("primary_location") or {}).get("source") or {}).get("display_name", "") or ""
                is_primary_journal = venue.lower() in JOURNAL_SET_LC
                # de-dup across properties
                key = (doi, label)
                if key in seen:
                    continue
                seen.add(key)
                oa = paper.get("open_access") or {}
                row = {
                    "doi": doi,
                    "year": paper.get("publication_year"),
                    "title": title.replace("\n", " "),
                    "venue": venue,
                    "is_primary_journal": is_primary_journal,
                    "is_oa": bool(oa.get("is_oa")),
                    "oa_url": oa.get("oa_url") or "",
                    "cited_by": paper.get("cited_by_count", 0),
                    "relevance_score": paper.get("relevance_score", 0),
                    "il_mentions_est": il_hits,
                    "target_prop": label,
                }
                all_rows.append(row)
                per_prop_rows[label].append(row)
                n_kept += 1
        print(f"  seen={n_seen}  kept={n_kept}  (title-drop={n_title_drop}, abs-drop={n_abs_drop}, no-IL={n_no_il})")
        time.sleep(0.2)

    # Rank: primary journal, then OA, then IL mentions, then citations.
    def sort_key(r):
        return (-int(r["is_primary_journal"]), -int(r["is_oa"]),
                -r["il_mentions_est"], -r["cited_by"])
    all_rows.sort(key=sort_key)

    cols = ["doi", "year", "title", "venue", "is_primary_journal",
            "is_oa", "oa_url", "cited_by", "relevance_score",
            "il_mentions_est", "target_prop"]
    with open(OUT_MERGED, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    for prop, rows in per_prop_rows.items():
        rows.sort(key=sort_key)
        path = OUT_DIR / f"candidate_papers_{prop}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        n_pj = sum(1 for r in rows if r["is_primary_journal"])
        n_oa = sum(1 for r in rows if r["is_oa"])
        print(f"  [{prop}] wrote {len(rows):3d} rows ({n_pj} primary-journal, {n_oa} OA) → {path.name}")
    print(f"\nMerged: {len(all_rows)} rows → {OUT_MERGED}")
    n_pj = sum(1 for r in all_rows if r["is_primary_journal"])
    n_oa = sum(1 for r in all_rows if r["is_oa"])
    print(f"  primary journal: {n_pj} | OA: {n_oa}")
    print("Top 10 overall:")
    for r in all_rows[:10]:
        tag = "PJ" if r["is_primary_journal"] else "--"
        tag += "OA" if r["is_oa"] else "  "
        print(f"  [{tag}] {r['year']} {r['target_prop']:9s} IL≈{r['il_mentions_est']:>2d}  {r['venue'][:28]:28s}  {r['title'][:60]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2000)
    ap.add_argument("--max-per-prop", type=int, default=400)
    args = ap.parse_args()
    run(args)
