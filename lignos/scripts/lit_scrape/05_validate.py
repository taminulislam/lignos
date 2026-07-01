"""Option Y — Stage 5: reliability gate.

Runs after 04_normalize.py. Before any rows touch `cached_train.npz`, check:

  A. Schema   — every row has (SMILES, T, x1 or T+target-only, >=1 target value)
  B. Physical — T in 250-700 K, x1 in [0,1], targets in expected ranges
  C. Dedup    — no duplicate rows within the new batch AND no duplicates vs
                the existing training cache (same DOI+T+SMILES+x1+props)
  D. SMILES   — every il_smiles parses with RDKit; canonicalize; confirm cation+anion
  E. Coverage — report per-property / per-paper / per-IL counts so the user can
                gauge how much signal each paper contributes

Outputs:
  data/lit_scrape/validation_report.md   — human-readable go/no-go per row/table
  data/lit_scrape/validated_rows.csv     — subset that passes all checks; ready
                                           to be merged into the training cache
  data/lit_scrape/rejected_rows.csv      — with reason column

Exit code: 0 if at least one row passes, 1 if nothing passes.

Usage:
    python 05_validate.py [--cache lignos/data/LignoIL_unified_v2/cached_train.npz]
                          [--high-conf data/lit_scrape/high_confidence.csv
                                       data/lit_scrape/high_confidence_camelot.csv]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "lit_scrape"
REPORT_MD = OUT_DIR / "validation_report.md"
VALID_CSV = OUT_DIR / "validated_rows.csv"
REJECT_CSV = OUT_DIR / "rejected_rows.csv"

TARGET_KEYS = ["gamma1", "gamma2", "gamma_inf", "H_E", "H_vap",
               "G_E", "G_mix", "P", "activity"]

# Same ranges as 04_normalize but enforced again as a second line of defense.
RANGES = {
    "T":         (250.0, 700.0),
    "x1":        (0.0, 1.0),
    "x2":        (0.0, 1.0),
    "gamma1":    (0.01, 100.0),
    "gamma2":    (0.01, 100.0),
    "gamma_inf": (0.01, 1000.0),
    "G_E":       (-50000, 50000),
    "H_E":       (-50000, 50000),
    "G_mix":     (-50000, 50000),
    "H_vap":     (0, 300000),
    "P":         (0, 1e9),
    "activity":  (0, 100.0),
}


def canon_smiles(smi: str):
    if not isinstance(smi, str) or not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def is_valid_il(smi: str):
    """Need at least one positively-charged fragment AND one negatively-charged
    fragment — catches the OPSIN-returns-cation-only failure mode."""
    m = Chem.MolFromSmiles(smi) if smi else None
    if not m:
        return False
    frags = Chem.GetMolFrags(m, asMols=True)
    if len(frags) < 2:
        return False
    pos = neg = False
    for f in frags:
        q = sum(a.GetFormalCharge() for a in f.GetAtoms())
        if q > 0: pos = True
        elif q < 0: neg = True
    return pos and neg


def load_existing_cache(cache_path: Path):
    """Return a set of row-fingerprints from the current training cache for
    dedup. Fingerprint: (SMILES_canon, round(T,2), round(x1,3)) — x1 absent
    in the schema so we use a placeholder. Returns empty set if cache missing."""
    if not cache_path or not Path(cache_path).exists():
        return set()
    z = np.load(cache_path, allow_pickle=True)
    smiles = [canon_smiles(s) for s in z["smiles"]]
    # Cache lacks per-row T or x1. We can only dedup by SMILES+target-vector.
    # Build target fingerprint for each row.
    t = z["targets"]
    fps = set()
    for i, s in enumerate(smiles):
        if s is None:
            continue
        tv = tuple(None if np.isnan(x) else round(float(x), 4) for x in t[i])
        fps.add((s, tv))
    return fps


def validate_row(rec, existing_fps, fresh_fps):
    """Return (ok: bool, reason: str). Mutates rec['il_smiles'] to canonical."""
    # A. Schema
    if not rec.get("T") or not isinstance(rec.get("T"), (int, float)):
        return False, "missing_T"
    if not rec.get("il_smiles"):
        return False, "missing_il_smiles"
    # D. SMILES
    smi = canon_smiles(rec["il_smiles"])
    if not smi or not is_valid_il(smi):
        return False, f"invalid_il_smiles:{rec['il_smiles'][:40]}"
    rec["il_smiles"] = smi
    # B. Physical ranges
    for k, (lo, hi) in RANGES.items():
        v = rec.get(k)
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            return False, f"non_numeric:{k}"
        if not (lo <= v <= hi):
            return False, f"out_of_range:{k}={v}"
        rec[k] = v
    # Must have ≥1 target numeric
    tgt_vals = {k: rec.get(k) for k in TARGET_KEYS if rec.get(k) is not None}
    if not tgt_vals:
        return False, "no_target_value"
    # x1 requirement: most targets are composition-dependent. γ∞ (infinite
    # dilution) is x1→0 by definition and does NOT need x1.
    needs_x1 = bool(set(tgt_vals) & {"gamma1", "gamma2", "H_E", "G_E", "G_mix", "activity"})
    if needs_x1 and rec.get("x1") is None:
        return False, f"needs_x1_for:{','.join(sorted(tgt_vals))}"
    # γ∞ needs a solute SMILES — (IL, T, γ∞) alone is ambiguous.
    if "gamma_inf" in tgt_vals and not rec.get("solute_smiles"):
        return False, "gamma_inf_needs_solute_smiles"
    # Validate the solute SMILES parses and is neutral (not another salt fragment)
    if rec.get("solute_smiles"):
        m = Chem.MolFromSmiles(rec["solute_smiles"])
        if m is None or sum(a.GetFormalCharge() for a in m.GetAtoms()) != 0:
            return False, "invalid_solute_smiles"
        rec["solute_smiles"] = Chem.MolToSmiles(m)
    # γ∞ rows with solute are valid for a separate gamma_inf dataset but are
    # NOT mergeable into the current cached_train.npz schema (targets list has
    # no gamma_inf slot). Flag them as "cache_schema_pending" so the user can
    # still harvest them into a gamma_inf-specific file.
    if "gamma_inf" in tgt_vals:
        return False, "valid_but_gamma_inf_needs_schema_extension"
    # C. Dedup within fresh batch
    fp_fresh = (smi, round(float(rec["T"]), 2),
                round(float(rec.get("x1") or -1.0), 3),
                tuple(sorted(tgt_vals.items())))
    if fp_fresh in fresh_fps:
        return False, "duplicate_within_batch"
    fresh_fps.add(fp_fresh)
    # C. Dedup vs existing cache (SMILES + target-vector match)
    tgt_vec = tuple(round(float(rec.get(k)), 4) if rec.get(k) is not None else None
                    for k in ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"])
    if (smi, tgt_vec) in existing_fps:
        return False, "duplicate_vs_training_cache"
    return True, "ok"


def validate_file(path: Path, existing_fps: set, fresh_fps: set):
    rows_ok, rows_rej = [], []
    if not path.exists() or path.stat().st_size == 0:
        return rows_ok, rows_rej
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return rows_ok, [{"source": str(path), "reason": f"read_error:{e}"}]
    for _, r in df.iterrows():
        rec = {k: (r[k] if pd.notna(r[k]) else None) for k in r.index}
        ok, reason = validate_row(rec, existing_fps, fresh_fps)
        if ok:
            rows_ok.append(rec)
        else:
            rec["_reject_reason"] = reason
            rows_rej.append(rec)
    return rows_ok, rows_rej


def coverage_report(rows):
    """Summaries by property, paper, IL."""
    lines = []
    if not rows:
        return "No valid rows.\n"
    df = pd.DataFrame(rows)
    lines.append(f"Total valid rows: {len(df)}")
    lines.append(f"Unique ILs (canonical SMILES): {df['il_smiles'].nunique()}")
    lines.append(f"Unique source papers: {df['source_doi'].nunique()}")
    lines.append("")
    lines.append("Per-property row counts (numeric, non-null):")
    for k in TARGET_KEYS:
        if k in df.columns:
            n = df[k].notna().sum()
            if n:
                vals = df[k].dropna()
                lines.append(f"  {k:10s} n={n:4d}  range=[{vals.min():.3g}, {vals.max():.3g}]  mean={vals.mean():.3g}")
    lines.append("")
    lines.append("Top source papers:")
    for doi, n in df["source_doi"].value_counts().head(10).items():
        lines.append(f"  {n:4d}  {doi}")
    return "\n".join(lines) + "\n"


def main(args):
    cache_path = Path(args.cache) if args.cache else (
        PROJECT_ROOT / "lignos" / "data" / "LignoIL_unified_v2" / "cached_train.npz")
    print(f"Training cache:   {cache_path}")
    existing_fps = load_existing_cache(cache_path)
    print(f"  fingerprints: {len(existing_fps)} rows for dedup")
    fresh_fps: set = set()

    ok_all, rej_all = [], []
    for path_str in args.high_conf:
        path = Path(path_str)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        print(f"\nValidating {path.name} ...")
        ok, rej = validate_file(path, existing_fps, fresh_fps)
        print(f"  ok={len(ok)}  rejected={len(rej)}")
        for r in rej[:5]:
            print(f"    - {r.get('_reject_reason','?')}")
        ok_all.extend(ok)
        rej_all.extend(rej)

    # Split gamma_inf rows (valid but cache-schema pending) into their own file.
    gi_rows = [r for r in rej_all
               if r.get("_reject_reason") == "valid_but_gamma_inf_needs_schema_extension"]
    rej_all = [r for r in rej_all
               if r.get("_reject_reason") != "valid_but_gamma_inf_needs_schema_extension"]

    # Write outputs
    def _write(rows, path):
        if not rows:
            path.write_text("")
            return
        cols = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: (r.get(c) if r.get(c) is not None else "") for c in cols})

    _write(ok_all, VALID_CSV)
    _write(rej_all, REJECT_CSV)
    gi_csv = OUT_DIR / "validated_gamma_inf.csv"
    _write(gi_rows, gi_csv)
    print(f"gamma_inf (valid, pending schema): {len(gi_rows)} → {gi_csv}")

    # Report
    report = ["# Lit-scrape validation report\n",
              f"Cache reference: `{cache_path}` ({len(existing_fps)} existing fingerprints)\n",
              f"Inputs: {', '.join(args.high_conf)}\n",
              f"\n## Summary\n",
              f"- Valid rows:    **{len(ok_all)}** → `{VALID_CSV.name}`",
              f"- Rejected rows: **{len(rej_all)}** → `{REJECT_CSV.name}`\n",
              "\n## Coverage\n", coverage_report(ok_all)]

    if rej_all:
        reasons = pd.Series([r["_reject_reason"] for r in rej_all]).value_counts()
        report.append("\n## Top rejection reasons\n")
        for reason, n in reasons.items():
            report.append(f"- {n:4d}  {reason}")
        report.append("")

    REPORT_MD.write_text("\n".join(report))
    print(f"\nReport: {REPORT_MD}")
    print(f"Valid:  {VALID_CSV} ({len(ok_all)} rows)")
    print(f"Reject: {REJECT_CSV} ({len(rej_all)} rows)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, default=None,
                    help="Training cache for dedup (default: LignoIL_unified_v2/cached_train.npz)")
    ap.add_argument("--high-conf", nargs="+", default=[
        "data/lit_scrape/high_confidence.csv",
        "data/lit_scrape/high_confidence_camelot.csv"])
    args = ap.parse_args()
    sys.exit(main(args))
