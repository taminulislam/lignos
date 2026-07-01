#!/usr/bin/env python3
"""Extract 1024-point clouds from LIGNO DFT surfaces.

Thin wrapper around scripts/pipeline/step4c_extract_point_clouds helpers.
step4c's main() is wired to the legacy 28-IL + ILThermo CSVs and does not see
our LIGNO_* compounds, so this script:

  1. Reads data/pipeline/lignoil_new_compounds.csv  (compound_id → SMILES, cat, an)
  2. For each compound_id, loads the pair/cation/anion DFT .npz
     from data/pipeline/dft_surface/
  3. Samples 1024-point clouds via the existing sample_point_cloud +
     farthest-point-sampling routines
  4. Writes per-fragment outputs to data/pipeline/point_clouds_ligno/
  5. Writes index.csv mapping (compound_id, fragment, smiles) → .npz path

Output .npz schema (same as step4c): points (1024, 7) = [x,y,z, nx,ny,nz, esp].
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline"))
from step4c_extract_point_clouds import load_dft_surface  # noqa: E402

PIPELINE = PROJECT_ROOT / "data" / "pipeline"
COMPOUND_CSV = PIPELINE / "lignoil_new_compounds.csv"
SURFACE_DIR = PIPELINE / "dft_surface"
OUT_DIR = PIPELINE / "point_clouds_ligno"
INDEX_CSV = OUT_DIR / "index.csv"
N_POINTS = 1024


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(COMPOUND_CSV)
    print(f"Compounds to process: {len(df)}")

    rng = np.random.default_rng(42)
    rows = []
    ok = missing = failed = 0
    for _, c in df.iterrows():
        cid = c["compound_id"]
        for frag, smi_col in [("pair", "smiles"),
                               ("cation", "cation_smiles"),
                               ("anion", "anion_smiles")]:
            smi = c[smi_col]
            src = SURFACE_DIR / f"{cid}_{frag}.npz"
            if not src.exists():
                missing += 1
                print(f"  MISS {cid}_{frag}: surface missing")
                continue
            try:
                # step4c's load_dft_surface uses a module-level np.random for
                # farthest-point sampling; seed per-call so runs are reproducible.
                np.random.seed(abs(hash((cid, frag))) % (2**31))
                pts = load_dft_surface(src, n_points=N_POINTS)
            except Exception as e:
                failed += 1
                print(f"  FAIL {cid}_{frag}: {type(e).__name__}: {e}")
                continue
            out = OUT_DIR / f"{cid}_{frag}.npz"
            np.savez(out, points=pts.astype(np.float32),
                     compound_id=cid, fragment=frag, smiles=smi)
            rows.append({
                "compound_id": cid, "fragment": frag,
                "smiles": smi, "file": out.name,
                "n_points": pts.shape[0],
            })
            ok += 1

    with open(INDEX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["compound_id", "fragment",
                                            "smiles", "file", "n_points"])
        w.writeheader(); w.writerows(rows)

    print(f"\nok={ok}  missing={missing}  failed={failed}")
    print(f"Output: {OUT_DIR}")
    print(f"Index:  {INDEX_CSV}")


if __name__ == "__main__":
    main()
