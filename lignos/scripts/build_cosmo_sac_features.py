"""A5.3 — COSMO-SAC σ-profile feature bank.

For every DFT-surface-covered IL, compute a 20-D physics-motivated descriptor
vector from the σ (surface-charge-density) column of its _pair.npz surface.
These features are:
  - derived from DFT (generalize to new ILs by construction — not learned)
  - cheap (seconds per IL)
  - the raw inputs to COSMO-SAC activity-coefficient prediction

Feature layout per IL (20-D):
  [ 0] n_points            total surface tessellation points (area proxy)
  [ 1] σ_mean              average surface charge density
  [ 2] σ_std               spread
  [ 3] σ_min               most negative (strongest HB-acceptor region)
  [ 4] σ_max               most positive (strongest HB-donor region)
  [ 5] frac_HB_donor       fraction with σ > +0.0084 e/Å² (electron-poor)
  [ 6] frac_HB_acceptor    fraction with σ < −0.0084 e/Å² (electron-rich)
  [ 7] frac_nonpolar       fraction with |σ| ≤ 0.0084
  [ 8] skewness            3rd moment
  [ 9] kurtosis            4th moment
  [10..19]                 10-bin histogram of σ in [−0.025, +0.025]

SMILES resolution:
  - geometry_status.csv (243 rows, original pipeline)
  - tier3_compounds.csv (156 rows, new ILs)
  - merged geometry_status.csv (381 rows, includes CACHEB_*)
  - If a DFT `{cid}_pair.npz` exists, use it; otherwise fall back to cation+anion
    separate surfaces (average).

Output: lignos/data/cosmo_sac_feat_bank.npz
  smiles       : (N,) object   canonical SMILES
  cosmo_feat   : (N, 20) float32
  compound_ids : (N,) object

Later use (in A5.3 trainer):
  Add `cosmo_feat` as a zero-init gated residual branch with mask
  `has_cosmo = (cosmo_feat != 0).any(axis=1)`. Branch is Tier-1 OOD therapy:
  σ-profile descriptors generalize across IL chemistry via DFT.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
DFT_SURF = PROJECT_ROOT / "data" / "pipeline" / "dft_surface"
OUT = V5 / "data" / "cosmo_sac_feat_bank.npz"

HB_THRESHOLD = 0.0084  # e/Å²  (standard COSMO-SAC cutoff for HB contribution)
SIGMA_RANGE = (-0.025, 0.025)
N_BINS = 10
FEAT_DIM = 20


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


def compute_descriptors(sigma: np.ndarray) -> np.ndarray:
    """20-D σ-profile descriptor vector."""
    x = sigma.astype(np.float64)
    n = len(x)
    out = np.zeros(FEAT_DIM, dtype=np.float32)
    if n == 0:
        return out
    mu = x.mean(); sd = x.std() + 1e-12
    out[0] = float(n)
    out[1] = mu
    out[2] = sd
    out[3] = x.min()
    out[4] = x.max()
    out[5] = float((x >  HB_THRESHOLD).sum()) / n
    out[6] = float((x < -HB_THRESHOLD).sum()) / n
    out[7] = float((np.abs(x) <= HB_THRESHOLD).sum()) / n
    # Moments
    z = (x - mu) / sd
    out[8] = float((z ** 3).mean())
    out[9] = float((z ** 4).mean() - 3.0)  # excess kurtosis
    # Histogram
    hist, _ = np.histogram(x, bins=N_BINS, range=SIGMA_RANGE, density=True)
    out[10:10 + N_BINS] = hist.astype(np.float32)
    return out


def load_sigma(cid: str) -> np.ndarray | None:
    """Load σ column from {cid}_pair.npz. Fall back to cation+anion average."""
    pair = DFT_SURF / f"{cid}_pair.npz"
    if pair.exists():
        d = np.load(pair, allow_pickle=True)
        if "surface" in d.files:
            return d["surface"][:, 6].astype(np.float32)
    cat = DFT_SURF / f"{cid}_cation.npz"
    an = DFT_SURF / f"{cid}_anion.npz"
    parts = []
    for p in (cat, an):
        if p.exists():
            d = np.load(p, allow_pickle=True)
            if "surface" in d.files:
                parts.append(d["surface"][:, 6].astype(np.float32))
    if parts:
        return np.concatenate(parts)
    return None


def compound_id_map():
    geom = pd.read_csv(PROJECT_ROOT / "data/pipeline/geometry_status.csv")
    tier3 = pd.read_csv(PROJECT_ROOT / "data/pipeline/tier3_compounds.csv")
    mapping = {}
    for _, r in geom.iterrows():
        mapping[r["compound_id"]] = r["smiles"]
    for _, r in tier3.iterrows():
        mapping[r["compound_id"]] = r["smiles"]
    return mapping


def main():
    id_to_smi = compound_id_map()
    print(f"compound_id → SMILES mappings: {len(id_to_smi)}")

    results = {}
    n_ok = n_noil = n_nodft = 0
    for cid, smi in id_to_smi.items():
        cs = canon(smi)
        if not cs:
            n_noil += 1
            continue
        if cs in results:
            continue
        sigma = load_sigma(cid)
        if sigma is None:
            n_nodft += 1
            continue
        feat = compute_descriptors(sigma)
        results[cs] = (feat, cid)
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"  [{n_ok}] cid={cid}  σ n={len(sigma)}  "
                   f"HBd={feat[5]:.3f}  HBa={feat[6]:.3f}")

    smis = np.array(list(results.keys()), dtype=object)
    feats = np.stack([results[s][0] for s in smis])
    cids = np.array([results[s][1] for s in smis], dtype=object)
    np.savez(OUT, smiles=smis, cosmo_feat=feats, compound_ids=cids)
    print(f"\nWrote {len(smis)} ILs × {feats.shape[1]}D → {OUT}")
    print(f"  encoded: {n_ok}   skipped (bad SMILES): {n_noil}   missing DFT: {n_nodft}")

    # Coverage diagnostic vs LignoIL_A1 train
    tr = np.load(V5 / "data/LignoIL_A1/cached_train.npz", allow_pickle=True)
    bank_smis = set(smis)
    cache_smis = {canon(s) for s in tr["smiles"]} - {None}
    overlap = cache_smis & bank_smis
    n_rows = sum(1 for s in tr["smiles"] if canon(s) in bank_smis)
    print(f"\nLignoIL_A1 train: unique SMILES {len(cache_smis)}, "
          f"bank overlap {len(overlap)}, rows {n_rows}/{len(tr['smiles'])} "
          f"({n_rows/len(tr['smiles']):.1%})")


if __name__ == "__main__":
    main()
