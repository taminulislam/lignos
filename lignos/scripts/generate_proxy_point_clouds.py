#!/usr/bin/env python3
"""Path 2 — Generate proxy COSMO-like point clouds from SMILES (no DFT).

Produces .npz files matching the format render_cosmo_views.py expects:
    points: (N, 7) float32  — columns (x, y, z, nx, ny, nz, ESP)

The real DFT pipeline (Psi4 B3LYP/def2-SVP CPCM) takes ~3 min/molecule and
yields ~1000 surface points with CPCM-derived ESP values. This script
approximates that with RDKit ETKDG embedding + MMFF optimization + Gasteiger
partial charges + per-atom vdW-sphere surface sampling. Each conformer uses a
different ETKDG seed so geometries are genuinely distinct.

Output layout:
    point_clouds_proxy/conf_{k}/{pc_hash}.npz

pc_hash is md5(smiles)[:12] — same convention as the DFT pipeline — so the
downstream renderer and ViT feature extractor can treat proxy point clouds as
drop-in replacements.

Usage:
    python generate_proxy_point_clouds.py --n_conformers 5 --start_id 0
    python generate_proxy_point_clouds.py --n_conformers 1 --start_id 0 --limit 2   # smoke test
"""

import argparse
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

VDW_RADII = {
    1: 1.20, 3: 1.82, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    11: 2.27, 12: 1.73, 13: 1.84, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75,
    19: 2.75, 20: 2.31, 35: 1.85, 53: 1.98,
}

N_TARGET_POINTS = 1024  # ~matches DFT point-cloud density (~1000)


def smiles_to_hash(smiles: str) -> str:
    return hashlib.md5(smiles.encode()).hexdigest()[:12]


def embed_il(smiles: str, seed: int):
    """ETKDG-embed a full IL (cation.anion) and MMFF-optimize.

    Returns (mol_with_conformer, gasteiger_charges: np.ndarray).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True  # helps with salts / disconnected fragments
    if AllChem.EmbedMolecule(mol, params) == -1:
        # Fallback: plain ETKDG
        fb = AllChem.ETKDG()
        fb.randomSeed = int(seed)
        fb.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, fb) == -1:
            raise RuntimeError(f"ETKDG embedding failed for seed={seed}")

    # Optimize (MMFF first, UFF fallback)
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props) if props is not None else None
        if ff is not None:
            ff.Minimize(maxIts=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        # Geometry is still usable even if minimization fails
        pass

    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array([
        float(a.GetProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0
        for a in mol.GetAtoms()
    ], dtype=np.float32)
    charges[~np.isfinite(charges)] = 0.0
    return mol, charges


def sample_surface(mol, charges: np.ndarray, seed: int, n_target: int = N_TARGET_POINTS):
    """Sample atomic-vdW-sphere surface points with normals and ESP proxy.

    Uses a Fibonacci-like uniform sphere sampler per atom so the point
    distribution is reproducible given a seed.
    """
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    n_atoms = positions.shape[0]

    # Allocate points per atom roughly proportional to vdW surface area
    radii = np.array([
        VDW_RADII.get(a.GetAtomicNum(), 1.70) for a in mol.GetAtoms()
    ], dtype=np.float32)
    areas = radii ** 2
    weights = areas / areas.sum()
    pts_per_atom = np.maximum(6, (weights * n_target).astype(int))

    rng = np.random.default_rng(seed)

    all_pts, all_norms, all_esp = [], [], []

    for i in range(n_atoms):
        n = int(pts_per_atom[i])
        # Fibonacci sphere with jitter
        k = np.arange(n, dtype=np.float64) + 0.5 + rng.uniform(-0.25, 0.25, size=n)
        phi = np.arccos(1 - 2 * k / n)
        theta = np.pi * (1 + 5 ** 0.5) * k  # golden angle

        nx = np.sin(phi) * np.cos(theta)
        ny = np.sin(phi) * np.sin(theta)
        nz = np.cos(phi)
        normals = np.stack([nx, ny, nz], axis=1).astype(np.float32)

        pts = positions[i] + radii[i] * normals
        all_pts.append(pts.astype(np.float32))
        all_norms.append(normals)
        all_esp.append(np.full(n, charges[i], dtype=np.float32))

    pts = np.concatenate(all_pts, axis=0)
    norms = np.concatenate(all_norms, axis=0)
    esp = np.concatenate(all_esp, axis=0)

    # Remove points buried inside other atoms (basic solvent-accessible filter)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(n_atoms):
        d2 = np.sum((pts - positions[i]) ** 2, axis=1)
        # point is inside atom i if closer than (radii[i] - eps) to atom center
        mask = d2 < (radii[i] - 0.1) ** 2
        keep &= ~mask
    pts = pts[keep]
    norms = norms[keep]
    esp = esp[keep]

    if len(pts) == 0:
        raise RuntimeError("All surface points were filtered out")

    # Downsample (or upsample by sampling with replacement) to N_TARGET
    if len(pts) > n_target:
        idx = rng.choice(len(pts), n_target, replace=False)
    else:
        idx = rng.choice(len(pts), n_target, replace=True)
    pts = pts[idx]
    norms = norms[idx]
    esp = esp[idx]

    # Zero-center coordinates (DFT pipeline does this implicitly)
    pts = pts - pts.mean(axis=0, keepdims=True)

    # Zero-mean ESP (DFT point clouds have mean≈0)
    esp = esp - esp.mean()

    points = np.concatenate([pts, norms, esp[:, None]], axis=1).astype(np.float32)
    return points  # (N_TARGET, 7)


def process_one(args):
    smiles, pc_hash, seed, out_path = args
    if out_path.exists():
        return (pc_hash, "skip", 0.0)
    t0 = time.time()
    try:
        mol, charges = embed_il(smiles, seed)
        points = sample_surface(mol, charges, seed)
    except Exception as e:
        return (pc_hash, f"error: {type(e).__name__}: {e}", time.time() - t0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, points=points)
    return (pc_hash, "ok", time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_conformers", type=int, default=5,
                    help="Total conformers to produce per IL")
    ap.add_argument("--start_id", type=int, default=0,
                    help="First conformer id (inclusive)")
    ap.add_argument("--master_index", type=str,
                    default=str(PROJECT_ROOT / "lignos/data/master_index.csv"))
    ap.add_argument("--output_root", type=str,
                    default=str(PROJECT_ROOT / "data/pipeline/point_clouds_proxy"))
    ap.add_argument("--limit", type=int, default=0,
                    help="If > 0, process only the first N ILs (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--source", choices=["master_index", "cached_v4"],
                    default="cached_v4",
                    help="SMILES source. 'cached_v4' uses cosmobridge_v4/data/cached_*.npz "
                         "(matches the 223-sample eval split). 'master_index' filters by "
                         "data_complete=True (133 samples, over-filtered).")
    args = ap.parse_args()

    if args.source == "cached_v4":
        smiles_set = []
        for split in ["train", "val", "test"]:
            p = PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz"
            d = np.load(p, allow_pickle=True)
            smiles_set.extend(list(d["smiles"]))
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for s in smiles_set:
            if isinstance(s, bytes):
                s = s.decode()
            if s and s not in seen:
                seen.add(s)
                uniq.append(s)
        df = pd.DataFrame({"smiles": uniq})
        print(f"Loaded {len(df)} unique SMILES from cached_v4 splits")
    else:
        df = pd.read_csv(args.master_index)
        df = df[df["data_complete"] == True].reset_index(drop=True)
        print(f"Loaded {len(df)} complete ILs from master_index")
    if args.limit > 0:
        df = df.head(args.limit)

    out_root = Path(args.output_root)
    jobs = []
    for _, row in df.iterrows():
        smi = row["smiles"]
        if not isinstance(smi, str) or not smi:
            continue
        pc_hash = smiles_to_hash(smi)
        # Sanity: pc_hash in master_index must match our hash function
        if "pc_hash" in row and isinstance(row["pc_hash"], str) and row["pc_hash"] != pc_hash:
            print(f"  WARN: hash mismatch for {row.get('compound_id', '?')}: "
                  f"master={row['pc_hash']} vs computed={pc_hash}")
        for k in range(args.start_id, args.start_id + args.n_conformers):
            out_path = out_root / f"conf_{k}" / f"{pc_hash}.npz"
            seed = k * 10_000 + (abs(hash(pc_hash)) % 10_000)
            jobs.append((smi, pc_hash, seed, out_path))

    print(f"Dispatching {len(jobs)} (IL × conformer) jobs across {args.workers} workers")
    t_start = time.time()
    ok = skip = err = 0
    times = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            pc_hash, status, dt = fut.result()
            if status == "ok":
                ok += 1
                times.append(dt)
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"  [{i}/{len(jobs)}] {pc_hash}: {status}")
            if i % 50 == 0:
                elapsed = time.time() - t_start
                rate = i / elapsed
                eta = (len(jobs) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(jobs)}] ok={ok} skip={skip} err={err}  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s")

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s: ok={ok} skip={skip} err={err}")
    if times:
        print(f"Per-job wall time: median={np.median(times):.2f}s  "
              f"mean={np.mean(times):.2f}s  max={np.max(times):.2f}s")


if __name__ == "__main__":
    main()
