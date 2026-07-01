#!/usr/bin/env python3
"""Path 2 — Render COSMO views for each proxy conformer in parallel.

Loops over data/pipeline/point_clouds_proxy/conf_{k}/*.npz and writes
lignos/data/cosmo_images_multi/conf_{k}/{hash}_frames/frame_*.png,
reusing render_cosmo_views.render_cosmo_views for the actual rendering.

Matplotlib 3D scatter rendering is CPU-bound and the 36 frames per molecule
dominate wall time (~5-10 s per molecule, serial). Parallelizing across
processes gives near-linear speedup.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lignos/scripts"))

from render_cosmo_views import render_cosmo_views  # noqa: E402


def _render_one(args):
    pc_path, frame_dir, n_views, resolution = args
    frame_dir = Path(frame_dir)
    if frame_dir.exists() and len(list(frame_dir.glob("frame_*.png"))) >= n_views:
        return (pc_path.stem, "skip", 0.0)
    t0 = time.time()
    try:
        render_cosmo_views(
            pc_path, frame_dir,
            n_views=n_views, resolution=resolution, render_ep=False,
        )
    except Exception as e:
        return (pc_path.stem, f"error: {type(e).__name__}: {e}", time.time() - t0)
    return (pc_path.stem, "ok", time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_conformers", type=int, default=5)
    ap.add_argument("--start_id", type=int, default=0)
    ap.add_argument("--pc_root", type=str,
                    default=str(PROJECT_ROOT / "data/pipeline/point_clouds_proxy"))
    ap.add_argument("--out_root", type=str,
                    default=str(PROJECT_ROOT / "lignos/data/cosmo_images_multi"))
    ap.add_argument("--n_views", type=int, default=36)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    pc_root = Path(args.pc_root)
    out_root = Path(args.out_root)

    jobs = []
    for k in range(args.start_id, args.start_id + args.n_conformers):
        pc_dir = pc_root / f"conf_{k}"
        if not pc_dir.exists():
            print(f"  SKIP conf_{k}: point-cloud dir missing")
            continue
        out_dir = out_root / f"conf_{k}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for pc in sorted(pc_dir.glob("*.npz")):
            frame_dir = out_dir / f"{pc.stem}_frames"
            jobs.append((pc, frame_dir, args.n_views, args.resolution))

    print(f"Rendering {len(jobs)} (IL × conformer) jobs across {args.workers} workers")
    t_start = time.time()
    ok = skip = err = 0
    times = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_render_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            stem, status, dt = fut.result()
            if status == "ok":
                ok += 1
                times.append(dt)
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"  [{i}/{len(jobs)}] {stem}: {status}")
            if i % 25 == 0:
                elapsed = time.time() - t_start
                rate = i / elapsed
                eta = (len(jobs) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(jobs)}] ok={ok} skip={skip} err={err}  "
                      f"rate={rate:.2f}/s  eta={eta:.0f}s")

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s: ok={ok} skip={skip} err={err}")
    if times:
        import numpy as np
        print(f"Per-job wall time: median={np.median(times):.2f}s  "
              f"mean={np.mean(times):.2f}s")


if __name__ == "__main__":
    main()
