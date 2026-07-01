#!/usr/bin/env python3
"""Sigma-Surface Unfolding: 3D COSMO surface -> 2D Mercator projection.

Converts a 3D point cloud to a 2D charge map via spherical projection.
Like a Mercator map of Earth, this creates a consistent 2D representation
where pixel position encodes surface location and color encodes charge.

Usage:
    python unfold_surface.py --compound_id AAQcOE
    python unfold_surface.py --all --output_dir data/sigma_maps/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def unfold_cosmo_surface(point_cloud_path, resolution=(128, 256)):
    """Convert a 3D COSMO point cloud into a 2D charge map.

    Uses spherical (Mercator-like) projection: theta -> row, phi -> column.

    Args:
        point_cloud_path: path to .npz file with 'points' (N, 7)
        resolution: (height, width) of output map

    Returns:
        charge_map: (H, W) raw ESP values
        rgb_map: (H, W, 3) colorized version
        coverage_mask: (H, W) bool, True where surface exists
    """
    from scipy.ndimage import distance_transform_edt

    data = np.load(point_cloud_path)
    points = data["points"]  # (N, 7)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    esp = points[:, 6]

    # Center coordinates
    x = x - x.mean()
    y = y - y.mean()
    z = z - z.mean()

    # Convert to spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-8
    theta = np.arccos(np.clip(z / r, -1, 1))   # [0, pi] latitude
    phi = np.arctan2(y, x) + np.pi              # [0, 2pi] longitude

    H, W = resolution
    row = np.clip((theta / np.pi * (H - 1)).astype(int), 0, H - 1)
    col = np.clip((phi / (2 * np.pi) * (W - 1)).astype(int), 0, W - 1)

    # Accumulate charge values
    charge_map = np.zeros((H, W))
    count_map = np.zeros((H, W))

    for i in range(len(esp)):
        charge_map[row[i], col[i]] += esp[i]
        count_map[row[i], col[i]] += 1

    # Average overlapping points
    coverage_mask = count_map > 0
    charge_map[coverage_mask] /= count_map[coverage_mask]

    # Fill gaps with nearest-neighbor interpolation
    if (~coverage_mask).any():
        indices = distance_transform_edt(
            ~coverage_mask, return_distances=False, return_indices=True
        )
        charge_map = charge_map[tuple(indices)]

    # Colorize
    from matplotlib.cm import RdBu_r
    vmin, vmax = np.percentile(charge_map[coverage_mask], [2, 98])
    normed = np.clip((charge_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
    rgb_map = RdBu_r(normed)[:, :, :3]

    return charge_map, rgb_map, coverage_mask


def save_sigma_map(charge_map, rgb_map, output_path):
    """Save both raw and colorized sigma maps.

    Args:
        charge_map: (H, W) raw values
        rgb_map: (H, W, 3) colorized
        output_path: base path (extensions added automatically)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save raw as .npz for model input
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        charge_map=charge_map.astype(np.float32),
    )

    # Save colorized as .png for visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    ax.imshow(rgb_map, aspect="auto")
    ax.set_xlabel("Longitude (phi)")
    ax.set_ylabel("Latitude (theta)")
    ax.set_title(output_path.stem)
    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Unfold COSMO surfaces")
    parser.add_argument("--compound_id", type=str)
    parser.add_argument("--all", action="store_true",
                        help="Process all available point clouds")
    parser.add_argument("--point_cloud_dir", type=str,
                        default=str(PROJECT_ROOT / "data/pipeline/point_clouds"))
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "lignos/data/sigma_maps"))
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    pc_dir = Path(args.point_cloud_dir)
    out_dir = Path(args.output_dir)
    resolution = (args.height, args.width)

    if args.compound_id:
        compounds = [args.compound_id]
    elif args.all:
        compounds = [p.stem for p in pc_dir.glob("*.npz")]
    else:
        parser.error("Specify --compound_id or --all")
        return

    print(f"Unfolding {len(compounds)} surfaces at {resolution}")

    for i, cid in enumerate(compounds):
        pc_path = pc_dir / f"{cid}.npz"
        out_path = out_dir / f"{cid}"

        if not pc_path.exists():
            print(f"  [{i+1}/{len(compounds)}] SKIP {cid}: no point cloud")
            continue

        if out_path.with_suffix(".npz").exists():
            print(f"  [{i+1}/{len(compounds)}] SKIP {cid}: already unfolded")
            continue

        print(f"  [{i+1}/{len(compounds)}] Unfolding {cid}...")
        charge_map, rgb_map, mask = unfold_cosmo_surface(pc_path, resolution)
        save_sigma_map(charge_map, rgb_map, out_path)

    print("Done!")


if __name__ == "__main__":
    main()
