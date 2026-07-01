#!/usr/bin/env python3
"""Render 36-view COSMO surface images for a molecule.

Generates rotation frames at 10-degree increments around the molecule,
with COSMO charge density colormap and optional electrostatic potential overlay.

Usage:
    python render_cosmo_views.py --compound_id AAQcOE --n_views 36 --resolution 224
    python render_cosmo_views.py --compound_list missing_compounds.txt --output_dir data/cosmo_images/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def render_cosmo_views(
    point_cloud_path,
    output_dir,
    n_views=36,
    resolution=224,
    render_ep=True,
    colormap="RdBu_r",
):
    """Render rotation views of a COSMO surface from a point cloud.

    Args:
        point_cloud_path: path to .npz with 'points' array (N, 7)
        output_dir: directory to save rendered frames
        n_views: number of rotation views (evenly spaced around Y-axis)
        resolution: output image resolution (square)
        render_ep: also render electrostatic potential overlay
        colormap: matplotlib colormap for charge coloring

    Returns:
        list of saved image paths
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    from mpl_toolkits.mplot3d import Axes3D

    data = np.load(point_cloud_path)
    points = data["points"]  # (N, 7): x, y, z, nx, ny, nz, ESP

    coords = points[:, :3]
    normals = points[:, 3:6]
    esp = points[:, 6]

    # Center and scale
    center = coords.mean(axis=0)
    coords = coords - center
    scale = np.abs(coords).max()
    coords = coords / (scale + 1e-8)

    # Normalize ESP for coloring
    vmin, vmax = np.percentile(esp, [2, 98])
    esp_norm = np.clip((esp - vmin) / (vmax - vmin + 1e-8), 0, 1)

    cmap = get_cmap(colormap)
    colors = cmap(esp_norm)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    angles = np.linspace(0, 360, n_views, endpoint=False)

    for i, angle in enumerate(angles):
        fig = plt.figure(figsize=(resolution / 100, resolution / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        # Rotate around Y-axis
        theta = np.radians(angle)
        rot = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ])
        rotated = coords @ rot.T

        # Sort by depth for proper occlusion
        depth = rotated[:, 2]
        order = np.argsort(depth)

        # Compute lighting (simple diffuse from camera direction)
        rot_normals = normals @ rot.T
        light_dir = np.array([0, 0, 1])
        diffuse = np.clip(np.dot(rot_normals, light_dir), 0.2, 1.0)

        # Apply lighting to colors
        lit_colors = colors[order].copy()
        lit_colors[:, :3] *= diffuse[order, np.newaxis]

        ax.scatter(
            rotated[order, 0],
            rotated[order, 1],
            rotated[order, 2],
            c=lit_colors,
            s=max(1, 50000 // len(coords)),
            alpha=0.9,
            edgecolors="none",
        )

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.axis("off")
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        # Remove margins
        ax.set_position([0, 0, 1, 1])

        frame_path = output_dir / f"frame_{i:03d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches="tight",
                    pad_inches=0, facecolor="white")
        plt.close(fig)
        saved_paths.append(frame_path)

    # Also save a canonical "cosmo" view (front-facing)
    _save_canonical_view(coords, colors, normals, output_dir, resolution, "cosmo")

    if render_ep:
        # EP view: same geometry but with EP-specific colormap
        ep_cmap = get_cmap("coolwarm")
        ep_colors = ep_cmap(esp_norm)
        _save_canonical_view(coords, ep_colors, normals, output_dir, resolution, "ep")

    return saved_paths


def _save_canonical_view(coords, colors, normals, output_dir, resolution, suffix):
    """Save a single canonical front-facing view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(resolution / 100, resolution / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    light_dir = np.array([0, 0, 1])
    diffuse = np.clip(np.dot(normals, light_dir), 0.2, 1.0)

    depth = coords[:, 2]
    order = np.argsort(depth)

    lit_colors = colors[order].copy()
    lit_colors[:, :3] *= diffuse[order, np.newaxis]

    ax.scatter(
        coords[order, 0], coords[order, 1], coords[order, 2],
        c=lit_colors, s=max(1, 50000 // len(coords)),
        alpha=0.9, edgecolors="none",
    )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_position([0, 0, 1, 1])

    mol_id = output_dir.name.replace("_frames", "")
    out_path = output_dir.parent / f"{mol_id}_{suffix}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight",
                pad_inches=0, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render COSMO surface views")
    parser.add_argument("--compound_id", type=str, help="Single compound ID")
    parser.add_argument("--compound_list", type=str,
                        help="File with one compound ID per line")
    parser.add_argument("--point_cloud_dir", type=str,
                        default=str(PROJECT_ROOT / "data/pipeline/point_clouds"))
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "lignos/data/cosmo_images"))
    parser.add_argument("--n_views", type=int, default=36)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--render_ep", action="store_true", default=True)
    args = parser.parse_args()

    pc_dir = Path(args.point_cloud_dir)
    out_root = Path(args.output_dir)

    # Determine compound list
    if args.compound_id:
        compounds = [args.compound_id]
    elif args.compound_list:
        with open(args.compound_list) as f:
            compounds = [line.strip() for line in f if line.strip()]
    else:
        # Process all available point clouds
        compounds = [p.stem for p in pc_dir.glob("*.npz")]

    print(f"Rendering {len(compounds)} compounds, {args.n_views} views each")

    for i, cid in enumerate(compounds):
        pc_path = pc_dir / f"{cid}.npz"
        if not pc_path.exists():
            print(f"  [{i+1}/{len(compounds)}] SKIP {cid}: no point cloud")
            continue

        frame_dir = out_root / f"{cid}_frames"
        if frame_dir.exists() and len(list(frame_dir.glob("*.png"))) >= args.n_views:
            print(f"  [{i+1}/{len(compounds)}] SKIP {cid}: already rendered")
            continue

        print(f"  [{i+1}/{len(compounds)}] Rendering {cid}...")
        render_cosmo_views(
            pc_path, frame_dir,
            n_views=args.n_views,
            resolution=args.resolution,
            render_ep=args.render_ep,
        )

    print("Done!")


if __name__ == "__main__":
    main()
