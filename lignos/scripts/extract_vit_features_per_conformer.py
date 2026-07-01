#!/usr/bin/env python3
"""Stage 2b — Re-extract V-JEPA ViT features for one conformer's cosmo views.

Runs the V-JEPA ViT (same checkpoint used for the 0.83 baseline's `vit_feat`)
over a conformer-specific cosmo_images directory and saves
`cached_image_features_{split}_conf_{k}.npz` with the same `vit_feat` key
as the baseline file, so downstream code (ensemble eval, PerPropHead, PCA)
can treat it as a drop-in replacement.

Inputs:
    --cosmo_dir   path to a conformer's rotation frames
                  (directory containing {hash}_frames/frame_*.png)
    --conf_id     conformer id (for output filename)

Assumes a pre-existing V-JEPA checkpoint at
    lignos/checkpoints/vjepa/vit_pretrained_vjepa.pt
If a sigma-pretrained checkpoint from Stage 3 exists it will be preferred
when --use_sigma_pretrained is passed.

Usage:
    python extract_vit_features_per_conformer.py \
        --conf_id 1 \
        --cosmo_dir lignos/data/cosmo_images_multi/conf_1

    # All splits, loop over 5 conformers, using sigma-pretrained ViT:
    for k in 0 1 2 3 4; do
        python extract_vit_features_per_conformer.py \
            --conf_id $k \
            --cosmo_dir lignos/data/cosmo_images_multi/conf_$k \
            --use_sigma_pretrained
    done
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5))

from data.dataset import COSMOBridgeV5Dataset  # noqa: E402
from models.multiview_vit import MultiViewViT  # noqa: E402


def load_vit(checkpoint_path, device):
    vit = MultiViewViT(n_views=36, embed_dim=192, dropout=0.0)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder_state = state.get("encoder_state_dict", state)
    missing, _ = vit.load_state_dict(encoder_state, strict=False)
    print(f"  Loaded ViT from {Path(checkpoint_path).name} ({len(missing)} missing keys)")
    return vit.to(device).eval()


def extract_split(vit, split, cosmo_dir, device):
    # `cosmo_dir` here is a per-conformer directory containing
    # {pc_hash}_frames/ subdirs (matches the single-conformer v5 pipeline
    # naming that render_cosmo_views.py produces from hash-named point-cloud
    # .npz files). Pass it as `cosmo_images_dir` so _find_frames_dir resolves
    # via pc_hash lookup. orig_cosmo_dir falls back to the shared project
    # directory in case any sample is missing a per-conformer render.
    ds = COSMOBridgeV5Dataset(
        cached_features_path=str(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz"),
        cosmo_images_dir=str(cosmo_dir),
        ion_images_dir=str(V5 / "data/ion_images"),
        orig_cosmo_dir=str(PROJECT_ROOT / "data/pipeline/cosmo_images"),
        master_index_path=str(V5 / "data/master_index.csv"),
        n_views=36,
        image_size=224,
        view_sample_mode="all",
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    feats = []
    with torch.no_grad():
        for batch in loader:
            views = batch["views"].to(device)
            emb, _ = vit.encode_views_chunked(views, chunk_size=3)
            feats.append(emb.cpu())
    return torch.cat(feats).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf_id", type=int, required=True)
    ap.add_argument("--cosmo_dir", type=str, required=True)
    ap.add_argument("--use_sigma_pretrained", action="store_true")
    ap.add_argument("--output_dir", type=str, default=str(V5 / "data"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.use_sigma_pretrained:
        ckpt = V5 / "checkpoints/sigma_ssl/vit_pretrained_sigma.pt"
        if not ckpt.exists():
            print(f"WARNING: sigma-pretrained ckpt not found at {ckpt}; "
                  f"falling back to V-JEPA")
            ckpt = V5 / "checkpoints/vjepa/vit_pretrained_vjepa.pt"
    else:
        ckpt = V5 / "checkpoints/vjepa/vit_pretrained_vjepa.pt"

    vit = load_vit(ckpt, device)

    for split in ["train", "val", "test"]:
        print(f"Extracting {split} features from conformer {args.conf_id}...")
        feats = extract_split(vit, split, args.cosmo_dir, device)
        out = Path(args.output_dir) / f"cached_image_features_{split}_conf_{args.conf_id}.npz"
        np.savez(out, vit_feat=feats)
        print(f"  Saved {out} shape={feats.shape}")


if __name__ == "__main__":
    main()
