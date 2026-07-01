#!/usr/bin/env python3
"""SimCLR self-supervised pre-training for ViT-Tiny on COSMO surfaces.

Pre-trains the ViT-Tiny encoder using contrastive learning on all available
COSMO surface images. Uses natural rotation views as positive pairs.

Usage:
    python pretrain_simclr.py --config configs/simclr.yaml
    python pretrain_simclr.py --epochs 200 --batch_size 256 --gpus 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5_ROOT))

from models.multiview_vit import MultiViewViT
from models.simclr import SimCLR


class MultiDirViewPairDataset(torch.utils.data.Dataset):
    """Dataset yielding random view pairs, supporting multiple frame directories.

    Args:
        mol_ids: list of molecule identifiers
        mol_to_dir: dict mapping mol_id -> Path to its *_frames/ directory
        transform: torchvision transform for images
        n_views: number of rotation views per molecule
    """

    def __init__(self, mol_ids, mol_to_dir, transform=None, n_views=36):
        self.mol_ids = mol_ids
        self.mol_to_dir = mol_to_dir
        self.transform = transform
        self.n_views = n_views

    def __len__(self):
        return len(self.mol_ids)

    def __getitem__(self, idx):
        from PIL import Image
        import random

        mol_id = self.mol_ids[idx]
        frame_dir = self.mol_to_dir[mol_id]

        frames = sorted(frame_dir.glob("frame_*.png"))
        n = min(len(frames), self.n_views)
        if n < 2:
            # Fallback: use whatever is available
            i, j = 0, 0
        else:
            i, j = random.sample(range(n), 2)

        img1 = Image.open(frames[i]).convert("RGB")
        img2 = Image.open(frames[j]).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, mol_id


def get_augmentation_transforms(config):
    """Build augmentation pipeline from config."""
    from torchvision import transforms

    aug_cfg = config.get("augmentations", {})

    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            config["data"]["image_size"],
            scale=tuple(aug_cfg.get("random_resized_crop", {}).get("scale", [0.6, 1.0])),
        ),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=aug_cfg.get("color_jitter", {}).get("brightness", 0.3),
                contrast=aug_cfg.get("color_jitter", {}).get("contrast", 0.3),
                saturation=aug_cfg.get("color_jitter", {}).get("saturation", 0.2),
                hue=aug_cfg.get("color_jitter", {}).get("hue", 0.1),
            )
        ], p=aug_cfg.get("color_jitter", {}).get("probability", 0.8)),
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=aug_cfg.get("gaussian_blur", {}).get("kernel_size", 23),
            )
        ], p=aug_cfg.get("gaussian_blur", {}).get("probability", 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return transform


def build_encoder(config):
    """Build the ViT-Tiny encoder from config."""
    mc = config.get("model", {})

    # Build a single-view encoder (not MultiViewViT)
    # SimCLR operates on individual views
    from models.multiview_vit import PatchEmbedding, ViTBlock

    class ViTTinyEncoder(nn.Module):
        def __init__(self, embed_dim=192, img_size=224, patch_size=16,
                     n_layers=6, n_heads=3, mlp_ratio=4, dropout=0.1,
                     stochastic_depth=0.1):
            super().__init__()
            self.embed_dim = embed_dim
            self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
            n_patches = self.patch_embed.n_patches

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
            self.pos_dropout = nn.Dropout(dropout)

            dpr = [x.item() for x in torch.linspace(0, stochastic_depth, n_layers)]
            self.blocks = nn.ModuleList([
                ViTBlock(embed_dim, n_heads, mlp_ratio, dropout, dpr[i])
                for i in range(n_layers)
            ])
            self.norm = nn.LayerNorm(embed_dim)

            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        def forward(self, x):
            patches = self.patch_embed(x)
            B = patches.shape[0]
            cls = self.cls_token.expand(B, -1, -1)
            tokens = self.pos_dropout(
                torch.cat([cls, patches], dim=1) + self.pos_embed
            )
            for block in self.blocks:
                tokens = block(tokens)
            return self.norm(tokens[:, 0])

    encoder = ViTTinyEncoder(
        embed_dim=mc.get("embed_dim", 192),
        patch_size=mc.get("patch_size", 16),
        n_layers=mc.get("num_layers", 6),
        n_heads=mc.get("num_heads", 3),
        mlp_ratio=mc.get("mlp_ratio", 4),
        stochastic_depth=mc.get("stochastic_depth", 0.1),
    )

    return encoder


def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, (view1, view2, _) in enumerate(dataloader):
        view1 = view1.to(device)
        view2 = view2.to(device)

        loss, aux = model(view1, view2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"loss={loss.item():.4f}")

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="SimCLR pre-training")
    parser.add_argument("--config", type=str, default="configs/simclr.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--extra_image_dirs", type=str, nargs="*", default=None,
                        help="Additional directories with *_frames/ subdirs")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load config
    config_path = V5_ROOT / args.config
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {"data": {}, "model": {}, "training": {}, "output": {}}

    # Override with CLI args
    tc = config.setdefault("training", {})
    if args.epochs:
        tc["epochs"] = args.epochs
    if args.batch_size:
        tc["batch_size"] = args.batch_size
    if args.lr:
        tc["base_lr"] = args.lr

    dc = config.setdefault("data", {})
    image_dir = args.image_dir or dc.get(
        "image_dir", str(PROJECT_ROOT / "data/pipeline/cosmo_images")
    )
    output_dir = args.output_dir or config.get("output", {}).get(
        "checkpoint_dir", str(V5_ROOT / "checkpoints/simclr")
    )

    epochs = tc.get("epochs", 200)
    batch_size = tc.get("batch_size", 256)
    lr = tc.get("base_lr", 0.3)
    temperature = tc.get("temperature", 0.07)

    device = torch.device(args.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"SimCLR Pre-training")
    print(f"  Image dir: {image_dir}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")

    # Discover molecules with rendered frames from all directories
    image_dir = Path(image_dir)
    all_dirs = [image_dir]

    # Add extra directories (e.g., original v3/v4 cosmo_images)
    extra = args.extra_image_dirs or dc.get("extra_image_dirs", [])
    if not extra:
        # Default: also scan original cosmo_images
        orig = PROJECT_ROOT / "data/pipeline/cosmo_images"
        if orig.exists() and orig != image_dir:
            extra = [str(orig)]
    for d in extra:
        p = Path(d)
        if p.exists():
            all_dirs.append(p)
            print(f"  Extra image dir: {p}")

    # Collect all molecule IDs and their frame directories
    mol_to_dir = {}
    for d in all_dirs:
        for frame_dir in d.iterdir():
            if frame_dir.is_dir() and frame_dir.name.endswith("_frames"):
                # Only include if directory has at least 2 frames
                n_frames = len(list(frame_dir.glob("frame_*.png")))
                if n_frames >= 2:
                    mid = frame_dir.name.replace("_frames", "")
                    if mid not in mol_to_dir:
                        mol_to_dir[mid] = frame_dir

    mol_ids = sorted(mol_to_dir.keys())
    print(f"  Found {len(mol_ids)} molecules with frame directories")

    if len(mol_ids) == 0:
        print("ERROR: No molecules found. Run render_cosmo_views.py first.")
        return

    # Build dataset that resolves per-molecule frame directories
    transform = get_augmentation_transforms(config)
    dataset = MultiDirViewPairDataset(mol_ids, mol_to_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=dc.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # Build model
    encoder = build_encoder(config)
    model = SimCLR(
        encoder,
        encoder_dim=config.get("model", {}).get("embed_dim", 192),
        projection_dim=config.get("model", {}).get("projection_dim", 128),
        projection_hidden=config.get("model", {}).get("projection_hidden", 256),
        temperature=temperature,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=tc.get("weight_decay", 1e-6),
    )

    warmup = tc.get("warmup_epochs", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup, eta_min=1e-6
    )

    # Training loop
    best_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, scheduler, epoch, device
        )
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "loss": avg_loss, "time": elapsed})
        print(f"Epoch {epoch}/{epochs}: loss={avg_loss:.4f}, time={elapsed:.1f}s")

        # Save checkpoints
        save_every = config.get("output", {}).get("save_every", 20)
        if epoch % save_every == 0 or avg_loss < best_loss:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config,
            }

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, output_dir / "vit_pretrained.pt")
                print(f"  Saved best model (loss={best_loss:.4f})")

            if epoch % save_every == 0:
                torch.save(checkpoint, output_dir / f"checkpoint_epoch{epoch}.pt")

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone! Best loss: {best_loss:.4f}")
    print(f"Pre-trained ViT saved to: {output_dir / 'vit_pretrained.pt'}")


if __name__ == "__main__":
    main()
