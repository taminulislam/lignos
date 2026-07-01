#!/usr/bin/env python3
"""V-JEPA self-supervised pre-training for COSMO surface rotation videos.

Alternative to SimCLR that learns by predicting masked rotation views in latent
space. Key advantage: no negative pairs needed, learns 3D geometry from masking.

Usage:
    python pretrain_vjepa.py --epochs 200 --batch_size 32 --device cuda
    python pretrain_vjepa.py --config configs/vjepa.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5_ROOT))

from models.multiview_vit import PatchEmbedding, ViTBlock
from models.vjepa import COSMO_VJEPA, COSMOViewDataset


class ViTTinyEncoder(nn.Module):
    """ViT-Tiny encoder for single-view encoding."""

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


def collect_frame_dirs():
    """Collect all frame directories from v5 and original locations."""
    dirs = [
        V5_ROOT / "data/cosmo_images",
        PROJECT_ROOT / "data/pipeline/cosmo_images",
    ]

    mol_to_dir = {}
    for d in dirs:
        if not d.exists():
            continue
        for frame_dir in d.iterdir():
            if frame_dir.is_dir() and frame_dir.name.endswith("_frames"):
                n = len(list(frame_dir.glob("frame_*.png")))
                if n >= 4:  # Need at least 4 views for meaningful masking
                    mid = frame_dir.name.replace("_frames", "")
                    if mid not in mol_to_dir:
                        mol_to_dir[mid] = frame_dir

    return mol_to_dir


def train_one_epoch(model, dataloader, optimizer, epoch, device,
                    ema_schedule=None):
    """Train V-JEPA for one epoch."""
    model.train()
    total_loss = 0
    total_sim = 0
    n_batches = 0

    for batch_idx, (views, mol_ids) in enumerate(dataloader):
        views = views.to(device)

        loss, aux = model(views)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update target encoder EMA
        if ema_schedule is not None:
            model.ema_decay = ema_schedule(epoch, batch_idx)
        model.update_target_encoder()

        total_loss += aux["loss"]
        total_sim += aux["cosine_sim"]
        n_batches += 1

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"loss={aux['loss']:.4f} cos_sim={aux['cosine_sim']:.4f} "
                  f"mask={aux['mask_ratio']:.1%}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "cosine_sim": total_sim / max(n_batches, 1),
    }


def cosine_ema_schedule(base_decay=0.996, max_decay=0.9999, n_epochs=200):
    """Cosine schedule for EMA decay: increases from base to max over training."""
    def schedule(epoch, batch_idx=0):
        progress = epoch / n_epochs
        return max_decay - (max_decay - base_decay) * (1 + math.cos(math.pi * progress)) / 2
    import math
    return schedule


def main():
    parser = argparse.ArgumentParser(description="V-JEPA pre-training")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--n_views", type=int, default=36)
    parser.add_argument("--mask_ratio_min", type=float, default=0.6)
    parser.add_argument("--mask_ratio_max", type=float, default=0.8)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--predictor_dim", type=int, default=96)
    parser.add_argument("--ema_decay", type=float, default=0.996)
    parser.add_argument("--output_dir", type=str,
                        default=str(V5_ROOT / "checkpoints/vjepa"))
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("V-JEPA Pre-training for COSMO Surfaces")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"  Mask ratio: {args.mask_ratio_min:.0%}-{args.mask_ratio_max:.0%}")
    print(f"  EMA decay: {args.ema_decay}")

    # Collect data
    mol_to_dir = collect_frame_dirs()
    mol_ids = sorted(mol_to_dir.keys())
    print(f"  Found {len(mol_ids)} molecules")

    if not mol_ids:
        print("ERROR: No molecules found.")
        return

    # Dataset
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = COSMOViewDataset(mol_ids, mol_to_dir, transform, n_views=args.n_views)
    dataloader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Build model
    encoder = ViTTinyEncoder(embed_dim=args.embed_dim)
    model = COSMO_VJEPA(
        encoder,
        embed_dim=args.embed_dim,
        predictor_dim=args.predictor_dim,
        n_views=args.n_views,
        mask_ratio=(args.mask_ratio_min, args.mask_ratio_max),
        ema_decay=args.ema_decay,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,} (trainable: {trainable:,})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # EMA schedule (linearly increase decay)
    ema_sched = cosine_ema_schedule(args.ema_decay, 0.9999, args.epochs)

    # Training
    best_loss = float("inf")
    history = []
    start_epoch = 1

    # Auto-resume from last.pt if present. Triggered by SLURM --requeue so a
    # preempted run continues from the most recent snapshot instead of epoch 0.
    last_path = output_dir / "last.pt"
    if last_path.exists():
        print(f"  Resuming from {last_path}")
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        history = ckpt.get("history", [])
        torch.set_rng_state(ckpt["rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in ckpt:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
        print(f"  Resumed at epoch {start_epoch}, best_loss so far: {best_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        metrics = train_one_epoch(model, dataloader, optimizer, epoch, device,
                                   ema_sched)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({"epoch": epoch, **metrics, "time": elapsed})
        print(f"Epoch {epoch}/{args.epochs}: loss={metrics['loss']:.4f} "
              f"cos_sim={metrics['cosine_sim']:.4f} time={elapsed:.1f}s")

        # Save checkpoints
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.context_encoder.state_dict(),
                "target_encoder_state_dict": model.target_encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }
            torch.save(checkpoint, output_dir / "vjepa_pretrained.pt")
            # Also save encoder-only for loading into COSMOBridgeV5
            torch.save(
                {"encoder_state_dict": model.context_encoder.state_dict()},
                output_dir / "vit_pretrained_vjepa.pt",
            )
            print(f"  Saved best (loss={best_loss:.4f})")

        if epoch % 50 == 0:
            torch.save(checkpoint, output_dir / f"checkpoint_epoch{epoch}.pt")

        # Resume snapshot: written every 5 epochs so a preempted job restarts
        # from at most ~5 epochs back (~1–2 min of lost work).
        if epoch % 5 == 0 or epoch == args.epochs:
            last_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "history": history,
                "rng_state": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                last_ckpt["cuda_rng_state"] = torch.cuda.get_rng_state()
            tmp = output_dir / "last.pt.tmp"
            torch.save(last_ckpt, tmp)
            tmp.replace(output_dir / "last.pt")

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone! Best loss: {best_loss:.4f}")
    print(f"Encoder saved to: {output_dir / 'vit_pretrained_vjepa.pt'}")
    print(f"(Compatible with COSMOBridgeV5.load_simclr_weights())")


if __name__ == "__main__":
    main()
