#!/usr/bin/env python3
"""Train COSMOBridge v4 (existing architecture) with merged iThermo data.

This establishes the baseline: how much does data merging alone improve v4?
Uses the existing v4 model (500K params) with the expanded iThermo dataset.
No image modalities -- just graph + surface + tabular with masked multi-task loss.

Usage:
    python train_v4_merged.py --seeds 0-9 --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5_ROOT))

from data.merged_dataset import MergedMultiTaskDataset, masked_mse_loss


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class V4MergedModel(nn.Module):
    """v4-style model: Graph + Surface + Tabular with per-property routing.

    Same architecture as COSMOBridge v4 but trained on merged data.
    No image modalities -- uses only the frozen Chemprop and PointNet features.

    Parameters
    ----------
    graph_dim : int
        Chemprop fingerprint dimension (300D).
    surface_dim : int
        PointNet surface feature dimension (256D).
    thermo_dim : int
        Thermodynamic feature dimension (25D).
    fused_dim : int
        Internal fused dimension.
    n_properties : int
        Number of targets.
    dropout : float
        Dropout rate.
    """

    def __init__(self, graph_dim=300, surface_dim=256, thermo_dim=25,
                 fused_dim=256, n_properties=7, dropout=0.3):
        super().__init__()

        # Path A: Bilinear fusion (graph x surface)
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.surface_proj = nn.Linear(surface_dim, fused_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Combine with thermo
        self.fused_head = nn.Sequential(
            nn.Linear(fused_dim + thermo_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_properties),
        )

        # Path B: Direct (graph + thermo only)
        self.direct_head = nn.Sequential(
            nn.Linear(graph_dim + thermo_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_properties),
        )

        # Per-property gate (v4 style)
        self.gate_logits = nn.Parameter(torch.tensor([
            2.0, 2.0, -2.0, -2.0, -2.0, 0.0, 1.5  # same as v4 init
        ]))

    def forward(self, graph_feat, surface_feat, thermo_feat):
        # Path A: Fusion
        g_proj = self.graph_proj(graph_feat)
        s_proj = self.surface_proj(surface_feat)
        fused = self.fusion_mlp(g_proj * s_proj)  # element-wise product (simplified bilinear)
        fused_with_thermo = torch.cat([fused, thermo_feat], dim=-1)
        preds_fused = self.fused_head(fused_with_thermo)

        # Path B: Direct
        direct_input = torch.cat([graph_feat, thermo_feat], dim=-1)
        preds_direct = self.direct_head(direct_input)

        # Per-property gate
        alpha = torch.sigmoid(self.gate_logits)
        predictions = alpha.unsqueeze(0) * preds_fused + (1 - alpha.unsqueeze(0)) * preds_direct

        return predictions


def compute_metrics(preds, targets, masks=None):
    props = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
    metrics = {}
    for i, name in enumerate(props):
        if masks is not None:
            m = masks[:, i].bool()
            if m.sum() < 2:
                metrics[f"{name}_r2"] = float("nan")
                continue
            p, t = preds[m, i], targets[m, i]
        else:
            p, t = preds[:, i], targets[:, i]
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        metrics[f"{name}_r2"] = (1 - ss_res / (ss_tot + 1e-8)).item()

    valid = [v for v in metrics.values() if not np.isnan(v)]
    metrics["avg_r2"] = np.mean(valid) if valid else 0.0
    return metrics


def train_single_seed(seed, device):
    set_seed(seed)
    print(f"\n{'#'*60}")
    print(f"  SEED {seed} - v4 + Merged iThermo Data")
    print(f"{'#'*60}")

    # Build merged training dataset
    train_ds = MergedMultiTaskDataset(
        original_cached_path=str(PROJECT_ROOT / "cosmobridge_v4/data/cached_train.npz"),
        ilthermo_csv_path=str(PROJECT_ROOT / "data/augmented/ilthermo_data.csv"),
        cosmo_images_dir="",
        ion_images_dir="",
        orig_cosmo_dir="",
        split="train",
        include_ilthermo=True,
        n_views=1,
        image_size=1,
    )

    val_ds = MergedMultiTaskDataset(
        original_cached_path=str(PROJECT_ROOT / "cosmobridge_v4/data/cached_val.npz"),
        split="val",
        include_ilthermo=False,
        cosmo_images_dir="",
        ion_images_dir="",
        orig_cosmo_dir="",
        n_views=1,
        image_size=1,
    )

    test_ds = MergedMultiTaskDataset(
        original_cached_path=str(PROJECT_ROOT / "cosmobridge_v4/data/cached_test.npz"),
        split="test",
        include_ilthermo=False,
        cosmo_images_dir="",
        ion_images_dir="",
        orig_cosmo_dir="",
        n_views=1,
        image_size=1,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=0,
    )

    # Build v4-style model
    model = V4MergedModel(
        graph_dim=300, surface_dim=256, thermo_dim=25,
        fused_dim=256, n_properties=7, dropout=0.3,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  v4 model: {params:,} params")
    print(f"  Params/sample ratio: {params / len(train_ds):.0f}:1")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    ckpt_dir = V5_ROOT / "checkpoints/v4_merged" / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_r2 = -float("inf")
    patience_counter = 0

    for epoch in range(1, 101):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            graph_f = batch["graph_feat"].to(device)
            surface_f = batch["surface_feat"].to(device)
            thermo_f = batch["thermo_feat"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            preds = model(graph_f, surface_f, thermo_f)
            loss = masked_mse_loss(preds, targets, mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                preds = model(
                    batch["graph_feat"].to(device),
                    batch["surface_feat"].to(device),
                    batch["thermo_feat"].to(device),
                )
                val_preds.append(preds.cpu())
                val_targets.append(batch["targets"])

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_metrics = compute_metrics(val_preds, val_targets)
        elapsed = time.time() - t0

        if epoch % 10 == 1 or epoch <= 5:
            print(f"  Epoch {epoch:3d}/100 | loss={total_loss/n_batches:.4f} | "
                  f"val R²={val_metrics['avg_r2']:.4f} | {elapsed:.1f}s")

        if val_metrics["avg_r2"] > best_val_r2:
            best_val_r2 = val_metrics["avg_r2"]
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= 25:
                print(f"  Early stopping at epoch {epoch} (best val R²={best_val_r2:.4f})")
                break

    # Test
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", weights_only=True))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            preds = model(
                batch["graph_feat"].to(device),
                batch["surface_feat"].to(device),
                batch["thermo_feat"].to(device),
            )
            test_preds.append(preds.cpu())
            test_targets.append(batch["targets"])

    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)
    test_metrics = compute_metrics(test_preds, test_targets)

    print(f"\n  Test Results (seed {seed}):")
    for prop in ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]:
        print(f"    {prop:8s}: R² = {test_metrics[f'{prop}_r2']:.4f}")
    print(f"    {'avg':8s}: R² = {test_metrics['avg_r2']:.4f}")

    # Save
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    pred_dir = V5_ROOT / "results/v4_merged/seed_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        pred_dir / f"seed_{seed}.npz",
        predictions=test_preds.numpy(),
        targets=test_targets.numpy(),
    )

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train v4 with merged data")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        start, end = map(int, args.seeds.split("-"))
        seeds = list(range(start, end + 1))
    else:
        seeds = list(range(10))

    print(f"v4 + Merged iThermo Data")
    print(f"  Purpose: Measure improvement from data merging alone (no images)")
    print(f"  Seeds: {seeds}, Device: {device}")

    all_metrics = {}
    for seed in seeds:
        metrics = train_single_seed(seed, device)
        all_metrics[seed] = metrics

    if all_metrics:
        print(f"\n{'='*60}")
        print("v4 + MERGED DATA SUMMARY")
        print(f"{'='*60}")
        r2_vals = [m["avg_r2"] for m in all_metrics.values()]
        print(f"  avg R²: {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}")
        print(f"  (v4 original: 0.810, v5+SimCLR: 0.657)")

    results_dir = V5_ROOT / "results/v4_merged"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "training_summary.json", "w") as f:
        json.dump({str(k): v for k, v in all_metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()
