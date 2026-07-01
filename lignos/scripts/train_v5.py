#!/usr/bin/env python3
"""3-Stage supervised training for COSMOBridge v5.

Stage 1: Freeze all encoders, train fusion + prediction heads
Stage 2: Unfreeze image encoders (ViT + Siamese), differential LR
Stage 3: Full fine-tuning with low LR (optional, if R² < target)

Usage:
    python train_v5.py --config configs/v5_full.yaml --seed 0
    python train_v5.py --config configs/v5_full.yaml --seeds 0-9
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


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(preds, targets):
    """Compute R², MAE, RMSE per property.

    Args:
        preds: (N, P) predictions
        targets: (N, P) ground truth

    Returns:
        dict with per-property and average metrics
    """
    property_names = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]
    metrics = {}

    for i, name in enumerate(property_names):
        p = preds[:, i]
        t = targets[:, i]

        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        metrics[f"{name}_r2"] = r2.item()
        metrics[f"{name}_mae"] = (t - p).abs().mean().item()
        metrics[f"{name}_rmse"] = ((t - p) ** 2).mean().sqrt().item()

    metrics["avg_r2"] = np.mean([metrics[f"{n}_r2"] for n in property_names])
    metrics["avg_mae"] = np.mean([metrics[f"{n}_mae"] for n in property_names])

    return metrics


def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    anchor_loss_lambda=0.0, anchor_weights=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch in dataloader:
        # Unpack batch -- adapt to your DataLoader
        views = batch["views"].to(device)
        cation_img = batch["cation_img"].to(device)
        anion_img = batch["anion_img"].to(device)
        graph_feat = batch["graph_feat"].to(device)
        surface_feat = batch["surface_feat"].to(device)
        thermo_feat = batch["thermo_feat"].to(device)
        targets = batch["targets"].to(device)

        preds, aux = model(
            views, cation_img, anion_img,
            graph_feat, surface_feat, thermo_feat,
        )

        loss = criterion(preds, targets)

        # Anchor loss on routing weights (prevent collapse)
        if anchor_loss_lambda > 0 and anchor_weights is not None:
            routing = model.fusion.routing_logits
            anchor_loss = anchor_loss_lambda * nn.functional.mse_loss(
                routing, anchor_weights.to(routing.device)
            )
            loss = loss + anchor_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * targets.shape[0]
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(all_preds)

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch in dataloader:
        views = batch["views"].to(device)
        cation_img = batch["cation_img"].to(device)
        anion_img = batch["anion_img"].to(device)
        graph_feat = batch["graph_feat"].to(device)
        surface_feat = batch["surface_feat"].to(device)
        thermo_feat = batch["thermo_feat"].to(device)
        targets = batch["targets"].to(device)

        preds, aux = model(
            views, cation_img, anion_img,
            graph_feat, surface_feat, thermo_feat,
            use_chunked_views=True,  # save memory during eval
        )

        loss = criterion(preds, targets)
        total_loss += loss.item() * targets.shape[0]
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(all_preds)

    return metrics, all_preds, all_targets


def train_stage(model, train_loader, val_loader, optimizer, scheduler,
                criterion, device, stage_name, epochs, patience=20,
                anchor_loss_lambda=0.0, anchor_weights=None,
                checkpoint_dir=None):
    """Run one training stage with early stopping."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {stage_name}")
    print(f"  Epochs: {epochs}, Patience: {patience}")
    print(f"{'='*60}")

    best_val_r2 = -float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            anchor_loss_lambda, anchor_weights,
        )

        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "time": elapsed,
        })

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train R²={train_metrics['avg_r2']:.4f} loss={train_metrics['loss']:.4f} | "
            f"val R²={val_metrics['avg_r2']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping on validation R²
        if val_metrics["avg_r2"] > best_val_r2:
            best_val_r2 = val_metrics["avg_r2"]
            patience_counter = 0
            if checkpoint_dir:
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (best val R²={best_val_r2:.4f})")
                break

    # Restore best model
    if checkpoint_dir and (checkpoint_dir / "best.pt").exists():
        model.load_state_dict(torch.load(checkpoint_dir / "best.pt", weights_only=True))

    return history, best_val_r2


def train_single_seed(config, seed, device):
    """Train one seed through all 3 stages.

    Args:
        config: training configuration dict
        seed: random seed
        device: torch device

    Returns:
        test_metrics: dict with test set metrics
        test_preds: (N, P) test predictions
    """
    from models.cosmobridge_v5 import COSMOBridgeV5

    set_seed(seed)
    print(f"\n{'#'*60}")
    print(f"  SEED {seed}")
    print(f"{'#'*60}")

    # Build model
    mc = config.get("model", {})
    model = COSMOBridgeV5(
        embed_dim=mc.get("embed_dim", 256),
        n_properties=mc.get("n_properties", 7),
        n_views=mc.get("multiview_vit", {}).get("n_views", 36),
        graph_dim=mc.get("graph", {}).get("dim", 300),
        surface_dim=mc.get("pointcloud", {}).get("dim", 256),
        thermo_dim=mc.get("tabular", {}).get("dim", 25),
        dropout=mc.get("dropout", 0.2),
    ).to(device)

    # Load SimCLR / V-JEPA pre-trained weights
    simclr_path = mc.get("multiview_vit", {}).get("pretrained_checkpoint")
    if simclr_path:
        p = Path(simclr_path)
        if not p.is_absolute():
            p = V5_ROOT / p
        if p.exists():
            print(f"  Loading pretrained ViT weights from {p}")
            model.load_simclr_weights(str(p))
        else:
            print(f"  WARNING: pretrained_checkpoint not found at {p} — ViT will be randomly initialized")

    # Initialize routing with domain knowledge
    routing_init = config.get("model", {}).get("routing_init")
    if routing_init:
        model.fusion.init_routing_from_domain_knowledge(routing_init)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Build dataloaders
    from data.dataset import build_dataloader
    train_loader = build_dataloader("train", config, seed)
    val_loader = build_dataloader("val", config, seed)
    test_loader = build_dataloader("test", config, seed)

    tc = config.get("training", {})
    criterion = nn.MSELoss()

    checkpoint_dir = Path(tc.get("checkpoint_dir",
                                  str(V5_ROOT / "checkpoints/supervised")))
    seed_dir = checkpoint_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Save anchor weights for routing regularization
    anchor_weights = model.fusion.routing_logits.detach().clone()

    # ── Stage 1: Freeze encoders, train fusion + heads ──
    s1 = tc.get("stage1", {})
    model.freeze_encoders()
    optimizer1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=s1.get("lr", 1e-3),
        weight_decay=tc.get("weight_decay", 1e-3),
    )
    history1, val_r2_1 = train_stage(
        model, train_loader, val_loader, optimizer1, None,
        criterion, device, "Stage 1: Frozen Encoders",
        epochs=s1.get("epochs", 5),
        patience=s1.get("epochs", 5),
        anchor_loss_lambda=tc.get("anchor_loss_lambda", 0.05),
        anchor_weights=anchor_weights,
        checkpoint_dir=seed_dir,
    )

    # ── Stage 2: Unfreeze image encoders with differential LR ──
    s2 = tc.get("stage2", {})
    model.unfreeze_image_encoders()
    param_groups = model.get_parameter_groups(
        image_lr=s2.get("image_lr", 1e-4),
        fusion_lr=s2.get("fusion_lr", 1e-3),
        head_lr=s2.get("fusion_lr", 1e-3),
    )
    optimizer2 = torch.optim.AdamW(
        param_groups,
        weight_decay=tc.get("weight_decay", 1e-3),
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=s2.get("epochs", 50),
    )
    history2, val_r2_2 = train_stage(
        model, train_loader, val_loader, optimizer2, scheduler2,
        criterion, device, "Stage 2: Image Encoders Unfrozen",
        epochs=s2.get("epochs", 50),
        patience=s2.get("early_stopping_patience", 20),
        anchor_loss_lambda=tc.get("anchor_loss_lambda", 0.05) * 0.5,
        anchor_weights=anchor_weights,
        checkpoint_dir=seed_dir,
    )

    # ── Stage 3: Full fine-tuning (conditional) ──
    s3 = tc.get("stage3", {})
    r2_threshold = s3.get("only_if_r2_below", 0.90)
    if val_r2_2 < r2_threshold:
        for p in model.parameters():
            p.requires_grad = True
        optimizer3 = torch.optim.AdamW(
            model.parameters(),
            lr=s3.get("lr", 1e-5),
            weight_decay=tc.get("weight_decay", 1e-3),
        )
        history3, val_r2_3 = train_stage(
            model, train_loader, val_loader, optimizer3, None,
            criterion, device, "Stage 3: Full Fine-tuning",
            epochs=s3.get("epochs", 30),
            patience=15,
            checkpoint_dir=seed_dir,
        )
    else:
        print(f"\n  Skipping Stage 3: val R²={val_r2_2:.4f} >= {r2_threshold}")

    # ── Evaluate on test set ──
    test_metrics, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device,
    )

    print(f"\n  Test Results (seed {seed}):")
    for prop in ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]:
        print(f"    {prop:8s}: R² = {test_metrics[f'{prop}_r2']:.4f}")
    print(f"    {'avg':8s}: R² = {test_metrics['avg_r2']:.4f}")

    # Save results
    torch.save(model.state_dict(), seed_dir / "final.pt")
    with open(seed_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save predictions for ensemble evaluation
    results_base = config.get("output", {}).get("results_dir", "results")
    pred_dir = V5_ROOT / results_base / "seed_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        pred_dir / f"seed_{seed}.npz",
        predictions=test_preds.numpy(),
        targets=test_targets.numpy(),
    )

    print(f"\n  Seed {seed} complete. Avg R² = {test_metrics['avg_r2']:.4f}")
    return test_metrics, test_preds


def main():
    parser = argparse.ArgumentParser(description="Train COSMOBridge v5")
    parser.add_argument("--config", type=str, default="configs/v5_full.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Single seed")
    parser.add_argument("--seeds", type=str, default=None, help="Seed range, e.g., 0-9")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load config
    config_path = V5_ROOT / args.config
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config not found: {config_path}")
        config = {}

    device = torch.device(args.device)

    # Determine seeds
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        start, end = map(int, args.seeds.split("-"))
        seeds = list(range(start, end + 1))
    else:
        seeds = config.get("training", {}).get("seeds", [0])

    print(f"COSMOBridge v5 Training")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")

    all_metrics = {}
    for seed in seeds:
        metrics, preds = train_single_seed(config, seed, device)
        all_metrics[seed] = metrics

    # Summary
    if all_metrics and any(all_metrics.values()):
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        r2_values = [m.get("avg_r2", 0) for m in all_metrics.values() if m]
        if r2_values:
            print(f"  avg R²: {np.mean(r2_values):.4f} +/- {np.std(r2_values):.4f}")

    # Save aggregate results
    results_dir = V5_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "training_summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)


if __name__ == "__main__":
    main()
