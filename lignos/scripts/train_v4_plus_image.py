#!/usr/bin/env python3
"""Train COSMOBridge v4 + Image: exact v4 protocol with images as 3rd path.

Step 1: Pre-train ImageHead on frozen ViT features → preds_image
Step 2: Freeze ImageHead, train 3-way router (identical to v4 router training)

Uses the EXACT same cached data, config, and training protocol as v4.

Usage:
    python train_v4_plus_image.py --seeds 0-9 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5_ROOT))

from models.cosmobridge_v4_plus_image import (
    COSMOBridgeV4PlusImage, ImageHead,
)

PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


def load_cached(split):
    d = np.load(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz",
                allow_pickle=True)
    return {k: torch.from_numpy(d[k]).float() if d[k].dtype.kind == 'f'
            else d[k] for k in d.keys()}


def load_image_features(split, cached_data, device):
    """Load pre-computed ViT features for this split.

    If V-JEPA/SimCLR features were pre-computed, load them.
    Otherwise, compute on the fly from COSMO frame images.
    """
    # Try loading pre-cached image features
    feat_path = V5_ROOT / f"data/cached_image_features_{split}.npz"
    if feat_path.exists():
        d = np.load(feat_path)
        return torch.from_numpy(d["vit_feat"]).float()

    # Fall back: compute from images using frozen ViT
    print(f"  Computing ViT features for {split}...")
    from models.multiview_vit import MultiViewViT
    from data.dataset import COSMOBridgeV5Dataset

    vit = MultiViewViT(n_views=36, embed_dim=192, dropout=0.0)

    # Try loading V-JEPA or SimCLR weights
    for ckpt in [V5_ROOT / "checkpoints/vjepa/vit_pretrained_vjepa.pt",
                  V5_ROOT / "checkpoints/simclr/vit_pretrained.pt"]:
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            encoder_state = state.get("encoder_state_dict", {})
            if encoder_state:
                missing, _ = vit.load_state_dict(encoder_state, strict=False)
                print(f"  Loaded ViT from {ckpt.name} ({len(missing)} missing keys)")
                break

    vit = vit.to(device).eval()

    # Build dataset for image loading
    ds = COSMOBridgeV5Dataset(
        cached_features_path=str(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz"),
        cosmo_images_dir=str(V5_ROOT / "data/cosmo_images"),
        ion_images_dir=str(V5_ROOT / "data/ion_images"),
        orig_cosmo_dir=str(PROJECT_ROOT / "data/pipeline/cosmo_images"),
        master_index_path=str(V5_ROOT / "data/master_index.csv"),
        n_views=36, image_size=224,
        view_sample_mode="uniform_k", view_sample_k=6,
    )

    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    all_feats = []

    with torch.no_grad():
        for batch in loader:
            views = batch["views"].to(device)
            emb, _ = vit.encode_views_chunked(views, chunk_size=3)
            all_feats.append(emb.cpu())

    feats = torch.cat(all_feats)  # (N, 192)

    # Cache for next time
    np.savez(feat_path, vit_feat=feats.numpy())
    print(f"  Cached {feats.shape} to {feat_path}")

    return feats


def compute_metrics(preds, targets):
    metrics = {}
    for i, name in enumerate(PROPS):
        p, t = preds[:, i], targets[:, i]
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        metrics[f"{name}_r2"] = (1 - ss_res / (ss_tot + 1e-8)).item()
    metrics["avg_r2"] = np.mean([v for v in metrics.values()])
    return metrics


def pretrain_image_head(image_feat, targets, val_image_feat, val_targets,
                         device, n_epochs=200, patience=30):
    """Pre-train the image head to produce calibrated predictions.

    Same idea as how v4 pre-trains Path A and Path B separately.
    """
    head = ImageHead(image_dim=192, n_properties=7, dropout=0.3).to(device)
    optimizer = AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    ds = TensorDataset(image_feat, targets)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    val_ds = TensorDataset(val_image_feat, val_targets)
    val_loader = DataLoader(val_ds, batch_size=64)

    best_val_loss = float("inf")
    best_state = None
    no_imp = 0

    for epoch in range(n_epochs):
        head.train()
        for img_f, y in loader:
            img_f, y = img_f.to(device), y.to(device)
            preds = head(img_f)
            loss = ((preds - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        head.eval()
        val_loss = 0
        with torch.no_grad():
            for img_f, y in val_loader:
                img_f, y = img_f.to(device), y.to(device)
                val_loss += ((head(img_f) - y) ** 2).mean().item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    head.load_state_dict(best_state)
    head.eval()

    # Generate predictions for all splits
    with torch.no_grad():
        train_preds = head(image_feat.to(device)).cpu()
        val_preds = head(val_image_feat.to(device)).cpu()

    # Compute image-only R² for reference
    m = compute_metrics(train_preds.numpy(), targets.numpy())
    print(f"  Image head pre-training: train avg R²={m['avg_r2']:.4f}")

    return head, train_preds, val_preds


def train_one_seed(seed, train_data, val_data, test_data, device, config):
    """Train one seed: identical to v4 router training + image path."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = COSMOBridgeV4PlusImage(
        graph_dim=300, surface_dim=256, thermo_dim=25,
        image_dim=192, hidden=config["hidden"],
        n_properties=7, dropout=config["dropout"],
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"],
                      weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # DataLoader with all features + 3 path predictions
    train_ds = TensorDataset(
        train_data["chemprop_fp"], train_data["surface_fp"],
        train_data["thermo_feat"], train_data["image_feat"],
        train_data["preds_fusion"], train_data["preds_chemprop"],
        train_data["preds_image"], train_data["targets"],
    )
    val_ds = TensorDataset(
        val_data["chemprop_fp"], val_data["surface_fp"],
        val_data["thermo_feat"], val_data["image_feat"],
        val_data["preds_fusion"], val_data["preds_chemprop"],
        val_data["preds_image"], val_data["targets"],
    )

    train_ldr = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_ldr = DataLoader(val_ds, batch_size=64)

    best_val = float("inf")
    best_state = None
    no_imp = 0

    for epoch in range(config["epochs"]):
        # Anchor decay: 0.1 → 0.01 linearly (same as v4)
        anchor_w = config["anchor_init"] * (1 - epoch / config["epochs"]) + \
                   config["anchor_final"] * (epoch / config["epochs"])

        model.train()
        for g, s, t, img, pf, pc, pi, y in train_ldr:
            g, s, t, img = g.to(device), s.to(device), t.to(device), img.to(device)
            pf, pc, pi, y = pf.to(device), pc.to(device), pi.to(device), y.to(device)

            preds, aux = model(g, s, t, img, pf, pc, pi)
            mse = ((preds - y) ** 2).mean()

            # Anchor regularization (same as v4)
            router_logits = model.router.router(
                torch.cat([g, s, t, img], -1)
            )
            anchor = model.anchor_loss(router_logits)
            loss = mse + anchor_w * anchor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for g, s, t, img, pf, pc, pi, y in val_ldr:
                g, s, t, img = g.to(device), s.to(device), t.to(device), img.to(device)
                pf, pc, pi, y = pf.to(device), pc.to(device), pi.to(device), y.to(device)
                preds, _ = model(g, s, t, img, pf, pc, pi)
                val_losses.append(((preds - y) ** 2).mean().item())
        val_mse = np.mean(val_losses)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= config["patience"]:
                break

    # Test
    model.load_state_dict(best_state)
    model.eval()

    test_ds = TensorDataset(
        test_data["chemprop_fp"], test_data["surface_fp"],
        test_data["thermo_feat"], test_data["image_feat"],
        test_data["preds_fusion"], test_data["preds_chemprop"],
        test_data["preds_image"], test_data["targets"],
    )
    test_ldr = DataLoader(test_ds, batch_size=64)

    test_preds, test_targets, test_weights = [], [], []
    with torch.no_grad():
        for g, s, t, img, pf, pc, pi, y in test_ldr:
            g, s, t, img = g.to(device), s.to(device), t.to(device), img.to(device)
            pf, pc, pi, y = pf.to(device), pc.to(device), pi.to(device), y.to(device)
            preds, aux = model(g, s, t, img, pf, pc, pi)
            test_preds.append(preds.cpu().numpy())
            test_targets.append(y.cpu().numpy())
            test_weights.append(aux["weights"].cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    test_weights = np.concatenate(test_weights)

    metrics = compute_metrics(test_preds, test_targets)
    return metrics, test_preds, test_targets, test_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    s, e = map(int, args.seeds.split("-"))
    seeds = list(range(s, e + 1))

    print("COSMOBridge v4 + Image Path")
    print("  Exact v4 protocol + images as 3rd frozen pathway")
    print(f"  Seeds: {seeds}, Device: {device}")

    # Load v4 cached data (same as v4 training)
    print("\nLoading v4 cached data...")
    train_data = load_cached("train")
    val_data = load_cached("val")
    test_data = load_cached("test")

    # Load ViT image features
    print("\nLoading image features...")
    train_data["image_feat"] = load_image_features("train", train_data, device)
    val_data["image_feat"] = load_image_features("val", val_data, device)
    test_data["image_feat"] = load_image_features("test", test_data, device)

    # Step 1: Pre-train image head (produces preds_image)
    print("\nPre-training image head...")
    image_head, train_img_preds, val_img_preds = pretrain_image_head(
        train_data["image_feat"], train_data["targets"],
        val_data["image_feat"], val_data["targets"],
        device,
    )

    # Generate test image predictions
    image_head.eval()
    with torch.no_grad():
        test_img_preds = image_head(test_data["image_feat"].to(device)).cpu()

    train_data["preds_image"] = train_img_preds
    val_data["preds_image"] = val_img_preds
    test_data["preds_image"] = test_img_preds

    # Step 2: Train 3-way router (identical config to v4)
    config = {
        "hidden": 64,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "batch_size": 32,
        "epochs": 300,
        "patience": 40,
        "anchor_init": 0.1,
        "anchor_final": 0.01,
    }

    print(f"\nTraining 3-way router (v4 config)...")
    all_metrics = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        metrics, preds, targets, weights = train_one_seed(
            seed, train_data, val_data, test_data, device, config)

        print(f"  avg R²: {metrics['avg_r2']:.4f}")
        for p in PROPS:
            print(f"    {p}: R²={metrics[f'{p}_r2']:.4f}")

        # Print average routing weights
        w_mean = weights.mean(axis=0)  # (7, 3)
        print(f"  Routing weights (fusion/chemprop/image):")
        for i, p in enumerate(PROPS):
            print(f"    {p}: [{w_mean[i,0]:.3f}, {w_mean[i,1]:.3f}, {w_mean[i,2]:.3f}]")

        all_metrics.append(metrics)

        # Save
        pred_dir = V5_ROOT / "results/v4_plus_image/seed_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez(pred_dir / f"seed_{seed}.npz",
                 predictions=preds, targets=targets, weights=weights)

    # Summary
    print(f"\n{'='*60}")
    print("v4 + IMAGE SUMMARY")
    print(f"{'='*60}")
    avgs = [m["avg_r2"] for m in all_metrics]
    print(f"  avg R²: {np.mean(avgs):.4f} ± {np.std(avgs):.4f}")
    print(f"  v4 original: 0.810")
    print(f"  Delta: {np.mean(avgs) - 0.810:+.4f}")

    for p in PROPS:
        vals = [m[f"{p}_r2"] for m in all_metrics]
        print(f"  {p:8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    out = V5_ROOT / "results/v4_plus_image"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({"per_seed": [m for m in all_metrics],
                    "avg_mean": float(np.mean(avgs)),
                    "avg_std": float(np.std(avgs))}, f, indent=2)


if __name__ == "__main__":
    main()
