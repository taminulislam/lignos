#!/usr/bin/env python3
"""v4 + Residual Image: Images predict the ERROR of v4, not the full signal.

Three innovations combined:
    Solution 1: RESIDUAL prediction (images learn v4's errors)
    Solution 4: PCA dimensionality reduction (192D -> 20D, prevents overfitting)
    Solution 9: Temperature conditioning (modulates static image features by T)

Final prediction: y = v4_pred + alpha * residual_head(PCA(ViT_feat) * T_embed)

Usage:
    python train_v4_residual_image.py --seeds 0-9 --device cuda
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
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5_ROOT))

PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


class TemperatureConditionedResidualHead(nn.Module):
    """Predicts v4 residuals from PCA-reduced ViT features x temperature.

    Input: PCA(ViT_feat) (20D) + thermo_feat (25D)
    Output: 7 residual corrections

    The temperature conditioning allows static molecular images
    to produce different corrections at different temperatures.
    """

    def __init__(self, image_pca_dim=20, thermo_dim=25, n_properties=7, dropout=0.3):
        super().__init__()
        # Temperature modulation: learn which image features matter at each T
        self.temp_gate = nn.Sequential(
            nn.Linear(thermo_dim, 32),
            nn.GELU(),
            nn.Linear(32, image_pca_dim),
            nn.Sigmoid(),  # gates in [0, 1]
        )

        # Residual prediction head (small to prevent overfitting)
        input_dim = image_pca_dim + thermo_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_properties),
        )

        # Initialize output near zero (residual starts small)
        with torch.no_grad():
            self.head[-1].weight.mul_(0.01)
            self.head[-1].bias.zero_()

    def forward(self, image_pca, thermo_feat):
        """
        Args:
            image_pca: (B, 20) PCA-reduced ViT features
            thermo_feat: (B, 25) thermodynamic features

        Returns:
            residual: (B, 7) predicted corrections to v4
        """
        # Temperature-modulated image features
        gate = self.temp_gate(thermo_feat)  # (B, 20)
        modulated = image_pca * gate  # element-wise gating

        # Predict residual
        combined = torch.cat([modulated, thermo_feat], dim=-1)
        return self.head(combined)


class V4PlusResidualImage(nn.Module):
    """v4 predictions + learned residual from images.

    y = v4_router_pred + alpha * residual_head(PCA(ViT), thermo)

    Alpha is a learnable per-property scale that starts at 0
    (pure v4) and gradually allows image corrections.
    """

    def __init__(self, image_pca_dim=20, thermo_dim=25, n_properties=7, dropout=0.3):
        super().__init__()
        self.residual_head = TemperatureConditionedResidualHead(
            image_pca_dim, thermo_dim, n_properties, dropout)

        # Per-property scaling (starts at 0 = no image contribution)
        self.alpha_logits = nn.Parameter(torch.full((n_properties,), -3.0))
        # sigmoid(-3) ≈ 0.047, so images start with <5% contribution

    def forward(self, v4_preds, image_pca, thermo_feat):
        residual = self.residual_head(image_pca, thermo_feat)
        alpha = torch.sigmoid(self.alpha_logits)  # (7,) in [0, 1]
        corrected = v4_preds + alpha.unsqueeze(0) * residual
        return corrected, {"residual": residual.detach(), "alpha": alpha.detach()}


def compute_metrics(preds, targets):
    metrics = {}
    for i, name in enumerate(PROPS):
        p, t = preds[:, i], targets[:, i]
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        metrics[f"{name}_r2"] = (1 - ss_res / (ss_tot + 1e-8)).item()
    metrics["avg_r2"] = np.mean([v for v in metrics.values()])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--pca_dim", type=int, default=20)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    s, e = map(int, args.seeds.split("-"))
    seeds = list(range(s, e + 1))

    print("v4 + Residual Image (Solutions 1+4+9)")
    print(f"  PCA dim: {args.pca_dim}, Seeds: {seeds}")

    # Load data
    splits = {}
    for split in ["train", "val", "test"]:
        cached = np.load(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz",
                         allow_pickle=True)
        img = np.load(V5_ROOT / f"data/cached_image_features_{split}.npz")
        splits[split] = {
            "thermo_feat": cached["thermo_feat"].astype(np.float32),
            "targets": cached["targets"].astype(np.float32),
            "preds_fusion": cached["preds_fusion"].astype(np.float32),
            "preds_chemprop": cached["preds_chemprop"].astype(np.float32),
            "image_feat": img["vit_feat"].astype(np.float32),
        }

    # Compute v4 router predictions (best blend of fusion + chemprop)
    # Use the v4 result: alpha * fusion + (1-alpha) * chemprop
    # For simplicity, use the fusion predictions as the v4 base
    # (the router adds ~0.01, so fusion is a good proxy)
    for split in splits:
        d = splits[split]
        # Approximate v4 by averaging fusion and chemprop (since gate ~0.5)
        d["v4_preds"] = 0.5 * d["preds_fusion"] + 0.5 * d["preds_chemprop"]

    # ── Solution 4: PCA on image features ──
    print(f"\nApplying PCA: 192D -> {args.pca_dim}D")
    pca = PCA(n_components=args.pca_dim)
    pca.fit(splits["train"]["image_feat"])
    explained = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance: {explained:.1%}")

    for split in splits:
        splits[split]["image_pca"] = pca.transform(
            splits[split]["image_feat"]).astype(np.float32)

    # Print correlation of PCA components with residuals
    train_residuals = splits["train"]["targets"] - splits["train"]["v4_preds"]
    print(f"\nResidual statistics:")
    for i, p in enumerate(PROPS):
        res = train_residuals[:, i]
        print(f"  {p:8s}: mean={res.mean():.4f}, std={res.std():.4f}, "
              f"|max|={np.abs(res).max():.4f}")

    # Check PCA feature correlation with residuals
    print(f"\nPCA feature correlation with residuals:")
    from scipy.stats import pearsonr
    for i, p in enumerate(PROPS):
        corrs = [abs(pearsonr(splits["train"]["image_pca"][:, j],
                               train_residuals[:, i])[0])
                 for j in range(args.pca_dim)]
        print(f"  {p:8s}: max|r|={max(corrs):.4f} (PC{np.argmax(corrs)})")

    # ── Train ──
    all_metrics = []
    v4_only_metrics = compute_metrics(splits["test"]["v4_preds"],
                                       splits["test"]["targets"])
    print(f"\nv4 baseline (fusion+chemprop avg): avg R²={v4_only_metrics['avg_r2']:.4f}")

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = V4PlusResidualImage(
            image_pca_dim=args.pca_dim, thermo_dim=25,
            n_properties=7, dropout=0.3,
        ).to(device)

        params = sum(p.numel() for p in model.parameters())
        print(f"\n=== Seed {seed} === (model: {params:,} params)")

        # DataLoader
        train_ds = TensorDataset(
            torch.from_numpy(splits["train"]["v4_preds"]),
            torch.from_numpy(splits["train"]["image_pca"]),
            torch.from_numpy(splits["train"]["thermo_feat"]),
            torch.from_numpy(splits["train"]["targets"]),
        )
        val_ds = TensorDataset(
            torch.from_numpy(splits["val"]["v4_preds"]),
            torch.from_numpy(splits["val"]["image_pca"]),
            torch.from_numpy(splits["val"]["thermo_feat"]),
            torch.from_numpy(splits["val"]["targets"]),
        )

        train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_ldr = DataLoader(val_ds, batch_size=64)

        optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=300)

        best_val = float("inf")
        best_state = None
        no_imp = 0

        for epoch in range(300):
            model.train()
            for v4_p, img_p, thermo, y in train_ldr:
                v4_p, img_p, thermo, y = [x.to(device) for x in [v4_p, img_p, thermo, y]]
                preds, aux = model(v4_p, img_p, thermo)
                loss = ((preds - y) ** 2).mean()
                # L2 on residual output to keep corrections small
                loss += 0.01 * (aux["residual"] ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_losses = []
                for v4_p, img_p, thermo, y in val_ldr:
                    v4_p, img_p, thermo, y = [x.to(device) for x in [v4_p, img_p, thermo, y]]
                    preds, _ = model(v4_p, img_p, thermo)
                    val_losses.append(((preds - y) ** 2).mean().item())
            val_mse = np.mean(val_losses)

            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= 50:
                    break

        # Test
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_preds, _ = model(
                torch.from_numpy(splits["test"]["v4_preds"]).to(device),
                torch.from_numpy(splits["test"]["image_pca"]).to(device),
                torch.from_numpy(splits["test"]["thermo_feat"]).to(device),
            )
        test_preds = test_preds.cpu().numpy()
        metrics = compute_metrics(test_preds, splits["test"]["targets"])

        # Get learned alpha values
        alpha = torch.sigmoid(model.alpha_logits).detach().cpu().numpy()

        print(f"  avg R²: {metrics['avg_r2']:.4f} (v4 base: {v4_only_metrics['avg_r2']:.4f})")
        print(f"  Alpha (image contribution): {dict(zip(PROPS, [f'{a:.3f}' for a in alpha]))}")
        for p in PROPS:
            d = metrics[f'{p}_r2'] - v4_only_metrics[f'{p}_r2']
            print(f"    {p:8s}: R²={metrics[f'{p}_r2']:.4f} (Δ={d:+.4f})")

        all_metrics.append(metrics)

        # Save
        pred_dir = V5_ROOT / "results/v4_residual_image/seed_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez(pred_dir / f"seed_{seed}.npz",
                 predictions=test_preds,
                 targets=splits["test"]["targets"],
                 alpha=alpha)

    # Summary
    print(f"\n{'='*60}")
    print("v4 + RESIDUAL IMAGE SUMMARY")
    print(f"{'='*60}")
    avgs = [m["avg_r2"] for m in all_metrics]
    print(f"  avg R²: {np.mean(avgs):.4f} ± {np.std(avgs):.4f}")
    print(f"  v4 base: {v4_only_metrics['avg_r2']:.4f}")
    print(f"  Delta: {np.mean(avgs) - v4_only_metrics['avg_r2']:+.4f}")

    for p in PROPS:
        vals = [m[f"{p}_r2"] for m in all_metrics]
        base = v4_only_metrics[f"{p}_r2"]
        delta = np.mean(vals) - base
        print(f"  {p:8s}: {np.mean(vals):.4f}±{np.std(vals):.4f} (Δ={delta:+.4f})")

    out = V5_ROOT / "results/v4_residual_image"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({"per_seed": all_metrics,
                    "avg_mean": float(np.mean(avgs)),
                    "v4_baseline": v4_only_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
