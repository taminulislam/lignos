#!/usr/bin/env python3
"""Train with properly-aligned merged data.

Runs two experiments:
    Path 2: Same 28 ILs at varied T/x1 (223 -> ~983 samples)
    Path 1+2: All 143 ILs with precomputed features (223 -> ~5,845 samples)

Also runs baseline (original 223 samples only) for clean comparison.

Usage:
    python train_merged_v2.py --mode path2 --seeds 0-9
    python train_merged_v2.py --mode path1 --seeds 0-9
    python train_merged_v2.py --mode baseline --seeds 0-9
    python train_merged_v2.py --mode all --seeds 0-9
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

from data.merged_dataset_v2 import MergedDatasetV2, masked_mse_loss


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class V4Model(nn.Module):
    """v4-style model: Graph + Surface + Tabular with per-property gate."""

    def __init__(self, graph_dim=300, surface_dim=256, thermo_dim=25,
                 fused_dim=256, n_properties=7, dropout=0.3):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.surface_proj = nn.Linear(surface_dim, fused_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.LayerNorm(fused_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.fused_head = nn.Sequential(
            nn.Linear(fused_dim + thermo_dim, 128), nn.BatchNorm1d(128),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(128, n_properties),
        )
        self.direct_head = nn.Sequential(
            nn.Linear(graph_dim + thermo_dim, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_properties),
        )
        self.gate_logits = nn.Parameter(torch.tensor(
            [2.0, 2.0, -2.0, -2.0, -2.0, 0.0, 1.5]))

    def forward(self, batch):
        g = batch["graph_feat"]
        s = batch["surface_feat"]
        t = batch["thermo_feat"]
        g_proj = self.graph_proj(g)
        s_proj = self.surface_proj(s)
        fused = self.fusion_mlp(g_proj * s_proj)
        preds_fused = self.fused_head(torch.cat([fused, t], -1))
        preds_direct = self.direct_head(torch.cat([g, t], -1))
        alpha = torch.sigmoid(self.gate_logits)
        return alpha * preds_fused + (1 - alpha) * preds_direct


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
    valid = [v for k, v in metrics.items() if not np.isnan(v)]
    metrics["avg_r2"] = np.mean(valid) if valid else 0.0
    return metrics


def train_experiment(mode, seed, device, precomputed_path=None):
    """Run one experiment (one seed)."""
    set_seed(seed)

    ilthermo_csv = str(PROJECT_ROOT / "data/augmented/ilthermo_data.csv")

    # Build datasets
    include_it = mode in ("path2", "path1")
    same_ils = (mode == "path2")

    train_ds = MergedDatasetV2(
        cached_path=str(PROJECT_ROOT / "cosmobridge_v4/data/cached_train.npz"),
        ilthermo_csv=ilthermo_csv if include_it else None,
        project_root=str(PROJECT_ROOT),
        same_ils_only=same_ils,
        precomputed_features_path=precomputed_path if mode == "path1" else None,
        include_ilthermo=include_it,
    )

    val_ds = MergedDatasetV2(
        cached_path=str(PROJECT_ROOT / "cosmobridge_v4/data/cached_val.npz"),
        project_root=str(PROJECT_ROOT),
        include_ilthermo=False,
    )

    test_ds = MergedDatasetV2(
        cached_path=str(PROJECT_ROOT / "cosmobridge_v4/data/cached_test.npz"),
        project_root=str(PROJECT_ROOT),
        include_ilthermo=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=min(64, len(train_ds)),
        shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=0,
    )

    # Model
    model = V4Model(dropout=0.3).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} params, ratio: {params/len(train_ds):.0f}:1")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    ckpt_dir = V5_ROOT / f"checkpoints/{mode}" / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_r2 = -float("inf")
    patience = 0

    for epoch in range(1, 151):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0, 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            preds = model(batch)
            loss = masked_mse_loss(preds, batch["targets"], batch["mask"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Validate
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                vp.append(model(batch).cpu())
                vt.append(batch["targets"].cpu())
        vp, vt = torch.cat(vp), torch.cat(vt)
        val_m = compute_metrics(vp, vt)
        elapsed = time.time() - t0

        if epoch <= 5 or epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | loss={total_loss/n_batches:.4f} | "
                  f"val R²={val_m['avg_r2']:.4f} | {elapsed:.1f}s")

        if val_m["avg_r2"] > best_val_r2:
            best_val_r2 = val_m["avg_r2"]
            patience = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            patience += 1
            if patience >= 30:
                print(f"  Early stop epoch {epoch} (best val={best_val_r2:.4f})")
                break

    # Test
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", weights_only=True))
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            tp.append(model(batch).cpu())
            tt.append(batch["targets"].cpu())
    tp, tt = torch.cat(tp), torch.cat(tt)
    test_m = compute_metrics(tp, tt)

    print(f"\n  Test (seed {seed}):")
    for prop in ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]:
        print(f"    {prop:8s}: R²={test_m.get(f'{prop}_r2', 0):.4f}")
    print(f"    {'avg':8s}: R²={test_m['avg_r2']:.4f}")

    # Save predictions
    pred_dir = V5_ROOT / f"results/{mode}/seed_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / f"seed_{seed}.npz",
             predictions=tp.numpy(), targets=tt.numpy())

    return test_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "path2", "path1", "all"],
                        default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--precomputed", type=str, default=None,
                        help="Path to precomputed Chemprop features for Path 1")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.seed is not None:
        seeds = [args.seed]
    else:
        s, e = map(int, args.seeds.split("-"))
        seeds = list(range(s, e + 1))

    modes = ["baseline", "path2", "path1"] if args.mode == "all" else [args.mode]

    # Skip path1 if no precomputed features
    if "path1" in modes and args.precomputed is None:
        precomp = V5_ROOT / "data/precomputed_chemprop_features.npz"
        if not precomp.exists():
            print("WARNING: Skipping path1 (no precomputed features)")
            print(f"  Run: python scripts/precompute_chemprop.py first")
            modes = [m for m in modes if m != "path1"]
        else:
            args.precomputed = str(precomp)

    all_results = {}
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {mode.upper()}")
        print(f"{'='*60}")

        results = {}
        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")
            m = train_experiment(mode, seed, device, args.precomputed)
            results[seed] = m

        avg_r2 = [m["avg_r2"] for m in results.values()]
        print(f"\n  {mode} SUMMARY: avg R²={np.mean(avg_r2):.4f}±{np.std(avg_r2):.4f}")
        all_results[mode] = results

    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Mode':>12s} {'avg R²':>12s} {'gamma1':>8s} {'gamma2':>8s} "
          f"{'G_E':>8s} {'H_E':>8s} {'G_mix':>8s}")

    for mode, results in all_results.items():
        avg = np.mean([m["avg_r2"] for m in results.values()])
        g1 = np.mean([m.get("gamma1_r2", 0) for m in results.values()])
        g2 = np.mean([m.get("gamma2_r2", 0) for m in results.values()])
        ge = np.mean([m.get("G_E_r2", 0) for m in results.values()])
        he = np.mean([m.get("H_E_r2", 0) for m in results.values()])
        gm = np.mean([m.get("G_mix_r2", 0) for m in results.values()])
        std = np.std([m["avg_r2"] for m in results.values()])
        print(f"{mode:>12s} {avg:>6.4f}±{std:.3f} {g1:>8.4f} {g2:>8.4f} "
              f"{ge:>8.4f} {he:>8.4f} {gm:>8.4f}")

    print(f"\n  v4 original baseline: R² = 0.810")

    # Save
    save = {mode: {str(k): v for k, v in res.items()}
            for mode, res in all_results.items()}
    out = V5_ROOT / "results/merged_comparison.json"
    with open(out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
