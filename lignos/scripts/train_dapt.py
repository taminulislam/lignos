#!/usr/bin/env python3
"""Domain-Adaptive Pre-training (DAPT) for COSMOBridge v5.

Two-phase training:
    Phase 1 (Domain pre-training): Train on ALL iThermo data (~5,000 samples)
        - ALL properties used (no masking!) including gamma1 and H_E
        - Masked multi-task loss for missing properties
        - Model learns broad IL property patterns from diverse structures

    Phase 2 (Task fine-tuning): Fine-tune on original 28 ILs (152 samples)
        - All 7 properties available for every sample
        - Lower learning rate, early stopping on validation
        - Model specializes to the target distribution

Why DAPT > STILT:
    - No gamma1 masking = no wasted data
    - No 48x oversampling tricks needed
    - Two-phase naturally prevents distribution interference
    - Proven paradigm: BERT -> domain BERT -> task fine-tuning

Usage:
    python train_dapt.py --seeds 0-9 --device cuda
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

PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


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
                 fused_dim=256, n_props=7, dropout=0.3):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.surface_proj = nn.Linear(surface_dim, fused_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.LayerNorm(fused_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.fused_head = nn.Sequential(
            nn.Linear(fused_dim + thermo_dim, 128), nn.BatchNorm1d(128),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(128, n_props))
        self.direct_head = nn.Sequential(
            nn.Linear(graph_dim + thermo_dim, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_props))
        self.gate = nn.Parameter(torch.tensor([2., 2., -2., -2., -2., 0., 1.5]))

    def forward(self, batch):
        g = batch["graph_feat"]
        s = batch["surface_feat"]
        t = batch["thermo_feat"]
        fused = self.fusion_mlp(self.graph_proj(g) * self.surface_proj(s))
        pf = self.fused_head(torch.cat([fused, t], -1))
        pd = self.direct_head(torch.cat([g, t], -1))
        alpha = torch.sigmoid(self.gate)
        return alpha * pf + (1 - alpha) * pd


def compute_metrics(preds, targets, masks=None):
    metrics = {}
    for i, name in enumerate(PROPS):
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


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        preds = model(batch)
        loss = masked_mse_loss(preds, batch["targets"], batch["mask"])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_p, all_t = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        all_p.append(model(batch).cpu())
        all_t.append(batch["targets"].cpu())
    return torch.cat(all_p), torch.cat(all_t)


def train_single_seed(seed, device):
    set_seed(seed)
    print(f"\n{'#'*60}")
    print(f"  SEED {seed} -- DAPT (Domain-Adaptive Pre-training)")
    print(f"{'#'*60}")

    precomp = str(V5_ROOT / "data/precomputed_chemprop_features.npz")
    ilthermo_csv = str(PROJECT_ROOT / "data/augmented/ilthermo_data.csv")

    # ══════════════════════════════════════════
    # Phase 1: Domain pre-training on ALL iThermo
    # ══════════════════════════════════════════
    print("\n  ── Phase 1: Domain Pre-training (ALL iThermo) ──")

    domain_ds = MergedDatasetV2(
        cached_path=str(PROJECT_ROOT / "cosmobridge_v4/data/cached_train.npz"),
        ilthermo_csv=ilthermo_csv,
        project_root=str(PROJECT_ROOT),
        same_ils_only=False,  # ALL iThermo ILs
        precomputed_features_path=precomp,
        include_ilthermo=True,
    )

    # Val set stays original (no iThermo contamination)
    val_ds = MergedDatasetV2(
        cached_path=str(PROJECT_ROOT / "cosmobridge_v4/data/cached_val.npz"),
        project_root=str(PROJECT_ROOT),
        include_ilthermo=False,
    )

    domain_loader = torch.utils.data.DataLoader(
        domain_ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, num_workers=0)

    model = V4Model(dropout=0.3).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} params")
    print(f"  Phase 1 data: {len(domain_ds)} samples")

    # Phase 1 optimizer: higher LR, train broadly
    opt1 = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=80)

    best_val_p1 = -float("inf")
    patience = 0
    ckpt_dir = V5_ROOT / f"checkpoints/dapt/seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 81):
        t0 = time.time()
        loss = train_epoch(model, domain_loader, opt1, device)
        sched1.step()
        vp, vt = evaluate(model, val_loader, device)
        vm = compute_metrics(vp, vt)
        elapsed = time.time() - t0

        if epoch <= 5 or epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}/80 | loss={loss:.4f} | "
                  f"val R²={vm['avg_r2']:.4f} | {elapsed:.1f}s")

        if vm["avg_r2"] > best_val_p1:
            best_val_p1 = vm["avg_r2"]
            patience = 0
            torch.save(model.state_dict(), ckpt_dir / "phase1_best.pt")
        else:
            patience += 1
            if patience >= 25:
                print(f"    Phase 1 early stop at epoch {epoch} "
                      f"(best val={best_val_p1:.4f})")
                break

    print(f"  Phase 1 complete: best val R²={best_val_p1:.4f}")

    # ══════════════════════════════════════════
    # Phase 2: Fine-tune on original 28 ILs
    # ══════════════════════════════════════════
    print("\n  ── Phase 2: Task Fine-tuning (Original 28 ILs) ──")

    # Load Phase 1 best checkpoint
    model.load_state_dict(torch.load(ckpt_dir / "phase1_best.pt", weights_only=True))

    # Original data only (no iThermo)
    finetune_ds = MergedDatasetV2(
        cached_path=str(PROJECT_ROOT / "cosmobridge_v4/data/cached_train.npz"),
        project_root=str(PROJECT_ROOT),
        include_ilthermo=False,
    )

    test_ds = MergedDatasetV2(
        cached_path=str(PROJECT_ROOT / "cosmobridge_v4/data/cached_test.npz"),
        project_root=str(PROJECT_ROOT),
        include_ilthermo=False,
    )

    ft_loader = torch.utils.data.DataLoader(
        finetune_ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=64, num_workers=0)

    print(f"  Phase 2 data: {len(finetune_ds)} samples (original only)")

    # Phase 2 optimizer: lower LR for fine-tuning
    opt2 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=100)

    best_val_p2 = -float("inf")
    patience = 0

    for epoch in range(1, 101):
        t0 = time.time()
        loss = train_epoch(model, ft_loader, opt2, device)
        sched2.step()
        vp, vt = evaluate(model, val_loader, device)
        vm = compute_metrics(vp, vt)
        elapsed = time.time() - t0

        if epoch <= 5 or epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}/100 | loss={loss:.4f} | "
                  f"val R²={vm['avg_r2']:.4f} | {elapsed:.1f}s")

        if vm["avg_r2"] > best_val_p2:
            best_val_p2 = vm["avg_r2"]
            patience = 0
            torch.save(model.state_dict(), ckpt_dir / "phase2_best.pt")
        else:
            patience += 1
            if patience >= 30:
                print(f"    Phase 2 early stop at epoch {epoch} "
                      f"(best val={best_val_p2:.4f})")
                break

    print(f"  Phase 2 complete: best val R²={best_val_p2:.4f}")

    # ══════════════════════════════════════════
    # Test evaluation
    # ══════════════════════════════════════════
    model.load_state_dict(torch.load(ckpt_dir / "phase2_best.pt", weights_only=True))
    tp, tt = evaluate(model, test_loader, device)
    tm = compute_metrics(tp, tt)

    print(f"\n  Test Results (seed {seed}):")
    for prop in PROPS:
        print(f"    {prop:8s}: R² = {tm[f'{prop}_r2']:.4f}")
    print(f"    {'avg':8s}: R² = {tm['avg_r2']:.4f}")

    # Save
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(tm, f, indent=2)

    pred_dir = V5_ROOT / "results/dapt/seed_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / f"seed_{seed}.npz",
             predictions=tp.numpy(), targets=tt.numpy())

    return tm


def main():
    parser = argparse.ArgumentParser(description="DAPT Training")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.seed is not None:
        seeds = [args.seed]
    else:
        s, e = map(int, args.seeds.split("-"))
        seeds = list(range(s, e + 1))

    print("Domain-Adaptive Pre-training (DAPT)")
    print(f"  Phase 1: ALL iThermo (broad domain learning)")
    print(f"  Phase 2: Original 28 ILs (task specialization)")
    print(f"  Seeds: {seeds}, Device: {device}")

    all_results = {}
    for seed in seeds:
        tm = train_single_seed(seed, device)
        all_results[seed] = tm

    # Summary
    print(f"\n{'='*60}")
    print("DAPT SUMMARY")
    print(f"{'='*60}")
    avgs = [m["avg_r2"] for m in all_results.values()]
    print(f"  avg R²: {np.mean(avgs):.4f} ± {np.std(avgs):.4f}")
    print(f"  (v4 original: 0.810, v5+SimCLR: 0.657, Path2: 0.642)")

    for prop in PROPS:
        key = f"{prop}_r2"
        vals = [m[key] for m in all_results.values() if not np.isnan(m.get(key, 0))]
        if vals:
            print(f"  {prop:8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    out = V5_ROOT / "results/dapt"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved: {out}/summary.json")


if __name__ == "__main__":
    main()
