#!/usr/bin/env python3
"""Train COSMOBridge v5.1: Better Fusion + Descriptors + Scale.

Two modes:
    --mode original: Train on 152 samples only (compare with v4)
    --mode dapt: Phase 1 on iThermo (143 ILs) → Phase 2 on original 28 ILs

Usage:
    python train_v5_1.py --mode original --seeds 0-9
    python train_v5_1.py --mode dapt --seeds 0-9
"""

import argparse
import json
import sys
import time
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

from models.cosmobridge_v5_1 import COSMOBridgeV51

PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cached(split):
    d = np.load(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz",
                allow_pickle=True)
    return {k: torch.from_numpy(d[k]).float() if d[k].dtype.kind == 'f'
            else d[k] for k in d.keys()}


def load_dapt_data():
    """Load iThermo data with real Chemprop features for DAPT Phase 1."""
    from data.merged_dataset_v2 import MergedDatasetV2

    ds = MergedDatasetV2(
        cached_path=str(PROJECT_ROOT / "cosmobridge_v4/data/cached_train.npz"),
        ilthermo_csv=str(PROJECT_ROOT / "data/augmented/ilthermo_data.csv"),
        project_root=str(PROJECT_ROOT),
        same_ils_only=False,
        precomputed_features_path=str(V5_ROOT / "data/precomputed_chemprop_features.npz"),
        include_ilthermo=True,
    )
    return ds


def compute_metrics(preds, targets):
    metrics = {}
    for i, name in enumerate(PROPS):
        p, t = preds[:, i], targets[:, i]
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        metrics[f"{name}_r2"] = (1 - ss_res / (ss_tot + 1e-8)).item()
    metrics["avg_r2"] = np.mean([v for v in metrics.values()])
    return metrics


def masked_mse_loss(preds, targets, mask=None):
    if mask is None:
        return ((preds - targets) ** 2).mean()
    mask_f = mask.float()
    n = mask_f.sum()
    if n == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    return ((preds - targets) ** 2 * mask_f).sum() / n


def train_epoch(model, loader, optimizer, device, has_mask=False):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        if has_mask:
            g, s, t, tgt, mask = [x.to(device) for x in batch[:5]]
            preds, _ = model(g, s, t)
            loss = masked_mse_loss(preds, tgt, mask)
        else:
            g, s, t, tgt = [x.to(device) for x in batch[:4]]
            preds, _ = model(g, s, t)
            loss = masked_mse_loss(preds, tgt)
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
        g, s, t, tgt = [x.to(device) for x in batch[:4]]
        preds, _ = model(g, s, t)
        all_p.append(preds.cpu())
        all_t.append(tgt.cpu())
    return torch.cat(all_p), torch.cat(all_t)


def train_single_seed(seed, mode, device):
    set_seed(seed)
    print(f"\n{'#'*60}")
    print(f"  SEED {seed} — v5.1 ({mode})")
    print(f"{'#'*60}")

    # Build model
    model = COSMOBridgeV51(
        graph_dim=300, surface_dim=256, thermo_dim=25,
        fused_dim=256, n_properties=7, image_dim=0, dropout=0.3,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} params")

    # Load original splits
    train_data = load_cached("train")
    val_data = load_cached("val")
    test_data = load_cached("test")

    val_ds = TensorDataset(
        val_data["chemprop_fp"], val_data["surface_fp"],
        val_data["thermo_feat"], val_data["targets"])
    test_ds = TensorDataset(
        test_data["chemprop_fp"], test_data["surface_fp"],
        test_data["thermo_feat"], test_data["targets"])
    val_ldr = DataLoader(val_ds, batch_size=64)
    test_ldr = DataLoader(test_ds, batch_size=64)

    ckpt_dir = V5_ROOT / f"checkpoints/v5_1_{mode}" / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if mode == "dapt":
        # ── Phase 1: Domain pre-training on ALL iThermo ──
        print("\n  Phase 1: Domain pre-training...")
        dapt_ds = load_dapt_data()
        dapt_ldr = DataLoader(
            TensorDataset(
                dapt_ds.graph_feat, dapt_ds.surface_feat,
                dapt_ds.thermo_feat, dapt_ds.targets, dapt_ds.masks,
            ),
            batch_size=128, shuffle=True, drop_last=True,
        )
        print(f"  DAPT data: {len(dapt_ds)} samples")

        opt1 = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        sched1 = CosineAnnealingLR(opt1, T_max=80)

        best_val_p1 = -1e9
        patience = 0
        for epoch in range(1, 81):
            loss = train_epoch(model, dapt_ldr, opt1, device, has_mask=True)
            sched1.step()
            vp, vt = evaluate(model, val_ldr, device)
            vm = compute_metrics(vp.numpy(), vt.numpy())
            if epoch <= 3 or epoch % 20 == 0:
                print(f"    P1 Ep {epoch:3d} | loss={loss:.4f} | val R²={vm['avg_r2']:.4f}")
            if vm["avg_r2"] > best_val_p1:
                best_val_p1 = vm["avg_r2"]
                patience = 0
                torch.save(model.state_dict(), ckpt_dir / "phase1.pt")
            else:
                patience += 1
                if patience >= 25:
                    break
        model.load_state_dict(torch.load(ckpt_dir / "phase1.pt", weights_only=True))
        print(f"  Phase 1 done: best val R²={best_val_p1:.4f}")

    # ── Phase 2 (or only phase for 'original'): Fine-tune on original ──
    phase_name = "Phase 2: Fine-tune" if mode == "dapt" else "Training"
    print(f"\n  {phase_name} on original 152 samples...")

    train_ds = TensorDataset(
        train_data["chemprop_fp"], train_data["surface_fp"],
        train_data["thermo_feat"], train_data["targets"])
    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True)

    lr = 5e-4 if mode == "dapt" else 1e-3
    opt2 = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched2 = CosineAnnealingLR(opt2, T_max=300)

    best_val = -1e9
    patience = 0
    for epoch in range(1, 301):
        loss = train_epoch(model, train_ldr, opt2, device)
        sched2.step()
        vp, vt = evaluate(model, val_ldr, device)
        vm = compute_metrics(vp.numpy(), vt.numpy())
        if epoch <= 5 or epoch % 50 == 0:
            print(f"    Ep {epoch:3d} | loss={loss:.4f} | val R²={vm['avg_r2']:.4f}")
        if vm["avg_r2"] > best_val:
            best_val = vm["avg_r2"]
            patience = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            patience += 1
            if patience >= 40:
                print(f"    Early stop at {epoch} (best val={best_val:.4f})")
                break

    # Test
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", weights_only=True))
    tp, tt = evaluate(model, test_ldr, device)
    tm = compute_metrics(tp.numpy(), tt.numpy())

    print(f"\n  Test Results (seed {seed}):")
    for p in PROPS:
        print(f"    {p:8s}: R²={tm[f'{p}_r2']:.4f}")
    print(f"    {'avg':8s}: R²={tm['avg_r2']:.4f}")

    # Routing weights
    model.eval()
    with torch.no_grad():
        _, aux = model(
            test_data["chemprop_fp"].to(device),
            test_data["surface_fp"].to(device),
            test_data["thermo_feat"].to(device),
        )
    w = aux["weights"].cpu().mean(0).numpy()  # (7, 3)
    path_names = ["Fusion", "Chemprop", "Descriptors"]
    print(f"  Routing (mean):")
    for i, p in enumerate(PROPS):
        vals = " ".join(f"{path_names[j]}={w[i,j]:.3f}" for j in range(3))
        print(f"    {p:8s}: {vals}")

    # Save
    pred_dir = V5_ROOT / f"results/v5_1_{mode}/seed_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / f"seed_{seed}.npz",
             predictions=tp.numpy(), targets=tt.numpy())

    return tm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["original", "dapt", "both"],
                        default="both")
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    s, e = map(int, args.seeds.split("-"))
    seeds = list(range(s, e + 1))
    modes = ["original", "dapt"] if args.mode == "both" else [args.mode]

    print("COSMOBridge v5.1: Better Fusion + Descriptors")
    print(f"  Modes: {modes}, Seeds: {seeds}")

    all_results = {}
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  MODE: {mode.upper()}")
        print(f"{'='*60}")

        results = {}
        for seed in seeds:
            tm = train_single_seed(seed, mode, device)
            results[seed] = tm

        avgs = [m["avg_r2"] for m in results.values()]
        print(f"\n  {mode} SUMMARY: avg R²={np.mean(avgs):.4f} ± {np.std(avgs):.4f}")
        all_results[mode] = results

    # Final comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  v4 router (actual):    0.8078 ± 0.0003")
    for mode, results in all_results.items():
        avgs = [m["avg_r2"] for m in results.values()]
        print(f"  v5.1 ({mode:8s}):   {np.mean(avgs):.4f} ± {np.std(avgs):.4f}  "
              f"(Δ={np.mean(avgs)-0.8078:+.4f})")

    # Save
    out = V5_ROOT / "results/v5_1"
    out.mkdir(exist_ok=True)
    save = {mode: {str(k): v for k, v in res.items()}
            for mode, res in all_results.items()}
    with open(out / "summary.json", "w") as f:
        json.dump(save, f, indent=2)


if __name__ == "__main__":
    main()
