#!/usr/bin/env python3
"""v5.1 Staged: Pre-train 3 paths separately, then train router.

Follows the EXACT v4 paradigm:
    Stage 1: Pre-train Path A (cross-attention fusion) → freeze
    Stage 2: Pre-train Path C (descriptor pathway) → freeze
    Stage 3: Path B already exists (v4 chemprop preds) → frozen
    Stage 4: Train 3-way router on frozen predictions (~50K params)

This matches v4's parameter efficiency while adding the improved fusion
and descriptor pathway.

Usage:
    python train_v5_1_staged.py --seeds 0-9
"""

import argparse, json, sys, time
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

from models.cosmobridge_v5_1 import CrossAttentionFusion, DescriptorPathway

PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def load_cached(split):
    d = np.load(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz", allow_pickle=True)
    return {k: torch.from_numpy(d[k]).float() if d[k].dtype.kind == 'f' else d[k] for k in d.keys()}


def compute_metrics(p, t):
    m = {}
    for i, n in enumerate(PROPS):
        ss_r = ((t[:,i]-p[:,i])**2).sum(); ss_t = ((t[:,i]-t[:,i].mean())**2).sum()
        m[f"{n}_r2"] = (1-ss_r/(ss_t+1e-8)).item()
    m["avg_r2"] = np.mean(list(m.values()))
    return m


class PathAModel(nn.Module):
    """Path A: Cross-attention fusion + thermo → 7 predictions."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.fusion = CrossAttentionFusion(300, 256, 256, 4, dropout)
        self.head = nn.Sequential(
            nn.Linear(256+25, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 7))
    def forward(self, g, s, t):
        f = self.fusion(g, s)
        return self.head(torch.cat([f, t], -1))


class PathCModel(nn.Module):
    """Path C: Descriptor pathway + T-modulation → 7 predictions."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.pathway = DescriptorPathway(20, 5, 64, 7, dropout)
    def forward(self, thermo):
        return self.pathway(thermo[:, 5:], thermo[:, :5])


def pretrain_path(model, train_data, val_data, device, name,
                  lr=1e-3, epochs=300, patience=40):
    """Pre-train a single path model."""
    print(f"\n  Pre-training {name}...")
    params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(train_data["chemprop_fp"], train_data["surface_fp"],
                              train_data["thermo_feat"], train_data["targets"])
    val_ds = TensorDataset(val_data["chemprop_fp"], val_data["surface_fp"],
                            val_data["thermo_feat"], val_data["targets"])
    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ldr = DataLoader(val_ds, batch_size=64)

    best_val, best_state, no_imp = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for g, s, t, y in train_ldr:
            g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
            if name == "Path C":
                preds = model(t)
            else:
                preds = model(g, s, t)
            loss = ((preds - y) ** 2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for g, s, t, y in val_ldr:
                g, s, t, y = g.to(device), s.to(device), t.to(device), y.to(device)
                p = model(t) if name == "Path C" else model(g, s, t)
                val_loss += ((p - y) ** 2).mean().item()
        val_loss /= len(val_ldr)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"    Early stop at {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    # Generate frozen predictions for all splits
    preds = {}
    for split_name, data in [("train", train_data), ("val", val_data)]:
        with torch.no_grad():
            g = data["chemprop_fp"].to(device)
            s = data["surface_fp"].to(device)
            t = data["thermo_feat"].to(device)
            if name == "Path C":
                p = model(t)
            else:
                p = model(g, s, t)
            preds[split_name] = p.cpu()

    m = compute_metrics(preds["train"].numpy(), train_data["targets"].numpy())
    print(f"    {name} train R²={m['avg_r2']:.4f}")

    return model, preds


class ThreeWayRouter(nn.Module):
    """Router: blends 3 frozen path predictions."""
    def __init__(self, input_dim=581, hidden=64, n_props=7, dropout=0.3):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(hidden),
            nn.Linear(hidden, n_props * 3),
        )
        self.n_props = n_props
        # Init: fusion and chemprop dominant
        with torch.no_grad():
            self.router[-1].weight.zero_()
            bias = torch.zeros(n_props * 3)
            for p in range(n_props):
                bias[p*3:p*3+3] = torch.tensor([0.5, 0.5, -0.5])
            self.router[-1].bias.copy_(bias)

    def forward(self, features, preds_a, preds_b, preds_c):
        logits = self.router(features).view(-1, self.n_props, 3)
        weights = torch.softmax(logits, dim=-1)
        paths = torch.stack([preds_a, preds_b, preds_c], dim=-1)
        return (paths * weights).sum(-1), weights, logits


def train_router(train_data, val_data, test_data, device, config, seed):
    """Train 3-way router on frozen predictions (same as v4 protocol)."""
    set_seed(seed)
    model = ThreeWayRouter(
        input_dim=300+256+25, hidden=config["hidden"],
        n_props=7, dropout=config["dropout"]).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Router: {params:,} params")

    optimizer = AdamW(model.parameters(), lr=config["lr"],
                      weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    train_ds = TensorDataset(
        train_data["features"], train_data["preds_a"],
        train_data["preds_b"], train_data["preds_c"], train_data["targets"])
    val_ds = TensorDataset(
        val_data["features"], val_data["preds_a"],
        val_data["preds_b"], val_data["preds_c"], val_data["targets"])

    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ldr = DataLoader(val_ds, batch_size=64)

    best_val, best_state, no_imp = float("inf"), None, 0

    for epoch in range(1, config["epochs"] + 1):
        anchor_w = config["anchor_init"] * (1-epoch/config["epochs"]) + \
                   config["anchor_final"] * (epoch/config["epochs"])

        model.train()
        for feat, pa, pb, pc, y in train_ldr:
            feat, pa, pb, pc, y = [x.to(device) for x in [feat, pa, pb, pc, y]]
            preds, weights, logits = model(feat, pa, pb, pc)
            mse = ((preds - y) ** 2).mean()
            # Anchor regularization
            init = model.router[-1].bias.detach()
            anchor = ((logits.view(-1, 21) - init.unsqueeze(0)) ** 2).mean()
            loss = mse + anchor_w * anchor
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        model.eval()
        vl = 0
        with torch.no_grad():
            for feat, pa, pb, pc, y in val_ldr:
                feat, pa, pb, pc, y = [x.to(device) for x in [feat, pa, pb, pc, y]]
                p, _, _ = model(feat, pa, pb, pc)
                vl += ((p-y)**2).mean().item()
        vl /= len(val_ldr)

        if vl < best_val:
            best_val = vl; best_state = {k:v.clone() for k,v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= config["patience"]: break

    model.load_state_dict(best_state)
    model.eval()

    # Test
    with torch.no_grad():
        feat = test_data["features"].to(device)
        pa = test_data["preds_a"].to(device)
        pb = test_data["preds_b"].to(device)
        pc = test_data["preds_c"].to(device)
        preds, weights, _ = model(feat, pa, pb, pc)

    tm = compute_metrics(preds.cpu().numpy(), test_data["targets"].numpy())
    w = weights.cpu().mean(0).numpy()

    return tm, preds.cpu().numpy(), w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    s, e = map(int, args.seeds.split("-"))
    seeds = list(range(s, e + 1))

    print("COSMOBridge v5.1 STAGED: Pre-train paths → Freeze → Train router")

    train_data = load_cached("train")
    val_data = load_cached("val")
    test_data = load_cached("test")

    # ── Stage 1: Pre-train Path A (cross-attention fusion) ──
    path_a = PathAModel(dropout=0.3).to(device)
    path_a, preds_a = pretrain_path(path_a, train_data, val_data, device, "Path A")

    # ── Stage 2: Pre-train Path C (descriptors) ──
    path_c = PathCModel(dropout=0.3).to(device)
    path_c, preds_c = pretrain_path(path_c, train_data, val_data, device, "Path C")

    # ── Generate frozen predictions for all splits ──
    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        with torch.no_grad():
            g = data["chemprop_fp"].to(device)
            s = data["surface_fp"].to(device)
            t = data["thermo_feat"].to(device)
            data["preds_a"] = path_a(g, s, t).cpu()
            data["preds_c"] = path_c(t).cpu()
        # Path B: already cached from v4
        data["preds_b"] = data["preds_chemprop"]
        data["features"] = torch.cat([data["chemprop_fp"], data["surface_fp"],
                                       data["thermo_feat"]], dim=-1)

    # Individual path performance
    for name, key in [("Path A (fusion)", "preds_a"), ("Path B (chemprop)", "preds_b"),
                       ("Path C (desc)", "preds_c")]:
        m = compute_metrics(test_data[key].numpy(), test_data["targets"].numpy())
        print(f"\n  {name} test: avg R²={m['avg_r2']:.4f}")
        for p in PROPS:
            print(f"    {p:8s}: {m[f'{p}_r2']:.4f}")

    # ── Stage 3: Train 3-way router ──
    config = {
        "hidden": 64, "dropout": 0.3, "lr": 1e-3,
        "weight_decay": 1e-3, "batch_size": 32,
        "epochs": 300, "patience": 40,
        "anchor_init": 0.1, "anchor_final": 0.01,
    }

    all_metrics = []
    for seed in seeds:
        print(f"\n=== Router Seed {seed} ===")
        tm, preds, weights = train_router(train_data, val_data, test_data,
                                           device, config, seed)
        print(f"  avg R²: {tm['avg_r2']:.4f}")
        for i, p in enumerate(PROPS):
            print(f"    {p:8s}: R²={tm[f'{p}_r2']:.4f}  "
                  f"[A={weights[i,0]:.3f} B={weights[i,1]:.3f} C={weights[i,2]:.3f}]")
        all_metrics.append(tm)

        pred_dir = V5_ROOT / "results/v5_1_staged/seed_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez(pred_dir / f"seed_{seed}.npz",
                 predictions=preds, targets=test_data["targets"].numpy())

    print(f"\n{'='*60}")
    print("v5.1 STAGED SUMMARY")
    print(f"{'='*60}")
    avgs = [m["avg_r2"] for m in all_metrics]
    print(f"  avg R²: {np.mean(avgs):.4f} ± {np.std(avgs):.4f}")
    print(f"  v4 router: 0.8078 ± 0.0003")
    print(f"  Delta: {np.mean(avgs)-0.8078:+.4f}")
    for p in PROPS:
        vals = [m[f"{p}_r2"] for m in all_metrics]
        print(f"  {p:8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    out = V5_ROOT / "results/v5_1_staged"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({"per_seed": all_metrics,
                    "avg": float(np.mean(avgs)),
                    "std": float(np.std(avgs))}, f, indent=2)


if __name__ == "__main__":
    main()
