#!/usr/bin/env python3
"""v4 + Descriptor Pathway: Add descriptors as 3rd path to ACTUAL v4 predictions.

Uses the EXACT v4 frozen predictions (preds_fusion, preds_chemprop) from the paper,
and adds a 3rd path: surface descriptors with T-modulation.

This is the cleanest test of whether descriptors add value on top of v4.

Usage:
    python train_v4_plus_descriptors.py --seeds 0-9
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent

PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]

# v3 gate values from the paper (Table 2)
V3_GATES = torch.tensor([0.36, 0.39, 0.36, 0.42, 0.45, 0.37, 0.69])


class DescriptorHead(nn.Module):
    """Path C: Surface descriptors + T-modulation → 7 predictions.

    Pre-trained separately, then frozen for the router.
    """
    def __init__(self, desc_dim=20, thermo_dim=5, hidden=64, n_props=7, dropout=0.3):
        super().__init__()
        self.temp_gate = nn.Sequential(
            nn.Linear(thermo_dim, 32), nn.GELU(),
            nn.Linear(32, desc_dim), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(desc_dim + thermo_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_props))

    def forward(self, thermo_feat):
        desc = thermo_feat[:, 5:]   # 20D surface descriptors
        temp = thermo_feat[:, :5]   # T, x1, 1/T, T², T³
        modulated = desc * self.temp_gate(temp)
        return self.head(torch.cat([modulated, temp], -1))


class ThreePathRouter(nn.Module):
    """3-way router over frozen Path A (fusion) + Path B (chemprop) + Path C (descriptors)."""
    def __init__(self, input_dim=581, hidden=64, n_props=7, dropout=0.3):
        super().__init__()
        self.n_props = n_props
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(hidden),
            nn.Linear(hidden, n_props * 3))

        # Init: start near v4 gate values, Path C low
        with torch.no_grad():
            self.router[-1].weight.zero_()
            bias = torch.zeros(n_props * 3)
            for p in range(n_props):
                a = V3_GATES[p].item()
                bias[p*3 + 0] = np.log(a + 1e-8)       # fusion
                bias[p*3 + 1] = np.log(1-a + 1e-8)     # chemprop
                bias[p*3 + 2] = -2.0                     # descriptors (low)
            self.router[-1].bias.copy_(bias)

    def forward(self, features, pa, pb, pc):
        logits = self.router(features).view(-1, self.n_props, 3)
        w = torch.softmax(logits, dim=-1)
        paths = torch.stack([pa, pb, pc], dim=-1)
        return (paths * w).sum(-1), w, logits


def compute_metrics(p, t):
    m = {}
    for i, n in enumerate(PROPS):
        ss_r = ((t[:,i]-p[:,i])**2).sum()
        ss_t = ((t[:,i]-t[:,i].mean())**2).sum()
        m[f"{n}_r2"] = (1-ss_r/(ss_t+1e-8)).item()
    m["avg_r2"] = np.mean(list(m.values()))
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    s, e = map(int, args.seeds.split("-"))
    seeds = list(range(s, e + 1))

    print("v4 + Descriptor Pathway")
    print("  Path A: ACTUAL v4 frozen fusion predictions")
    print("  Path B: ACTUAL v4 frozen chemprop predictions")
    print("  Path C: NEW descriptor head (T-modulated, pre-trained)")
    print(f"  Seeds: {seeds}")

    # Load ACTUAL v4 cached data
    splits = {}
    for split in ["train", "val", "test"]:
        d = np.load(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz",
                     allow_pickle=True)
        splits[split] = {k: torch.from_numpy(d[k]).float() if d[k].dtype.kind == 'f'
                          else d[k] for k in d.keys()}

    # ── Step 1: Pre-train Path C (descriptor head) ──
    print("\nPre-training Path C (descriptor head)...")
    desc_model = DescriptorHead(dropout=0.3).to(device)
    opt = AdamW(desc_model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = CosineAnnealingLR(opt, T_max=300)

    train_ds = TensorDataset(splits["train"]["thermo_feat"], splits["train"]["targets"])
    val_ds = TensorDataset(splits["val"]["thermo_feat"], splits["val"]["targets"])
    train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ldr = DataLoader(val_ds, batch_size=64)

    best_val, best_state, no_imp = float("inf"), None, 0
    for epoch in range(1, 301):
        desc_model.train()
        for t, y in train_ldr:
            t, y = t.to(device), y.to(device)
            loss = ((desc_model(t) - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        desc_model.eval()
        vl = sum(((desc_model(t.to(device))-y.to(device))**2).mean().item()
                 for t,y in val_ldr) / len(val_ldr)
        if vl < best_val:
            best_val = vl; best_state = {k:v.clone() for k,v in desc_model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 40: break

    desc_model.load_state_dict(best_state)
    desc_model.eval()

    # Generate frozen Path C predictions
    for split in splits:
        with torch.no_grad():
            splits[split]["preds_desc"] = desc_model(
                splits[split]["thermo_feat"].to(device)).cpu()

    # Report individual path R²
    print("\nIndividual path performance on TEST:")
    for name, key in [("Path A (v4 fusion)", "preds_fusion"),
                       ("Path B (v4 chemprop)", "preds_chemprop"),
                       ("Path C (descriptors)", "preds_desc")]:
        m = compute_metrics(splits["test"][key].numpy(), splits["test"]["targets"].numpy())
        print(f"  {name}: avg R²={m['avg_r2']:.4f}")
        for p in PROPS:
            print(f"    {p:8s}: {m[f'{p}_r2']:.4f}")

    # Also show v4 2-way gate result
    v4_2way = V3_GATES.unsqueeze(0) * splits["test"]["preds_fusion"] + \
              (1-V3_GATES.unsqueeze(0)) * splits["test"]["preds_chemprop"]
    m_v4 = compute_metrics(v4_2way.numpy(), splits["test"]["targets"].numpy())
    print(f"\n  v4 (2-way, paper gates): avg R²={m_v4['avg_r2']:.4f}")

    # ── Step 2: Train 3-way router (v4 protocol) ──
    config = {
        "hidden": 64, "dropout": 0.3, "lr": 1e-3,
        "weight_decay": 1e-3, "epochs": 300, "patience": 40,
        "anchor_init": 0.1, "anchor_final": 0.01,
    }

    # Prepare router data
    for split in splits:
        d = splits[split]
        d["features"] = torch.cat([d["chemprop_fp"], d["surface_fp"], d["thermo_feat"]], -1)

    all_metrics = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n=== Router Seed {seed} ===")

        router = ThreePathRouter(
            input_dim=300+256+25, hidden=config["hidden"],
            n_props=7, dropout=config["dropout"]).to(device)
        print(f"  Router: {sum(p.numel() for p in router.parameters()):,} params")

        opt = AdamW(router.parameters(), lr=config["lr"],
                     weight_decay=config["weight_decay"])
        sched = CosineAnnealingLR(opt, T_max=config["epochs"])

        train_ds = TensorDataset(
            splits["train"]["features"], splits["train"]["preds_fusion"],
            splits["train"]["preds_chemprop"], splits["train"]["preds_desc"],
            splits["train"]["targets"])
        val_ds = TensorDataset(
            splits["val"]["features"], splits["val"]["preds_fusion"],
            splits["val"]["preds_chemprop"], splits["val"]["preds_desc"],
            splits["val"]["targets"])

        train_ldr = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_ldr = DataLoader(val_ds, batch_size=64)

        best_val, best_state, no_imp = float("inf"), None, 0
        for epoch in range(1, config["epochs"]+1):
            anchor_w = config["anchor_init"]*(1-epoch/config["epochs"]) + \
                       config["anchor_final"]*(epoch/config["epochs"])
            router.train()
            for feat, pa, pb, pc, y in train_ldr:
                feat,pa,pb,pc,y = [x.to(device) for x in [feat,pa,pb,pc,y]]
                preds, w, logits = router(feat, pa, pb, pc)
                mse = ((preds-y)**2).mean()
                anchor = ((logits.view(-1,21) - router.router[-1].bias.detach().unsqueeze(0))**2).mean()
                loss = mse + anchor_w * anchor
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

            router.eval()
            vl = 0
            with torch.no_grad():
                for feat,pa,pb,pc,y in val_ldr:
                    feat,pa,pb,pc,y = [x.to(device) for x in [feat,pa,pb,pc,y]]
                    p,_,_ = router(feat,pa,pb,pc)
                    vl += ((p-y)**2).mean().item()
            vl /= len(val_ldr)
            if vl < best_val:
                best_val = vl; best_state = {k:v.clone() for k,v in router.state_dict().items()}; no_imp = 0
            else:
                no_imp += 1
                if no_imp >= config["patience"]: break

        router.load_state_dict(best_state); router.eval()
        with torch.no_grad():
            feat = splits["test"]["features"].to(device)
            pa = splits["test"]["preds_fusion"].to(device)
            pb = splits["test"]["preds_chemprop"].to(device)
            pc = splits["test"]["preds_desc"].to(device)
            preds, w, _ = router(feat, pa, pb, pc)

        tm = compute_metrics(preds.cpu().numpy(), splits["test"]["targets"].numpy())
        wm = w.cpu().mean(0).numpy()

        print(f"  avg R²: {tm['avg_r2']:.4f}")
        for i, p in enumerate(PROPS):
            print(f"    {p:8s}: R²={tm[f'{p}_r2']:.4f}  "
                  f"[Fusion={wm[i,0]:.3f} Chemprop={wm[i,1]:.3f} Desc={wm[i,2]:.3f}]")
        all_metrics.append(tm)

        pred_dir = V5_ROOT / "results/v4_plus_desc/seed_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez(pred_dir / f"seed_{seed}.npz",
                 predictions=preds.cpu().numpy(),
                 targets=splits["test"]["targets"].numpy(),
                 weights=w.cpu().numpy())

    # Summary
    print(f"\n{'='*60}")
    print("v4 + DESCRIPTOR PATH SUMMARY")
    print(f"{'='*60}")
    avgs = [m["avg_r2"] for m in all_metrics]
    print(f"  avg R²: {np.mean(avgs):.4f} ± {np.std(avgs):.4f}")
    print(f"  v4 2-way (paper gates): {m_v4['avg_r2']:.4f}")
    print(f"  v4 3-way router (paper): 0.8078")
    print(f"  Delta vs 2-way: {np.mean(avgs)-m_v4['avg_r2']:+.4f}")
    print(f"  Delta vs 3-way: {np.mean(avgs)-0.8078:+.4f}")
    for p in PROPS:
        vals = [m[f"{p}_r2"] for m in all_metrics]
        print(f"  {p:8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    out = V5_ROOT / "results/v4_plus_desc"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({"per_seed": all_metrics, "avg": float(np.mean(avgs)),
                    "v4_2way": m_v4, "v4_3way_paper": 0.8078}, f, indent=2)


if __name__ == "__main__":
    main()
