#!/usr/bin/env python3
"""Phase B: Expanded 70 ILs + DAPT + Image Residual.

Step 1: DAPT pre-train FFN on 70 ILs (expanded ILThermoPy data)
Step 2: Fine-tune on original 28 ILs → improved Path B predictions
Step 3: Run v4-style router with improved Path B + original Path A
Step 4: Add image residual + physics correction on top
Step 5: Compare with Phase A results (0.816)

Usage:
    python train_phase_b.py --seeds 0-9
"""

import argparse, json, sys, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.decomposition import PCA
from scipy.optimize import minimize_scalar
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5_ROOT))

PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def compute_metrics(p, t):
    m = {}
    for i, n in enumerate(PROPS):
        ss_r = ((t[:,i]-p[:,i])**2).sum()
        ss_t = ((t[:,i]-t[:,i].mean())**2).sum()
        m[f"{n}_r2"] = (1-ss_r/(ss_t+1e-8)).item()
    m["avg_r2"] = np.mean(list(m.values()))
    return m


def masked_mse(preds, targets, masks):
    m = masks.float()
    n = m.sum()
    if n == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    return ((preds - targets)**2 * m).sum() / n


def build_expanded_data(tscaler, fscaler):
    """Build expanded training set: original 28 ILs + ILThermoPy 42 new ILs."""
    cached = np.load(PROJECT_ROOT / "cosmobridge_v4/data/cached_train.npz", allow_pickle=True)
    orig_graph = cached["chemprop_fp"].astype(np.float32)
    orig_surface = cached["surface_fp"].astype(np.float32)
    orig_thermo = cached["thermo_feat"].astype(np.float32)
    orig_targets = cached["targets"].astype(np.float32)
    orig_masks = np.ones((len(orig_targets), 7), dtype=bool)
    orig_smiles = list(cached["smiles"])

    canon_to_idx = {}
    for i, smi in enumerate(orig_smiles):
        c = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        if c not in canon_to_idx:
            canon_to_idx[c] = i

    # Load ILThermoPy + features
    iltp = pd.read_csv(V5_ROOT / "data/ilthermopy_x05_filtered.csv")
    feats = np.load(V5_ROOT / "data/ilthermopy_chemprop_features.npz")
    smi_to_idx = {s: i for i, s in enumerate(feats["smiles"])}

    # Exclude val/test ILs
    leaky = set()
    for split in ["val", "test"]:
        d = np.load(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz", allow_pickle=True)
        for smi in d["smiles"]:
            leaky.add(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))

    new_g, new_s, new_t, new_tgt, new_m = [], [], [], [], []
    for _, row in iltp.iterrows():
        smi = row["il_smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        canon = Chem.MolToSmiles(mol)
        if canon in leaky: continue

        if smi in smi_to_idx:
            idx = smi_to_idx[smi]
            g = feats["graph_feat"][idx]; s = feats["surface_feat"][idx]
        elif canon in canon_to_idx:
            ref = canon_to_idx[canon]
            g = orig_graph[ref]; s = orig_surface[ref]
        else:
            continue

        T = float(row.get("temperature", 298.15))
        x1 = float(row.get("x1_water", 0.5))
        th = np.zeros(25, dtype=np.float32)
        th[0]=T; th[1]=x1; th[2]=1/T if T>0 else 0; th[3]=T**2; th[4]=T**3
        if canon in canon_to_idx:
            th[5:] = cached["thermo_feat"][canon_to_idx[canon], 5:]
        th_norm = ((th - fscaler.mean_) / fscaler.scale_).astype(np.float32)
        th_norm = np.nan_to_num(th_norm, nan=0.0, posinf=0.0, neginf=0.0)
        if canon in canon_to_idx:
            th_norm[5:] = cached["thermo_feat"][canon_to_idx[canon], 5:]

        target = np.zeros(7, dtype=np.float32)
        mask = np.zeros(7, dtype=bool)
        if pd.notna(row.get("gamma_water")):
            v = (float(row["gamma_water"]) - tscaler.mean_[0]) / tscaler.scale_[0]
            if abs(v) < 5: target[0]=v; mask[0]=True
        if pd.notna(row.get("H_E")):
            v = (float(row["H_E"]) - tscaler.mean_[3]) / tscaler.scale_[3]
            if abs(v) < 5: target[3]=v; mask[3]=True

        if mask.any():
            new_g.append(g.astype(np.float32)); new_s.append(s.astype(np.float32))
            new_t.append(th_norm); new_tgt.append(target); new_m.append(mask)

    if new_g:
        return {
            "graph": np.concatenate([orig_graph, np.array(new_g)]),
            "surface": np.concatenate([orig_surface, np.array(new_s)]),
            "thermo": np.concatenate([orig_thermo, np.array(new_t)]),
            "targets": np.concatenate([orig_targets, np.array(new_tgt)]),
            "masks": np.concatenate([orig_masks, np.array(new_m)]),
            "n_orig": len(orig_smiles), "n_new": len(new_g),
        }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    s, e = map(int, args.seeds.split("-"))
    seeds = list(range(s, e + 1))

    print("PHASE B: DAPT on 70 ILs → v4 Router + Image Residual")
    print(f"  Seeds: {seeds}, Device: {device}")

    with open(PROJECT_ROOT / "data/processed/target_scaler.pkl", "rb") as f:
        tscaler = pickle.load(f)
    with open(PROJECT_ROOT / "data/processed/feature_scaler.pkl", "rb") as f:
        fscaler = pickle.load(f)

    # Build expanded data
    print("\nBuilding expanded dataset...")
    expanded = build_expanded_data(tscaler, fscaler)
    if expanded is None:
        print("ERROR: Could not build expanded dataset")
        return
    print(f"  {expanded['n_orig']} original + {expanded['n_new']} new = "
          f"{len(expanded['targets'])} total")

    # Load val/test
    val_c = np.load(PROJECT_ROOT / "cosmobridge_v4/data/cached_val.npz", allow_pickle=True)
    test_c = np.load(PROJECT_ROOT / "cosmobridge_v4/data/cached_test.npz", allow_pickle=True)

    # Load original v4 frozen path predictions
    train_c = np.load(PROJECT_ROOT / "cosmobridge_v4/data/cached_train.npz", allow_pickle=True)

    # Image features
    img_train = np.load(V5_ROOT / "data/cached_image_features_train.npz")["vit_feat"]
    img_test = np.load(V5_ROOT / "data/cached_image_features_test.npz")["vit_feat"]
    pca = PCA(n_components=20)
    pca.fit(img_train)
    img_train_pca = pca.transform(img_train).astype(np.float32)
    img_test_pca = pca.transform(img_test).astype(np.float32)

    all_results = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'#'*50}")
        print(f"  SEED {seed}")
        print(f"{'#'*50}")

        # ── Step 1: DAPT Pre-train FFN on expanded 70 ILs ──
        print("  Step 1: DAPT pre-training on 70 ILs...")
        ffn = nn.Sequential(
            nn.Linear(581, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 7),
        ).to(device)

        exp_feats = torch.from_numpy(np.concatenate([
            expanded["graph"], expanded["surface"], expanded["thermo"]], axis=1).astype(np.float32))
        exp_targets = torch.from_numpy(expanded["targets"].astype(np.float32))
        exp_masks = torch.from_numpy(expanded["masks"])

        dapt_ds = TensorDataset(exp_feats, exp_targets, exp_masks)
        dapt_ldr = DataLoader(dapt_ds, batch_size=64, shuffle=True, drop_last=True)

        val_feats = torch.from_numpy(np.concatenate([
            val_c["chemprop_fp"], val_c["surface_fp"], val_c["thermo_feat"]], axis=1).astype(np.float32))
        val_targets = torch.from_numpy(val_c["targets"].astype(np.float32))
        val_ds = TensorDataset(val_feats, val_targets)
        val_ldr = DataLoader(val_ds, batch_size=64)

        opt = AdamW(ffn.parameters(), lr=2e-3, weight_decay=1e-3)
        sched = CosineAnnealingLR(opt, T_max=80)
        best_val, best_state, patience = float("inf"), None, 0

        for ep in range(80):
            ffn.train()
            for f, t, m in dapt_ldr:
                f,t,m = f.to(device), t.to(device), m.to(device)
                loss = masked_mse(ffn(f), t, m)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ffn.parameters(), 1.0)
                opt.step()
            sched.step()
            ffn.eval()
            vl = sum(((ffn(f.to(device))-t.to(device))**2).mean().item() for f,t in val_ldr)/len(val_ldr)
            if vl < best_val:
                best_val=vl; best_state={k:v.clone() for k,v in ffn.state_dict().items()}; patience=0
            else:
                patience += 1
                if patience >= 20: break

        # ── Step 2: Fine-tune on original 28 ILs ──
        print("  Step 2: Fine-tuning on original 28 ILs...")
        ffn.load_state_dict(best_state)
        n_orig = expanded["n_orig"]
        orig_feats = exp_feats[:n_orig]
        orig_targets_t = exp_targets[:n_orig]

        ft_ds = TensorDataset(orig_feats, orig_targets_t)
        ft_ldr = DataLoader(ft_ds, batch_size=32, shuffle=True)

        opt2 = AdamW(ffn.parameters(), lr=5e-4, weight_decay=1e-3)
        sched2 = CosineAnnealingLR(opt2, T_max=200)
        best_val2, best_state2, patience2 = float("inf"), None, 0

        for ep in range(200):
            ffn.train()
            for f, t in ft_ldr:
                f,t = f.to(device), t.to(device)
                loss = ((ffn(f)-t)**2).mean()
                opt2.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ffn.parameters(), 1.0)
                opt2.step()
            sched2.step()
            ffn.eval()
            vl = sum(((ffn(f.to(device))-t.to(device))**2).mean().item() for f,t in val_ldr)/len(val_ldr)
            if vl < best_val2:
                best_val2=vl; best_state2={k:v.clone() for k,v in ffn.state_dict().items()}; patience2=0
            else:
                patience2 += 1
                if patience2 >= 30: break

        ffn.load_state_dict(best_state2)
        ffn.eval()

        # Generate DAPT-improved Path B predictions
        test_feats = torch.from_numpy(np.concatenate([
            test_c["chemprop_fp"], test_c["surface_fp"], test_c["thermo_feat"]], axis=1).astype(np.float32))
        train_feats_orig = torch.from_numpy(np.concatenate([
            train_c["chemprop_fp"], train_c["surface_fp"], train_c["thermo_feat"]], axis=1).astype(np.float32))

        with torch.no_grad():
            dapt_preds_test = ffn(test_feats.to(device)).cpu().numpy()
            dapt_preds_train = ffn(train_feats_orig.to(device)).cpu().numpy()

        m_dapt = compute_metrics(dapt_preds_test, test_c["targets"])
        print(f"    DAPT FFN alone: avg R²={m_dapt['avg_r2']:.4f}")

        # ── Step 3: v4-style router with Path A (original fusion) + Path B (DAPT FFN) ──
        print("  Step 3: v4-style 2-path routing (fusion + DAPT-FFN)...")
        # Path A = original fusion predictions (frozen from v4)
        # Path B = DAPT-improved FFN predictions

        class TwoPathGates(nn.Module):
            def __init__(self):
                super().__init__()
                init = torch.tensor([0.36, 0.39, 0.36, 0.42, 0.45, 0.37, 0.69])
                self.logits = nn.Parameter(torch.log(init/(1-init)))
            def forward(self, pa, pb):
                a = torch.sigmoid(self.logits)
                return a*pa + (1-a)*pb

        gates = TwoPathGates().to(device)
        gate_opt = AdamW(gates.parameters(), lr=0.1)
        gate_sched = CosineAnnealingLR(gate_opt, T_max=200)

        train_pa = torch.from_numpy(train_c["preds_fusion"].astype(np.float32))
        train_pb = torch.from_numpy(dapt_preds_train.astype(np.float32))
        train_y = torch.from_numpy(train_c["targets"].astype(np.float32))

        val_pa = torch.from_numpy(val_c["preds_fusion"].astype(np.float32))
        val_pb_np = ffn(val_feats.to(device)).detach().cpu()
        val_pb = val_pb_np.float()

        gate_ds = TensorDataset(train_pa, train_pb, train_y)
        gate_ldr = DataLoader(gate_ds, batch_size=32, shuffle=True)
        gate_val_ds = TensorDataset(val_pa, val_pb, val_targets)
        gate_val_ldr = DataLoader(gate_val_ds, batch_size=64)

        best_vg, best_gs, pg = float("inf"), None, 0
        for ep in range(200):
            gates.train()
            for pa,pb,y in gate_ldr:
                pa,pb,y = pa.to(device),pb.to(device),y.to(device)
                loss = ((gates(pa,pb)-y)**2).mean()
                gate_opt.zero_grad(); loss.backward(); gate_opt.step()
            gate_sched.step()
            gates.eval()
            vl = sum(((gates(pa.to(device),pb.to(device))-y.to(device))**2).mean().item()
                     for pa,pb,y in gate_val_ldr)/len(gate_val_ldr)
            if vl < best_vg:
                best_vg=vl; best_gs={k:v.clone() for k,v in gates.state_dict().items()}; pg=0
            else:
                pg+=1
                if pg>=40: break

        gates.load_state_dict(best_gs)
        gates.eval()

        test_pa = torch.from_numpy(test_c["preds_fusion"].astype(np.float32))
        test_pb = torch.from_numpy(dapt_preds_test.astype(np.float32))
        with torch.no_grad():
            routed = gates(test_pa.to(device), test_pb.to(device)).cpu().numpy()

        m_routed = compute_metrics(routed, test_c["targets"])
        gate_vals = torch.sigmoid(gates.logits).detach().cpu().numpy()
        print(f"    Routed (fusion+DAPT): avg R²={m_routed['avg_r2']:.4f}")
        print(f"    Gate values: {dict(zip(PROPS, [f'{g:.3f}' for g in gate_vals]))}")

        # ── Step 4: Image residual on routed predictions ──
        print("  Step 4: Image residual correction...")

        class ResHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,20),nn.Sigmoid())
                self.head = nn.Sequential(nn.Linear(25,32),nn.LayerNorm(32),nn.GELU(),
                                           nn.Dropout(0.3),nn.Linear(32,7))
                self.alpha = nn.Parameter(torch.full((7,),-3.0))
                with torch.no_grad(): self.head[-1].weight.mul_(0.01); self.head[-1].bias.zero_()
            def forward(self, v4p, img, th):
                mod = img * self.gate(th[:,:5])
                res = self.head(torch.cat([mod, th[:,:5]],-1))
                return v4p + torch.sigmoid(self.alpha)*res

        # Train image residual on routed predictions
        train_routed = gates(train_pa.to(device), train_pb.to(device)).detach().cpu().numpy()

        res_model = ResHead().to(device)
        res_opt = AdamW(res_model.parameters(), lr=5e-4, weight_decay=1e-2)
        res_sched = CosineAnnealingLR(res_opt, T_max=300)

        res_ds = TensorDataset(
            torch.from_numpy(train_routed.astype(np.float32)),
            torch.from_numpy(img_train_pca),
            torch.from_numpy(train_c["thermo_feat"].astype(np.float32)),
            torch.from_numpy(train_c["targets"].astype(np.float32)))
        res_ldr = DataLoader(res_ds, batch_size=32, shuffle=True)

        best_rl, best_rs, pr = float("inf"), None, 0
        for ep in range(300):
            res_model.train()
            for v4p,img,th,y in res_ldr:
                v4p,img,th,y = [x.to(device) for x in [v4p,img,th,y]]
                loss = ((res_model(v4p,img,th)-y)**2).mean()
                res_opt.zero_grad(); loss.backward(); res_opt.step()
            res_sched.step()
            res_model.eval()
            with torch.no_grad():
                tl = ((res_model(
                    torch.from_numpy(train_routed.astype(np.float32)).to(device),
                    torch.from_numpy(img_train_pca).to(device),
                    torch.from_numpy(train_c["thermo_feat"].astype(np.float32)).to(device)
                ) - torch.from_numpy(train_c["targets"].astype(np.float32)).to(device))**2).mean().item()
            if tl < best_rl:
                best_rl=tl; best_rs={k:v.clone() for k,v in res_model.state_dict().items()}; pr=0
            else:
                pr+=1
                if pr>=50: break

        res_model.load_state_dict(best_rs)
        res_model.eval()
        with torch.no_grad():
            final = res_model(
                torch.from_numpy(routed.astype(np.float32)).to(device),
                torch.from_numpy(img_test_pca).to(device),
                torch.from_numpy(test_c["thermo_feat"].astype(np.float32)).to(device),
            ).cpu().numpy()

        m_final = compute_metrics(final, test_c["targets"])
        img_alpha = torch.sigmoid(res_model.alpha).detach().cpu().numpy()

        print(f"\n  SEED {seed} RESULTS:")
        print(f"    DAPT FFN alone:           avg R²={m_dapt['avg_r2']:.4f}")
        print(f"    Routed (fusion + DAPT):   avg R²={m_routed['avg_r2']:.4f}")
        print(f"    + Image residual:         avg R²={m_final['avg_r2']:.4f}")
        print(f"    Image alpha: {dict(zip(PROPS, [f'{a:.3f}' for a in img_alpha]))}")

        for p in PROPS:
            print(f"      {p:8s}: {m_final[f'{p}_r2']:.4f}")

        all_results.append({
            "dapt_ffn": m_dapt, "routed": m_routed, "final": m_final,
            "gate_values": gate_vals.tolist(), "image_alpha": img_alpha.tolist(),
        })

    # Summary
    print(f"\n{'='*60}")
    print("PHASE B SUMMARY")
    print(f"{'='*60}")

    for stage, key in [("DAPT FFN alone", "dapt_ffn"),
                        ("Routed (fusion+DAPT)", "routed"),
                        ("+ Image residual", "final")]:
        avgs = [r[key]["avg_r2"] for r in all_results]
        print(f"  {stage:30s}: {np.mean(avgs):.4f} ± {np.std(avgs):.4f}")

    print(f"\n  Phase A (v4 router + image): 0.816")
    print(f"  v4 paper:                    0.818")

    final_avgs = [r["final"]["avg_r2"] for r in all_results]
    print(f"  Phase B (DAPT + image):      {np.mean(final_avgs):.4f} ± {np.std(final_avgs):.4f}")

    print(f"\n  Per-property (Phase B final):")
    for p in PROPS:
        vals = [r["final"][f"{p}_r2"] for r in all_results]
        print(f"    {p:8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    out = V5_ROOT / "results/phase_b"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({"all_results": all_results,
                    "final_avg": float(np.mean(final_avgs)),
                    "final_std": float(np.std(final_avgs))}, f, indent=2, default=float)
    print(f"\nSaved: {out}/summary.json")


if __name__ == "__main__":
    main()
