#!/usr/bin/env python3
"""Complete Phase B: Uses V-JEPA features from 70 ILs for image residual.

Differs from train_phase_b.py by:
    - Using a NEW V-JEPA checkpoint trained on all 70 ILs
    - Computing fresh ViT features for ALL training ILs (including 42 new ones)
    - The image residual now has 70 unique molecular image features (vs 19 before)

Usage:
    python train_phase_b_complete.py --vjepa_checkpoint checkpoints/vjepa_70il/vit_pretrained_vjepa.pt --seeds 0-9
"""

import argparse, json, sys, pickle, time, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.decomposition import PCA
from rdkit import Chem
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5_ROOT))

PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def smiles_to_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def compute_metrics(p, t):
    m = {}
    for i, n in enumerate(PROPS):
        ss_r = ((t[:,i]-p[:,i])**2).sum()
        ss_t = ((t[:,i]-t[:,i].mean())**2).sum()
        m[f"{n}_r2"] = (1-ss_r/(ss_t+1e-8)).item()
    m["avg_r2"] = np.mean(list(m.values()))
    return m


def masked_mse(preds, targets, masks):
    m = masks.float(); n = m.sum()
    if n == 0: return torch.tensor(0.0, device=preds.device, requires_grad=True)
    return ((preds-targets)**2 * m).sum() / n


def extract_vit_features(smiles_list, vjepa_checkpoint, device):
    """Extract ViT features for a list of SMILES using V-JEPA encoder."""
    from models.multiview_vit import MultiViewViT

    vit = MultiViewViT(n_views=36, embed_dim=192, dropout=0.0).to(device)

    if vjepa_checkpoint and Path(vjepa_checkpoint).exists():
        state = torch.load(vjepa_checkpoint, map_location=device, weights_only=True)
        encoder_state = state.get("encoder_state_dict", {})
        if encoder_state:
            vit.load_state_dict(encoder_state, strict=False)
            print(f"  Loaded V-JEPA encoder from {Path(vjepa_checkpoint).name}")

    vit.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Find frame directories for each SMILES
    cosmo_dirs = [
        V5_ROOT / "data/cosmo_images",
        PROJECT_ROOT / "data/pipeline/cosmo_images",
    ]

    all_feats = []
    for smi in smiles_list:
        h = smiles_to_hash(smi)
        frames_dir = None
        for d in cosmo_dirs:
            candidate = d / f"{h}_frames"
            if candidate.exists() and len(list(candidate.glob("frame_*.png"))) >= 2:
                frames_dir = candidate
                break

        if frames_dir is None:
            # Try compound ID based lookup
            all_feats.append(np.zeros(192, dtype=np.float32))
            continue

        # Load 6 uniformly spaced views
        frames = sorted(frames_dir.glob("frame_*.png"))
        indices = np.linspace(0, len(frames)-1, 6, dtype=int)
        views = []
        for idx in indices:
            img = Image.open(frames[idx]).convert("RGB")
            views.append(transform(img))
        views = torch.stack(views).unsqueeze(0).to(device)  # (1, 6, 3, 224, 224)

        with torch.no_grad():
            emb, _ = vit.encode_views_chunked(views, chunk_size=3)
        all_feats.append(emb.cpu().numpy()[0])

    return np.array(all_feats, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vjepa_checkpoint", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    s, e = map(int, args.seeds.split("-"))
    seeds = list(range(s, e + 1))

    # Default V-JEPA checkpoint
    if args.vjepa_checkpoint is None:
        for p in [V5_ROOT / "checkpoints/vjepa_70il/vit_pretrained_vjepa.pt",
                   V5_ROOT / "checkpoints/vjepa/vit_pretrained_vjepa.pt"]:
            if p.exists():
                args.vjepa_checkpoint = str(p)
                break

    print("COMPLETE PHASE B: V-JEPA(70 ILs) + DAPT + v4 Router + Image Residual")
    print(f"  V-JEPA checkpoint: {args.vjepa_checkpoint}")
    print(f"  Seeds: {seeds}, Device: {device}")

    with open(PROJECT_ROOT / "data/processed/target_scaler.pkl", "rb") as f:
        tscaler = pickle.load(f)
    with open(PROJECT_ROOT / "data/processed/feature_scaler.pkl", "rb") as f:
        fscaler = pickle.load(f)

    # Load cached data
    train_c = np.load(PROJECT_ROOT / "cosmobridge_v4/data/cached_train.npz", allow_pickle=True)
    val_c = np.load(PROJECT_ROOT / "cosmobridge_v4/data/cached_val.npz", allow_pickle=True)
    test_c = np.load(PROJECT_ROOT / "cosmobridge_v4/data/cached_test.npz", allow_pickle=True)

    # Extract V-JEPA features for ALL splits using the 70-IL pre-trained encoder
    print("\nExtracting V-JEPA features for all splits...")
    for name, cached in [("train", train_c), ("val", val_c), ("test", test_c)]:
        smiles = list(cached["smiles"])
        feats = extract_vit_features(smiles, args.vjepa_checkpoint, device)
        out_path = V5_ROOT / f"data/cached_image_features_70il_{name}.npz"
        np.savez(out_path, vit_feat=feats)
        n_nonzero = (np.abs(feats).sum(1) > 0.01).sum()
        print(f"  {name}: {feats.shape}, {n_nonzero}/{len(feats)} with real images")

    # Load the 70-IL image features
    img_train = np.load(V5_ROOT / "data/cached_image_features_70il_train.npz")["vit_feat"]
    img_val = np.load(V5_ROOT / "data/cached_image_features_70il_val.npz")["vit_feat"]
    img_test = np.load(V5_ROOT / "data/cached_image_features_70il_test.npz")["vit_feat"]

    # PCA
    pca = PCA(n_components=20)
    pca.fit(img_train)
    img_train_pca = pca.transform(img_train).astype(np.float32)
    img_val_pca = pca.transform(img_val).astype(np.float32)
    img_test_pca = pca.transform(img_test).astype(np.float32)
    print(f"  PCA: 192D → 20D, explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # Build expanded training data (same as before)
    print("\nBuilding expanded dataset...")
    # [reuse build_expanded_data from train_phase_b.py]
    cached = train_c
    orig_graph = cached["chemprop_fp"].astype(np.float32)
    orig_surface = cached["surface_fp"].astype(np.float32)
    orig_thermo = cached["thermo_feat"].astype(np.float32)
    orig_targets = cached["targets"].astype(np.float32)
    orig_masks = np.ones((len(orig_targets), 7), dtype=bool)

    canon_to_idx = {}
    for i, smi in enumerate(cached["smiles"]):
        c = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        if c not in canon_to_idx: canon_to_idx[c] = i

    iltp = pd.read_csv(V5_ROOT / "data/ilthermopy_x05_filtered.csv")
    feats = np.load(V5_ROOT / "data/ilthermopy_chemprop_features.npz")
    smi_to_idx = {s: i for i, s in enumerate(feats["smiles"])}

    leaky = set()
    for d in [val_c, test_c]:
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
            g = feats["graph_feat"][smi_to_idx[smi]]; s = feats["surface_feat"][smi_to_idx[smi]]
        elif canon in canon_to_idx:
            g = orig_graph[canon_to_idx[canon]]; s = orig_surface[canon_to_idx[canon]]
        else: continue

        T = float(row.get("temperature", 298.15)); x1 = float(row.get("x1_water", 0.5))
        th = np.zeros(25, dtype=np.float32)
        th[0]=T; th[1]=x1; th[2]=1/T if T>0 else 0; th[3]=T**2; th[4]=T**3
        if canon in canon_to_idx: th[5:] = cached["thermo_feat"][canon_to_idx[canon], 5:]
        th_n = ((th-fscaler.mean_)/fscaler.scale_).astype(np.float32)
        th_n = np.nan_to_num(th_n, nan=0.0, posinf=0.0, neginf=0.0)
        if canon in canon_to_idx: th_n[5:] = cached["thermo_feat"][canon_to_idx[canon], 5:]

        target = np.zeros(7, dtype=np.float32); mask = np.zeros(7, dtype=bool)
        if pd.notna(row.get("gamma_water")):
            v = (float(row["gamma_water"])-tscaler.mean_[0])/tscaler.scale_[0]
            if abs(v)<5: target[0]=v; mask[0]=True
        if pd.notna(row.get("H_E")):
            v = (float(row["H_E"])-tscaler.mean_[3])/tscaler.scale_[3]
            if abs(v)<5: target[3]=v; mask[3]=True

        if mask.any():
            new_g.append(g.astype(np.float32)); new_s.append(s.astype(np.float32))
            new_t.append(th_n); new_tgt.append(target); new_m.append(mask)

    n_orig = len(orig_targets); n_new = len(new_g)
    exp_feats = np.concatenate([
        np.concatenate([orig_graph, orig_surface, orig_thermo], 1),
        np.concatenate([np.array(new_g), np.array(new_s), np.array(new_t)], 1)
    ]).astype(np.float32)
    exp_targets = np.concatenate([orig_targets, np.array(new_tgt)]).astype(np.float32)
    exp_masks = np.concatenate([orig_masks, np.array(new_m)])
    print(f"  Expanded: {n_orig} + {n_new} = {len(exp_targets)} samples")

    # v4 frozen predictions
    v4_router_preds = None
    pred_dir = PROJECT_ROOT / "cosmobridge_v4/results/seed_predictions"
    if pred_dir.exists():
        seed_files = sorted(pred_dir.glob("seed_*.npz"))
        if seed_files:
            preds_all = [np.load(f)["preds" if "preds" in np.load(f) else "predictions"] for f in seed_files]
            v4_router_preds = np.mean(preds_all, axis=0)
            print(f"  Loaded v4 router ensemble: {v4_router_preds.shape}")

    # Run seeds
    all_results = []
    for seed in seeds:
        set_seed(seed)
        print(f"\n{'#'*50}\n  SEED {seed}\n{'#'*50}")

        # Step 5: DAPT pre-train FFN
        print("  Step 5: DAPT on 70 ILs...")
        ffn = nn.Sequential(
            nn.Linear(581,256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128,7),
        ).to(device)

        dapt_ds = TensorDataset(torch.from_numpy(exp_feats), torch.from_numpy(exp_targets), torch.from_numpy(exp_masks))
        dapt_ldr = DataLoader(dapt_ds, batch_size=64, shuffle=True, drop_last=True)

        val_feats_t = torch.from_numpy(np.concatenate([val_c["chemprop_fp"], val_c["surface_fp"], val_c["thermo_feat"]], 1).astype(np.float32))
        val_tgt_t = torch.from_numpy(val_c["targets"].astype(np.float32))
        val_ldr = DataLoader(TensorDataset(val_feats_t, val_tgt_t), batch_size=64)

        opt = AdamW(ffn.parameters(), lr=2e-3, weight_decay=1e-3)
        sched = CosineAnnealingLR(opt, T_max=80)
        bv, bs, p = float("inf"), None, 0
        for ep in range(80):
            ffn.train()
            for f,t,m in dapt_ldr:
                f,t,m=f.to(device),t.to(device),m.to(device)
                loss=masked_mse(ffn(f),t,m); opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ffn.parameters(),1.0); opt.step()
            sched.step()
            ffn.eval()
            vl=sum(((ffn(f.to(device))-t.to(device))**2).mean().item() for f,t in val_ldr)/len(val_ldr)
            if vl<bv: bv=vl; bs={k:v.clone() for k,v in ffn.state_dict().items()}; p=0
            else:
                p+=1
                if p>=20: break

        # Step 5b: Fine-tune on original
        print("  Step 5b: Fine-tune on 28 ILs...")
        ffn.load_state_dict(bs)
        ft_ldr = DataLoader(TensorDataset(torch.from_numpy(exp_feats[:n_orig]), torch.from_numpy(exp_targets[:n_orig])), batch_size=32, shuffle=True)
        opt2 = AdamW(ffn.parameters(), lr=5e-4, weight_decay=1e-3)
        sched2 = CosineAnnealingLR(opt2, T_max=200)
        bv2, bs2, p2 = float("inf"), None, 0
        for ep in range(200):
            ffn.train()
            for f,t in ft_ldr:
                f,t=f.to(device),t.to(device)
                loss=((ffn(f)-t)**2).mean(); opt2.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ffn.parameters(),1.0); opt2.step()
            sched2.step()
            ffn.eval()
            vl=sum(((ffn(f.to(device))-t.to(device))**2).mean().item() for f,t in val_ldr)/len(val_ldr)
            if vl<bv2: bv2=vl; bs2={k:v.clone() for k,v in ffn.state_dict().items()}; p2=0
            else:
                p2+=1
                if p2>=30: break
        ffn.load_state_dict(bs2); ffn.eval()

        test_feats_t = torch.from_numpy(np.concatenate([test_c["chemprop_fp"], test_c["surface_fp"], test_c["thermo_feat"]], 1).astype(np.float32))
        with torch.no_grad():
            dapt_test = ffn(test_feats_t.to(device)).cpu().numpy()
            dapt_train = ffn(torch.from_numpy(exp_feats[:n_orig]).to(device)).cpu().numpy()

        m_dapt = compute_metrics(dapt_test, test_c["targets"])
        print(f"    DAPT FFN: avg R²={m_dapt['avg_r2']:.4f}")

        # Step 6: v4 routing (fusion + DAPT-FFN)
        print("  Step 6: v4-style routing...")
        class Gates(nn.Module):
            def __init__(self):
                super().__init__()
                init=torch.tensor([0.36,0.39,0.36,0.42,0.45,0.37,0.69])
                self.logits=nn.Parameter(torch.log(init/(1-init)))
            def forward(self,a,b): return torch.sigmoid(self.logits)*a+(1-torch.sigmoid(self.logits))*b

        gates = Gates().to(device)
        train_pa = torch.from_numpy(train_c["preds_fusion"].astype(np.float32))
        train_pb = torch.from_numpy(dapt_train.astype(np.float32))
        train_y = torch.from_numpy(train_c["targets"].astype(np.float32))
        g_ldr = DataLoader(TensorDataset(train_pa,train_pb,train_y), batch_size=32, shuffle=True)

        val_pa = torch.from_numpy(val_c["preds_fusion"].astype(np.float32))
        with torch.no_grad(): val_pb = ffn(val_feats_t.to(device)).cpu()
        g_val_ldr = DataLoader(TensorDataset(val_pa, val_pb, val_tgt_t), batch_size=64)

        go = AdamW(gates.parameters(), lr=0.1); gs = CosineAnnealingLR(go, T_max=200)
        bg, bgs, pg = float("inf"), None, 0
        for ep in range(200):
            gates.train()
            for a,b,y in g_ldr:
                a,b,y=a.to(device),b.to(device),y.to(device)
                loss=((gates(a,b)-y)**2).mean(); go.zero_grad(); loss.backward(); go.step()
            gs.step()
            gates.eval()
            vl=sum(((gates(a.to(device),b.to(device))-y.to(device))**2).mean().item() for a,b,y in g_val_ldr)/len(g_val_ldr)
            if vl<bg: bg=vl; bgs={k:v.clone() for k,v in gates.state_dict().items()}; pg=0
            else:
                pg+=1
                if pg>=40: break
        gates.load_state_dict(bgs); gates.eval()

        test_pa = torch.from_numpy(test_c["preds_fusion"].astype(np.float32))
        test_pb = torch.from_numpy(dapt_test.astype(np.float32))
        with torch.no_grad(): routed = gates(test_pa.to(device), test_pb.to(device)).cpu().numpy()
        train_routed = gates(train_pa.to(device), train_pb.to(device)).detach().cpu().numpy()

        m_routed = compute_metrics(routed, test_c["targets"])
        print(f"    Routed: avg R²={m_routed['avg_r2']:.4f}")

        # Step 7: Image residual with 70-IL V-JEPA features
        print("  Step 7: Image residual (70-IL V-JEPA features)...")
        class ResHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate=nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,20),nn.Sigmoid())
                self.head=nn.Sequential(nn.Linear(25,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(0.3),nn.Linear(32,7))
                self.alpha=nn.Parameter(torch.full((7,),-3.0))
                with torch.no_grad(): self.head[-1].weight.mul_(0.01); self.head[-1].bias.zero_()
            def forward(self,v,i,t):
                m=i*self.gate(t[:,:5]); r=self.head(torch.cat([m,t[:,:5]],-1))
                return v+torch.sigmoid(self.alpha)*r

        res = ResHead().to(device)
        ro = AdamW(res.parameters(), lr=5e-4, weight_decay=1e-2)
        rs = CosineAnnealingLR(ro, T_max=300)
        r_ldr = DataLoader(TensorDataset(
            torch.from_numpy(train_routed.astype(np.float32)),
            torch.from_numpy(img_train_pca),
            torch.from_numpy(train_c["thermo_feat"].astype(np.float32)),
            torch.from_numpy(train_c["targets"].astype(np.float32))), batch_size=32, shuffle=True)

        br, brs, prr = float("inf"), None, 0
        for ep in range(300):
            res.train()
            for v,i,t,y in r_ldr:
                v,i,t,y=[x.to(device) for x in [v,i,t,y]]
                loss=((res(v,i,t)-y)**2).mean(); ro.zero_grad(); loss.backward(); ro.step()
            rs.step()
            res.eval()
            with torch.no_grad():
                tl=((res(torch.from_numpy(train_routed.astype(np.float32)).to(device),
                         torch.from_numpy(img_train_pca).to(device),
                         torch.from_numpy(train_c["thermo_feat"].astype(np.float32)).to(device))-
                     torch.from_numpy(train_c["targets"].astype(np.float32)).to(device))**2).mean().item()
            if tl<br: br=tl; brs={k:v.clone() for k,v in res.state_dict().items()}; prr=0
            else:
                prr+=1
                if prr>=50: break

        res.load_state_dict(brs); res.eval()
        with torch.no_grad():
            final = res(torch.from_numpy(routed.astype(np.float32)).to(device),
                        torch.from_numpy(img_test_pca).to(device),
                        torch.from_numpy(test_c["thermo_feat"].astype(np.float32)).to(device)).cpu().numpy()

        m_final = compute_metrics(final, test_c["targets"])
        alpha_vals = torch.sigmoid(res.alpha).detach().cpu().numpy()

        # Also: apply on v4 router preds directly (Phase A style but with 70-IL V-JEPA)
        m_on_v4 = None
        if v4_router_preds is not None:
            res_v4 = ResHead().to(device)
            rv4o = AdamW(res_v4.parameters(), lr=5e-4, weight_decay=1e-2)
            rv4s = CosineAnnealingLR(rv4o, T_max=300)
            # Approximate v4 train preds
            v4_train_approx = 0.4*train_c["preds_fusion"]+0.6*train_c["preds_chemprop"]
            rv4_ldr = DataLoader(TensorDataset(
                torch.from_numpy(v4_train_approx.astype(np.float32)),
                torch.from_numpy(img_train_pca),
                torch.from_numpy(train_c["thermo_feat"].astype(np.float32)),
                torch.from_numpy(train_c["targets"].astype(np.float32))), batch_size=32, shuffle=True)
            brv, brsv, prrv = float("inf"), None, 0
            for ep in range(300):
                res_v4.train()
                for v,i,t,y in rv4_ldr:
                    v,i,t,y=[x.to(device) for x in [v,i,t,y]]
                    loss=((res_v4(v,i,t)-y)**2).mean(); rv4o.zero_grad(); loss.backward(); rv4o.step()
                rv4s.step()
                res_v4.eval()
                with torch.no_grad():
                    tl=((res_v4(torch.from_numpy(v4_train_approx.astype(np.float32)).to(device),
                                torch.from_numpy(img_train_pca).to(device),
                                torch.from_numpy(train_c["thermo_feat"].astype(np.float32)).to(device))-
                         torch.from_numpy(train_c["targets"].astype(np.float32)).to(device))**2).mean().item()
                if tl<brv: brv=tl; brsv={k:v.clone() for k,v in res_v4.state_dict().items()}; prrv=0
                else:
                    prrv+=1
                    if prrv>=50: break
            res_v4.load_state_dict(brsv); res_v4.eval()
            with torch.no_grad():
                v4_corrected = res_v4(torch.from_numpy(v4_router_preds.astype(np.float32)).to(device),
                                       torch.from_numpy(img_test_pca).to(device),
                                       torch.from_numpy(test_c["thermo_feat"].astype(np.float32)).to(device)).cpu().numpy()
            m_on_v4 = compute_metrics(v4_corrected, test_c["targets"])

        print(f"\n  SEED {seed} RESULTS:")
        print(f"    DAPT FFN:                 {m_dapt['avg_r2']:.4f}")
        print(f"    Routed (fusion+DAPT):     {m_routed['avg_r2']:.4f}")
        print(f"    + Image(70-IL V-JEPA):    {m_final['avg_r2']:.4f}")
        if m_on_v4:
            print(f"    v4 Router + Image(70-IL): {m_on_v4['avg_r2']:.4f}")

        result = {"dapt": m_dapt, "routed": m_routed, "final": m_final}
        if m_on_v4: result["v4_plus_70il_image"] = m_on_v4
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("COMPLETE PHASE B SUMMARY")
    print(f"{'='*60}")
    for key, label in [("dapt","DAPT FFN"), ("routed","Routed"), ("final","+ Image(70-IL)"), ("v4_plus_70il_image","v4 Router + Image(70-IL)")]:
        vals = [r[key]["avg_r2"] for r in all_results if key in r]
        if vals:
            print(f"  {label:30s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n  Phase A (v4 router + image 19 ILs): 0.816")
    print(f"  v4 paper:                           0.818")

    if "v4_plus_70il_image" in all_results[0]:
        vals = [r["v4_plus_70il_image"]["avg_r2"] for r in all_results]
        print(f"  v4 + Image(70-IL V-JEPA):           {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        print(f"\n  Per-property (v4 + 70-IL image):")
        for p in PROPS:
            pv = [r["v4_plus_70il_image"][f"{p}_r2"] for r in all_results]
            print(f"    {p:8s}: {np.mean(pv):.4f} ± {np.std(pv):.4f}")

    out = V5_ROOT / "results/phase_b_complete"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: {out}/summary.json")


if __name__ == "__main__":
    main()
