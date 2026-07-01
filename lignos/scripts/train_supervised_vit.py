#!/usr/bin/env python3
"""Train Multi-Task Supervised ViT + apply as image residual on v4.

Phase 1: Train ViT with joint V-JEPA + property prediction + contrastive
Phase 2: Extract features, PCA, apply as residual on v4 router
Phase 3: Compare with V-JEPA-only features

Usage:
    python train_supervised_vit.py --epochs 200 --seeds 0-9
"""

import argparse, json, sys, pickle, hashlib, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from PIL import Image
from torchvision import transforms
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5_ROOT))

from models.multiview_vit import PatchEmbedding, ViTBlock
from models.supervised_vit import SupervisedViT

PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


def smiles_to_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


class ILViewPropertyDataset(Dataset):
    """Dataset yielding molecule views + property labels."""

    def __init__(self, smiles_list, il_ids, targets, masks, n_views=6, img_size=224):
        self.smiles = smiles_list
        self.il_ids = il_ids
        self.targets = targets
        self.masks = masks
        self.n_views = n_views

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.ColorJitter(0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        cosmo_dirs = [V5_ROOT / "data/cosmo_images", PROJECT_ROOT / "data/pipeline/cosmo_images"]

        # Build per-IL frame directory lookup (same IL shares frames across temperatures)
        self.il_to_frames = {}
        seen_ils = set()
        for smi in smiles_list:
            il = Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else smi
            if il in seen_ils:
                continue
            seen_ils.add(il)
            h = smiles_to_hash(smi)
            for d in cosmo_dirs:
                c = d / f"{h}_frames"
                if c.exists() and len(list(c.glob("frame_*.png"))) >= 2:
                    self.il_to_frames[il] = c
                    break

        # Map each sample to its IL canonical SMILES
        self.sample_il = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            self.sample_il.append(Chem.MolToSmiles(mol) if mol else smi)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        il = self.sample_il[idx]
        frames_dir = self.il_to_frames.get(il)

        if frames_dir is not None:
            frames = sorted(frames_dir.glob("frame_*.png"))
            # Random sample of n_views
            selected = np.random.choice(len(frames), min(self.n_views, len(frames)), replace=False)
            views = [self.transform(Image.open(frames[i]).convert("RGB")) for i in selected]
            while len(views) < self.n_views:
                views.append(views[-1].clone())
        else:
            views = [torch.zeros(3, 224, 224) for _ in range(self.n_views)]

        return {
            "views": torch.stack(views),
            "targets": torch.from_numpy(self.targets[idx]).float(),
            "masks": torch.from_numpy(self.masks[idx]),
        }


class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=192, n_layers=6, n_heads=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(224, 16, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches+1, embed_dim))
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, n_heads, 4, 0.1, 0.1*i/n_layers) for i in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        p = self.patch_embed(x)
        B = p.shape[0]
        tokens = torch.cat([self.cls_token.expand(B,-1,-1), p], 1) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens[:, 0])


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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="0-4")
    parser.add_argument("--alpha", type=float, default=1.0, help="V-JEPA weight")
    parser.add_argument("--beta", type=float, default=0.5, help="Property weight")
    parser.add_argument("--gamma_w", type=float, default=0.1, help="Contrastive weight")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    s, e = map(int, args.seeds.split("-"))
    seeds = list(range(s, e + 1))

    print("MULTI-TASK SUPERVISED ViT")
    print(f"  V-JEPA(α={args.alpha}) + Property(β={args.beta}) + Contrastive(γ={args.gamma_w})")
    print(f"  Epochs: {args.epochs}, Seeds: {seeds}")

    # Load all data
    all_graph, all_surface, all_thermo, all_targets, all_smiles, all_il_ids = [], [], [], [], [], []
    all_masks = []
    for split in ["train", "val", "test"]:
        d = np.load(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz", allow_pickle=True)
        all_graph.append(d["chemprop_fp"])
        all_surface.append(d["surface_fp"])
        all_thermo.append(d["thermo_feat"])
        all_targets.append(d["targets"])
        all_smiles.extend(d["smiles"])
        all_il_ids.extend(d["il_ids"])
        all_masks.append(np.ones((len(d["targets"]), 7), dtype=bool))

    targets = np.concatenate(all_targets).astype(np.float32)
    masks = np.concatenate(all_masks)
    il_ids = np.array(all_il_ids)
    unique_ils = sorted(set(il_ids))

    # Original split sizes
    n_train = 152
    n_val = 32
    n_test = 39

    # Phase 1: Train supervised ViT on training data
    print("\n=== Phase 1: Multi-Task ViT Training ===")

    train_ds = ILViewPropertyDataset(
        all_smiles[:n_train], all_il_ids[:n_train],
        targets[:n_train], masks[:n_train], n_views=6)

    train_ldr = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    # Initialize from V-JEPA if available
    encoder = ViTEncoder(embed_dim=192).to(device)
    for ckpt in [V5_ROOT / "checkpoints/vjepa/vit_pretrained_vjepa.pt",
                  V5_ROOT / "checkpoints/vjepa_70il/vit_pretrained_vjepa.pt"]:
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device, weights_only=True)
            enc_state = state.get("encoder_state_dict", {})
            if enc_state:
                encoder.load_state_dict(enc_state, strict=False)
                print(f"  Initialized from {ckpt.name}")
                break

    model = SupervisedViT(
        encoder, embed_dim=192, n_properties=7, n_views=6,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma_w,
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {params:,}")

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in train_ldr:
            views = batch["views"].to(device)
            tgt = batch["targets"].to(device)
            msk = batch["masks"].to(device)

            loss, aux = model(views, tgt, msk)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target()

            total_loss += aux["losses"]["total"]
            n_batches += 1

        sched.step()
        if epoch <= 5 or epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"total={total_loss/n_batches:.4f} | "
                  f"vjepa={aux['losses']['vjepa']:.4f} | "
                  f"prop={aux['losses'].get('property',0):.4f} | "
                  f"contr={aux['losses'].get('contrastive',0):.4f}")

    # Phase 2: Extract features for all samples
    print("\n=== Phase 2: Feature Extraction ===")
    model.eval()

    all_feats = []
    extract_ds = ILViewPropertyDataset(all_smiles, all_il_ids, targets, masks, n_views=6)
    extract_ldr = DataLoader(extract_ds, batch_size=8, shuffle=False, num_workers=4)

    with torch.no_grad():
        for batch in extract_ldr:
            views = batch["views"].to(device)
            feat = model.extract_features(views)
            all_feats.append(feat.cpu().numpy())

    supervised_feats = np.concatenate(all_feats)
    print(f"  Extracted: {supervised_feats.shape}")

    # Save
    np.savez(V5_ROOT / "data/supervised_vit_features.npz", features=supervised_feats)

    # Phase 3: Compare supervised ViT vs V-JEPA via LOO on 28 ILs
    print("\n=== Phase 3: LOO Comparison ===")

    # Per-IL features
    il_feats_sup = np.array([supervised_feats[il_ids==il][0] for il in unique_ils])

    # Load V-JEPA features
    vjepa_feats = np.load(V5_ROOT / "data/cached_image_features_train.npz")["vit_feat"]
    # Extend to all samples
    vjepa_all = np.zeros_like(supervised_feats)
    vjepa_all[:n_train] = vjepa_feats
    # Val and test
    for split_name, start in [("val", n_train), ("test", n_train+n_val)]:
        vp = V5_ROOT / f"data/cached_image_features_{split_name}.npz"
        if vp.exists():
            vjepa_all[start:start+{"val":n_val,"test":n_test}[split_name]] = np.load(vp)["vit_feat"]
    il_feats_vj = np.array([vjepa_all[il_ids==il][0] for il in unique_ils])

    targets_il = np.array([targets[il_ids==il].mean(0) for il in unique_ils])

    print(f"\n  {'Property':>10s} {'V-JEPA LOO':>12s} {'Supervised LOO':>14s} {'Delta':>8s}")
    print(f"  {'-'*50}")
    for i, p in enumerate(PROPS):
        for name, feats in [("V-JEPA", il_feats_vj), ("Supervised", il_feats_sup)]:
            pca = PCA(n_components=min(10, len(unique_ils)-2))
            X = pca.fit_transform(feats)
            preds = np.zeros(len(unique_ils))
            for tr, te in LeaveOneOut().split(X):
                preds[te] = Ridge(alpha=10).fit(X[tr], targets_il[tr,i]).predict(X[te])
            if name == "V-JEPA":
                r2_vj = r2_score(targets_il[:,i], preds)
            else:
                r2_sup = r2_score(targets_il[:,i], preds)
        delta = r2_sup - r2_vj
        print(f"  {p:>10s} {r2_vj:>12.4f} {r2_sup:>14.4f} {delta:>+8.4f}")

    # Phase 4: Apply as residual on v4 router
    print("\n=== Phase 4: Image Residual on v4 Router ===")

    # Load v4 router predictions
    pred_dir = PROJECT_ROOT / "cosmobridge_v4/results/seed_predictions"
    seed_files = sorted(pred_dir.glob("seed_*.npz"))
    v4_preds = np.mean([np.load(f)["preds" if "preds" in np.load(f) else "predictions"] for f in seed_files], 0)

    test_targets = targets[n_train+n_val:]
    test_supervised = supervised_feats[n_train+n_val:]
    train_supervised = supervised_feats[:n_train]

    pca = PCA(n_components=20)
    pca.fit(train_supervised)
    train_pca = pca.transform(train_supervised).astype(np.float32)
    test_pca = pca.transform(test_supervised).astype(np.float32)

    # Approximate v4 train predictions
    train_c = np.load(PROJECT_ROOT / "cosmobridge_v4/data/cached_train.npz", allow_pickle=True)
    v4_train_approx = (0.4 * train_c["preds_fusion"] + 0.6 * train_c["preds_chemprop"]).astype(np.float32)
    train_thermo = train_c["thermo_feat"].astype(np.float32)
    test_thermo = np.load(PROJECT_ROOT / "cosmobridge_v4/data/cached_test.npz", allow_pickle=True)["thermo_feat"].astype(np.float32)

    all_res_metrics = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)

        class ResHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate=nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,20),nn.Sigmoid())
                self.head=nn.Sequential(nn.Linear(25,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(0.3),nn.Linear(32,7))
                self.alpha_param=nn.Parameter(torch.full((7,),-3.0))
                with torch.no_grad(): self.head[-1].weight.mul_(0.01); self.head[-1].bias.zero_()
            def forward(self,v,i,t):
                m=i*self.gate(t[:,:5]); r=self.head(torch.cat([m,t[:,:5]],-1))
                return v+torch.sigmoid(self.alpha_param)*r

        res = ResHead().to(device)
        ro = AdamW(res.parameters(), lr=5e-4, weight_decay=1e-2)
        rs = CosineAnnealingLR(ro, T_max=300)

        from torch.utils.data import TensorDataset
        res_ldr = DataLoader(TensorDataset(
            torch.from_numpy(v4_train_approx), torch.from_numpy(train_pca),
            torch.from_numpy(train_thermo), torch.from_numpy(targets[:n_train])),
            batch_size=32, shuffle=True)

        best, bs, p = float("inf"), None, 0
        for ep in range(300):
            res.train()
            for v,i,t,y in res_ldr:
                v,i,t,y=[x.to(device) for x in [v,i,t,y]]
                loss=((res(v,i,t)-y)**2).mean(); ro.zero_grad(); loss.backward(); ro.step()
            rs.step()
            res.eval()
            with torch.no_grad():
                tl=((res(torch.from_numpy(v4_train_approx).to(device),
                         torch.from_numpy(train_pca).to(device),
                         torch.from_numpy(train_thermo).to(device))-
                     torch.from_numpy(targets[:n_train]).to(device))**2).mean().item()
            if tl<best: best=tl; bs={k:v.clone() for k,v in res.state_dict().items()}; p=0
            else:
                p+=1
                if p>=50: break

        res.load_state_dict(bs); res.eval()
        with torch.no_grad():
            final = res(torch.from_numpy(v4_preds.astype(np.float32)).to(device),
                        torch.from_numpy(test_pca).to(device),
                        torch.from_numpy(test_thermo).to(device)).cpu().numpy()

        m = compute_metrics(final, test_targets)
        all_res_metrics.append(m)
        if seed == seeds[0]:
            print(f"  Seed {seed}: avg R²={m['avg_r2']:.4f}")

    avgs = [m["avg_r2"] for m in all_res_metrics]
    print(f"\n  Supervised ViT + v4 residual: {np.mean(avgs):.4f} ± {np.std(avgs):.4f}")
    print(f"  V-JEPA + v4 residual (Phase A): 0.816")
    print(f"  v4 paper: 0.818")

    for p in PROPS:
        vals = [m[f"{p}_r2"] for m in all_res_metrics]
        print(f"    {p:8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    out = V5_ROOT / "results/supervised_vit"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({"metrics": all_res_metrics, "avg": float(np.mean(avgs))}, f, indent=2, default=float)
    print(f"\nSaved: {out}/summary.json")


if __name__ == "__main__":
    main()
