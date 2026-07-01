#!/usr/bin/env python3
"""Leave-One-IL-Out Cross-Validation for COSMOBridge.

Instead of the single 19/4/5 IL split, evaluates on ALL 28 ILs.
Each IL takes a turn as the test set (8 samples per IL).

Runs 3 models for clean comparison:
    1. v4-style FFN (graph + surface + thermo) → baseline
    2. v4-style FFN + image residual → does image help?
    3. v4-style FFN trained with DAPT + image residual → does more data help?

This gives 28 evaluation points instead of 5, providing
statistically robust evidence for whether images help.

Usage:
    python train_loocv.py --device cuda
"""

import argparse, json, sys, pickle, hashlib
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
        if len(t) < 2: m[f"{n}_r2"] = float("nan"); continue
        ss_r = ((t[:,i]-p[:,i])**2).sum()
        ss_t = ((t[:,i]-t[:,i].mean())**2).sum()
        m[f"{n}_r2"] = (1-ss_r/(ss_t+1e-8)).item()
    m["avg_r2"] = np.nanmean([v for k,v in m.items() if k != "avg_r2"])
    return m


def load_full_dataset():
    """Load ALL 28 ILs (223 samples) with features."""
    raw = pd.read_csv(PROJECT_ROOT / "data/processed/il_data_raw.csv")

    # Load all cached splits and combine
    all_graph, all_surface, all_thermo, all_targets = [], [], [], []
    all_smiles, all_il_ids = [], []
    all_preds_fusion, all_preds_chemprop = [], []

    for split in ["train", "val", "test"]:
        d = np.load(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz", allow_pickle=True)
        all_graph.append(d["chemprop_fp"])
        all_surface.append(d["surface_fp"])
        all_thermo.append(d["thermo_feat"])
        all_targets.append(d["targets"])
        all_smiles.extend(d["smiles"])
        all_il_ids.extend(d["il_ids"])
        all_preds_fusion.append(d["preds_fusion"])
        all_preds_chemprop.append(d["preds_chemprop"])

    return {
        "graph": np.concatenate(all_graph).astype(np.float32),
        "surface": np.concatenate(all_surface).astype(np.float32),
        "thermo": np.concatenate(all_thermo).astype(np.float32),
        "targets": np.concatenate(all_targets).astype(np.float32),
        "preds_fusion": np.concatenate(all_preds_fusion).astype(np.float32),
        "preds_chemprop": np.concatenate(all_preds_chemprop).astype(np.float32),
        "smiles": all_smiles,
        "il_ids": all_il_ids,
    }


def load_image_features(smiles_list, vjepa_path, device):
    """Extract V-JEPA ViT features for all samples."""
    from models.multiview_vit import MultiViewViT
    from torchvision import transforms
    from PIL import Image

    vit = MultiViewViT(n_views=36, embed_dim=192, dropout=0.0).to(device)
    for ckpt in [vjepa_path,
                  V5_ROOT / "checkpoints/vjepa_70il/vit_pretrained_vjepa.pt",
                  V5_ROOT / "checkpoints/vjepa/vit_pretrained_vjepa.pt"]:
        if ckpt and Path(ckpt).exists():
            state = torch.load(ckpt, map_location=device, weights_only=True)
            enc = state.get("encoder_state_dict", {})
            if enc:
                vit.load_state_dict(enc, strict=False)
                print(f"  Loaded ViT from {Path(ckpt).name}")
                break
    vit.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    cosmo_dirs = [V5_ROOT/"data/cosmo_images", PROJECT_ROOT/"data/pipeline/cosmo_images"]

    all_feats = []
    for smi in smiles_list:
        h = smiles_to_hash(smi)
        frames_dir = None
        for d in cosmo_dirs:
            c = d / f"{h}_frames"
            if c.exists() and len(list(c.glob("frame_*.png"))) >= 2:
                frames_dir = c; break

        if frames_dir is None:
            # Try IL short name based dirs
            all_feats.append(np.zeros(192, dtype=np.float32))
            continue

        frames = sorted(frames_dir.glob("frame_*.png"))
        indices = np.linspace(0, len(frames)-1, 6, dtype=int)
        views = torch.stack([transform(Image.open(frames[i]).convert("RGB")) for i in indices])
        views = views.unsqueeze(0).to(device)

        with torch.no_grad():
            emb, _ = vit.encode_views_chunked(views, chunk_size=3)
        all_feats.append(emb.cpu().numpy()[0])

    feats = np.array(all_feats, dtype=np.float32)
    n_valid = (np.abs(feats).sum(1) > 0.01).sum()
    print(f"  Image features: {feats.shape}, {n_valid}/{len(feats)} with real images")
    return feats


def train_ffn(train_feats, train_targets, val_feats, val_targets, device, epochs=200, patience=30):
    """Train a simple FFN on (graph, surface, thermo) → 7 properties."""
    model = nn.Sequential(
        nn.Linear(581, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 7),
    ).to(device)

    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    ldr = DataLoader(TensorDataset(torch.from_numpy(train_feats), torch.from_numpy(train_targets)),
                      batch_size=32, shuffle=True)

    best, bs, p = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for f, t in ldr:
            f, t = f.to(device), t.to(device)
            loss = ((model(f)-t)**2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = ((model(torch.from_numpy(val_feats).to(device)) -
                   torch.from_numpy(val_targets).to(device))**2).mean().item()
        if vl < best: best=vl; bs={k:v.clone() for k,v in model.state_dict().items()}; p=0
        else:
            p += 1
            if p >= patience: break

    model.load_state_dict(bs); model.eval()
    return model


def train_image_residual(base_preds, img_pca, thermo, targets, device, epochs=300, patience=50):
    """Train image residual head."""
    class ResHead(nn.Module):
        def __init__(self, pca_dim):
            super().__init__()
            self.gate = nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,pca_dim),nn.Sigmoid())
            self.head = nn.Sequential(nn.Linear(pca_dim+5,32),nn.LayerNorm(32),nn.GELU(),
                                       nn.Dropout(0.3),nn.Linear(32,7))
            self.alpha = nn.Parameter(torch.full((7,),-3.0))
            with torch.no_grad(): self.head[-1].weight.mul_(0.01); self.head[-1].bias.zero_()
        def forward(self, v, i, t):
            m = i * self.gate(t[:,:5])
            return v + torch.sigmoid(self.alpha) * self.head(torch.cat([m,t[:,:5]],-1))

    model = ResHead(img_pca.shape[1]).to(device)
    opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    ldr = DataLoader(TensorDataset(
        torch.from_numpy(base_preds), torch.from_numpy(img_pca),
        torch.from_numpy(thermo), torch.from_numpy(targets)),
        batch_size=32, shuffle=True)

    best, bs, p = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for v,i,t,y in ldr:
            v,i,t,y = [x.to(device) for x in [v,i,t,y]]
            loss = ((model(v,i,t)-y)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            tl = ((model(torch.from_numpy(base_preds).to(device),
                          torch.from_numpy(img_pca).to(device),
                          torch.from_numpy(thermo).to(device)) -
                   torch.from_numpy(targets).to(device))**2).mean().item()
        if tl < best: best=tl; bs={k:v.clone() for k,v in model.state_dict().items()}; p=0
        else:
            p += 1
            if p >= patience: break

    model.load_state_dict(bs); model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vjepa_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    set_seed(42)

    print("LEAVE-ONE-IL-OUT CROSS-VALIDATION")
    print("=" * 60)

    # Load all data
    print("\nLoading full dataset (28 ILs, 223 samples)...")
    data = load_full_dataset()
    il_ids = np.array(data["il_ids"])
    unique_ils = sorted(set(il_ids))
    n_ils = len(unique_ils)
    print(f"  {len(data['smiles'])} samples, {n_ils} unique ILs")

    # Load image features
    print("\nExtracting image features...")
    img_feats = load_image_features(data["smiles"], args.vjepa_checkpoint, device)

    # Combine features
    feats = np.concatenate([data["graph"], data["surface"], data["thermo"]], axis=1).astype(np.float32)

    # Run LOO-CV
    print(f"\nRunning {n_ils}-fold leave-one-IL-out CV...")

    results_ffn = []       # Model 1: FFN only
    results_ffn_img = []   # Model 2: FFN + image residual
    all_fold_metrics = []

    for fold_idx, test_il in enumerate(unique_ils):
        test_mask = il_ids == test_il
        train_mask = ~test_mask
        n_test = test_mask.sum()
        n_train = train_mask.sum()

        # Split into train (use 80% for train, 20% for val from remaining ILs)
        train_ils = [il for il in unique_ils if il != test_il]
        val_il = train_ils[fold_idx % len(train_ils)]  # rotate val IL
        val_mask = il_ids == val_il
        pure_train_mask = train_mask & ~val_mask

        # Features
        train_feats = feats[pure_train_mask]
        train_targets = data["targets"][pure_train_mask]
        val_feats = feats[val_mask]
        val_targets = data["targets"][val_mask]
        test_feats = feats[test_mask]
        test_targets = data["targets"][test_mask]

        # Image PCA (fit on training fold)
        train_img = img_feats[pure_train_mask]
        val_img = img_feats[val_mask]
        test_img = img_feats[test_mask]
        pca = PCA(n_components=min(20, train_img.shape[0]-1, train_img.shape[1]))
        pca.fit(train_img)
        train_img_pca = pca.transform(train_img).astype(np.float32)
        test_img_pca = pca.transform(test_img).astype(np.float32)

        # ── Model 1: FFN only ──
        set_seed(42 + fold_idx)
        ffn = train_ffn(train_feats, train_targets, val_feats, val_targets, device)
        with torch.no_grad():
            ffn_preds = ffn(torch.from_numpy(test_feats).to(device)).cpu().numpy()
            ffn_train_preds = ffn(torch.from_numpy(train_feats).to(device)).cpu().numpy()

        m_ffn = compute_metrics(ffn_preds, test_targets)
        results_ffn.append(m_ffn)

        # ── Model 2: FFN + image residual ──
        set_seed(42 + fold_idx)
        res_model = train_image_residual(
            ffn_train_preds.astype(np.float32), train_img_pca,
            data["thermo"][pure_train_mask].astype(np.float32),
            train_targets.astype(np.float32), device)

        with torch.no_grad():
            img_preds = res_model(
                torch.from_numpy(ffn_preds.astype(np.float32)).to(device),
                torch.from_numpy(test_img_pca).to(device),
                torch.from_numpy(data["thermo"][test_mask].astype(np.float32)).to(device),
            ).cpu().numpy()

        m_img = compute_metrics(img_preds, test_targets)
        results_ffn_img.append(m_img)

        fold_result = {
            "test_il": test_il, "n_test": int(n_test), "n_train": int(n_train),
            "ffn": m_ffn, "ffn_img": m_img,
        }
        all_fold_metrics.append(fold_result)

        delta = m_img["avg_r2"] - m_ffn["avg_r2"]
        marker = "+" if delta > 0 else ""
        print(f"  Fold {fold_idx+1:2d}/{n_ils} | IL={test_il:20s} | "
              f"FFN={m_ffn['avg_r2']:.4f} | +Image={m_img['avg_r2']:.4f} | "
              f"Δ={marker}{delta:.4f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("LEAVE-ONE-IL-OUT CV RESULTS")
    print(f"{'='*60}")

    ffn_avgs = [m["avg_r2"] for m in results_ffn]
    img_avgs = [m["avg_r2"] for m in results_ffn_img]

    print(f"\n  {'Method':30s} {'avg R²':>10s} {'std':>8s} {'median':>8s}")
    print(f"  {'-'*60}")
    print(f"  {'FFN only':30s} {np.mean(ffn_avgs):>10.4f} {np.std(ffn_avgs):>8.4f} {np.median(ffn_avgs):>8.4f}")
    print(f"  {'FFN + Image Residual':30s} {np.mean(img_avgs):>10.4f} {np.std(img_avgs):>8.4f} {np.median(img_avgs):>8.4f}")
    print(f"  {'Delta (Image - FFN)':30s} {np.mean(img_avgs)-np.mean(ffn_avgs):>+10.4f}")

    # Count how many folds image helped
    n_better = sum(1 for f, i in zip(ffn_avgs, img_avgs) if i > f)
    print(f"\n  Image helped in {n_better}/{n_ils} folds ({100*n_better/n_ils:.0f}%)")

    # Paired Wilcoxon test
    try:
        from scipy.stats import wilcoxon
        stat, pval = wilcoxon(img_avgs, ffn_avgs, alternative="greater")
        print(f"  Wilcoxon (image > FFN): p={pval:.4f} {'*' if pval<0.05 else ''}")
    except Exception as e:
        print(f"  Wilcoxon test failed: {e}")

    # Per-property comparison
    print(f"\n  Per-property (mean across {n_ils} folds):")
    print(f"  {'Property':>10s} {'FFN':>8s} {'FFN+Img':>8s} {'Delta':>8s}")
    for p in PROPS:
        f_vals = [m[f"{p}_r2"] for m in results_ffn]
        i_vals = [m[f"{p}_r2"] for m in results_ffn_img]
        delta = np.nanmean(i_vals) - np.nanmean(f_vals)
        print(f"  {p:>10s} {np.nanmean(f_vals):>8.4f} {np.nanmean(i_vals):>8.4f} {delta:>+8.4f}")

    # Save
    out = V5_ROOT / "results/loocv"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({
            "n_folds": n_ils,
            "ffn_avg": float(np.mean(ffn_avgs)),
            "img_avg": float(np.mean(img_avgs)),
            "delta": float(np.mean(img_avgs) - np.mean(ffn_avgs)),
            "n_image_better": n_better,
            "per_fold": all_fold_metrics,
        }, f, indent=2, default=float)
    print(f"\nSaved: {out}/summary.json")


if __name__ == "__main__":
    main()
