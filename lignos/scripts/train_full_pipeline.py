#!/usr/bin/env python3
"""Full Pipeline: v4 router + residual image + physics correction + DAPT.

Phase A (on actual v4 predictions):
    1. Reproduce v4 router (0.818)
    2. Add image residual (+0.001)
    3. Add physics correction (+0.005 on G_mix)
    → Expected: ~0.824

Phase B (expanded 70 ILs):
    4. DAPT: train FFN on 70 ILs → fine-tune on 28 ILs
    5. Use as improved Path B in v4 router
    6. Add image residual on top
    → Expected: 0.83+

Usage:
    python train_full_pipeline.py --phase A --seeds 0-9
    python train_full_pipeline.py --phase B --seeds 0-9
    python train_full_pipeline.py --phase all --seeds 0-9
"""

import argparse, json, sys, pickle, time
from pathlib import Path

import numpy as np
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


def load_v4_router_predictions():
    """Load actual v4 router predictions from saved seed files."""
    pred_dir = PROJECT_ROOT / "cosmobridge_v4/results/seed_predictions"
    if not pred_dir.exists():
        return None

    seed_files = sorted(pred_dir.glob("seed_*.npz"))
    if not seed_files:
        return None

    # Average across seeds for ensemble prediction
    all_preds = []
    targets = None
    for f in seed_files:
        d = np.load(f)
        key = "preds" if "preds" in d else "predictions"
        all_preds.append(d[key])
        if targets is None:
            targets = d["targets"]

    ensemble = np.mean(all_preds, axis=0)
    return ensemble, targets, len(seed_files)


def physics_correction(preds, thermo, tscaler, fscaler):
    """Apply G_E and G_mix thermodynamic corrections."""
    corrected = preds.copy()
    R = 8.314e-3

    # Denormalize
    preds_raw = preds * tscaler.scale_ + tscaler.mean_
    T_raw = thermo[:, 0] * fscaler.scale_[0] + fscaler.mean_[0]

    gamma1_raw = preds_raw[:, 0]
    gamma2_raw = preds_raw[:, 1]
    x1 = 0.5

    # Physics G_E from gamma1 + gamma2
    ln_term = x1 * np.log(np.clip(gamma1_raw, 1e-6, None)) + \
              (1-x1) * np.log(np.clip(gamma2_raw, 1e-6, None))
    ge_physics_raw = R * T_raw * ln_term
    ge_physics_norm = (ge_physics_raw - tscaler.mean_[2]) / tscaler.scale_[2]

    # Physics G_mix from G_E
    ideal = R * T_raw * (x1*np.log(x1) + (1-x1)*np.log(1-x1))
    gm_physics_raw = ge_physics_raw + ideal
    gm_physics_norm = (gm_physics_raw - tscaler.mean_[4]) / tscaler.scale_[4]

    return ge_physics_norm, gm_physics_norm


def image_residual_correction(v4_preds, image_feats_train, thermo_train, targets_train,
                               image_feats_test, thermo_test, v4_test_preds, device, seed):
    """Train PCA(ViT) + T-conditioning residual head."""
    set_seed(seed)

    # PCA
    pca = PCA(n_components=20)
    img_train = pca.fit_transform(image_feats_train).astype(np.float32)
    img_test = pca.transform(image_feats_test).astype(np.float32)

    class ResHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Sequential(nn.Linear(5,32), nn.GELU(), nn.Linear(32,20), nn.Sigmoid())
            self.head = nn.Sequential(nn.Linear(25,32), nn.LayerNorm(32), nn.GELU(),
                                       nn.Dropout(0.3), nn.Linear(32,7))
            self.alpha = nn.Parameter(torch.full((7,), -3.0))
            with torch.no_grad():
                self.head[-1].weight.mul_(0.01); self.head[-1].bias.zero_()
        def forward(self, v4p, img, th):
            mod = img * self.gate(th[:,:5])
            res = self.head(torch.cat([mod, th[:,:5]], -1))
            return v4p + torch.sigmoid(self.alpha) * res

    model = ResHead().to(device)
    opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = CosineAnnealingLR(opt, T_max=300)

    ds = TensorDataset(torch.from_numpy(v4_preds), torch.from_numpy(img_train),
                        torch.from_numpy(thermo_train), torch.from_numpy(targets_train))
    ldr = DataLoader(ds, batch_size=32, shuffle=True)

    best_loss, best_state, no_imp = float("inf"), None, 0
    for ep in range(300):
        model.train()
        for v4p, img, th, y in ldr:
            v4p,img,th,y = [x.to(device) for x in [v4p,img,th,y]]
            loss = ((model(v4p,img,th)-y)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            all_v4 = torch.from_numpy(v4_preds).to(device)
            all_img = torch.from_numpy(img_train).to(device)
            all_th = torch.from_numpy(thermo_train).to(device)
            tl = ((model(all_v4, all_img, all_th) - torch.from_numpy(targets_train).to(device))**2).mean().item()
        if tl < best_loss:
            best_loss = tl; best_state = {k:v.clone() for k,v in model.state_dict().items()}; no_imp=0
        else:
            no_imp += 1
            if no_imp >= 50: break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        corrected = model(
            torch.from_numpy(v4_test_preds).to(device),
            torch.from_numpy(img_test).to(device),
            torch.from_numpy(thermo_test).to(device),
        ).cpu().numpy()

    alpha = torch.sigmoid(model.alpha).detach().cpu().numpy()
    return corrected, alpha


def phase_a(seeds, device):
    """Phase A: v4 router + residual image + physics correction."""
    print("\n" + "="*60)
    print("PHASE A: Actual v4 + Residual Image + Physics Correction")
    print("="*60)

    with open(PROJECT_ROOT / "data/processed/target_scaler.pkl", "rb") as f:
        tscaler = pickle.load(f)
    with open(PROJECT_ROOT / "data/processed/feature_scaler.pkl", "rb") as f:
        fscaler = pickle.load(f)

    # Load v4 router predictions
    router_result = load_v4_router_predictions()
    if router_result is None:
        print("  v4 router predictions not found. Running v4 router training first...")
        print("  Please run: python cosmobridge_v4/scripts/train_v4_router.py")
        return None

    v4_ensemble_preds, test_targets, n_seeds = router_result
    m_v4 = compute_metrics(v4_ensemble_preds, test_targets)
    print(f"\n  Step 1: v4 router ensemble ({n_seeds} seeds): avg R²={m_v4['avg_r2']:.4f}")
    for p in PROPS:
        print(f"    {p:8s}: {m_v4[f'{p}_r2']:.4f}")

    # Load cached data for physics + image
    test_cached = np.load(PROJECT_ROOT / "cosmobridge_v4/data/cached_test.npz", allow_pickle=True)
    train_cached = np.load(PROJECT_ROOT / "cosmobridge_v4/data/cached_train.npz", allow_pickle=True)

    # Step 2: Physics correction on v4 ensemble
    ge_phys, gm_phys = physics_correction(
        v4_ensemble_preds, test_cached["thermo_feat"], tscaler, fscaler)

    # Calibrate blend on TRAINING data
    train_v4_preds = np.zeros_like(train_cached["targets"])
    for i in range(7):
        # Approximate v4 train predictions using gate values
        train_v4_preds[:, i] = 0.36 * train_cached["preds_fusion"][:, i] + \
                                0.64 * train_cached["preds_chemprop"][:, i]

    ge_phys_train, gm_phys_train = physics_correction(
        train_v4_preds, train_cached["thermo_feat"], tscaler, fscaler)

    # Find best blend for G_E
    def neg_r2_ge(alpha):
        blended = alpha * ge_phys_train + (1-alpha) * train_v4_preds[:, 2]
        ss_r = ((train_cached["targets"][:,2] - blended)**2).sum()
        ss_t = ((train_cached["targets"][:,2] - train_cached["targets"][:,2].mean())**2).sum()
        return -(1 - ss_r/ss_t)
    alpha_ge = minimize_scalar(neg_r2_ge, bounds=(0, 0.5), method='bounded').x

    def neg_r2_gm(alpha):
        blended = alpha * gm_phys_train + (1-alpha) * train_v4_preds[:, 4]
        ss_r = ((train_cached["targets"][:,4] - blended)**2).sum()
        ss_t = ((train_cached["targets"][:,4] - train_cached["targets"][:,4].mean())**2).sum()
        return -(1 - ss_r/ss_t)
    alpha_gm = minimize_scalar(neg_r2_gm, bounds=(0, 0.5), method='bounded').x

    physics_corrected = v4_ensemble_preds.copy()
    physics_corrected[:, 2] = alpha_ge * ge_phys + (1-alpha_ge) * v4_ensemble_preds[:, 2]
    physics_corrected[:, 4] = alpha_gm * gm_phys + (1-alpha_gm) * v4_ensemble_preds[:, 4]

    m_phys = compute_metrics(physics_corrected, test_targets)
    print(f"\n  Step 2: + Physics correction (G_E α={alpha_ge:.3f}, G_mix α={alpha_gm:.3f})")
    print(f"    avg R²={m_phys['avg_r2']:.4f} (Δ={m_phys['avg_r2']-m_v4['avg_r2']:+.4f})")

    # Step 3: Image residual on physics-corrected predictions
    img_train_path = V5_ROOT / "data/cached_image_features_train.npz"
    img_test_path = V5_ROOT / "data/cached_image_features_test.npz"

    if img_train_path.exists() and img_test_path.exists():
        img_train = np.load(img_train_path)["vit_feat"]
        img_test = np.load(img_test_path)["vit_feat"]

        all_img_metrics = []
        for seed in seeds:
            corrected, alpha = image_residual_correction(
                train_v4_preds.astype(np.float32),
                img_train, train_cached["thermo_feat"].astype(np.float32),
                train_cached["targets"].astype(np.float32),
                img_test, test_cached["thermo_feat"].astype(np.float32),
                physics_corrected.astype(np.float32),
                device, seed)
            m = compute_metrics(corrected, test_targets)
            all_img_metrics.append(m)

        avgs = [m["avg_r2"] for m in all_img_metrics]
        print(f"\n  Step 3: + Image residual ({len(seeds)} seeds)")
        print(f"    avg R²={np.mean(avgs):.4f} ± {np.std(avgs):.4f} "
              f"(Δ={np.mean(avgs)-m_v4['avg_r2']:+.4f} vs v4)")
        for p in PROPS:
            vals = [m[f"{p}_r2"] for m in all_img_metrics]
            v4_val = m_v4[f"{p}_r2"]
            print(f"      {p:8s}: {np.mean(vals):.4f}±{np.std(vals):.4f} (Δ={np.mean(vals)-v4_val:+.4f})")
    else:
        print("\n  Step 3: Image features not available, skipping")
        all_img_metrics = None

    return {
        "v4_router": m_v4,
        "physics": m_phys,
        "image_residual": all_img_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["A", "B", "all"], default="A")
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    s, e = map(int, args.seeds.split("-"))
    seeds = list(range(s, e + 1))

    print("FULL PIPELINE: v4 Router + Physics + Image + DAPT")
    print(f"  Phase: {args.phase}, Seeds: {seeds}, Device: {device}")

    results = {}

    if args.phase in ("A", "all"):
        results["phase_a"] = phase_a(seeds, device)

    # Save
    out = V5_ROOT / "results/full_pipeline"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}/summary.json")


if __name__ == "__main__":
    main()
