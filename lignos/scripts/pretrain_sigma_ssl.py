#!/usr/bin/env python3
"""Stage 3 — Self-supervised pretraining of the V-JEPA ViT with 1D sigma-profile
reconstruction as the pretext task.

Rationale
---------
Adding raw sigma features directly to the PerPropHead (Combined+Sigma 90D in
slurm_combined_sigma.sh) HURT performance (0.822 vs 0.830 for Combined 40D
without sigma) — classic curse of dimensionality on 152 training samples.
Sigma profiles contain the physics signal but the downstream head can't use
them as extra columns. The cleaner way to inject sigma information is to force
the ViT features themselves to encode it, by pretraining the ViT to regress the
1D sigma profile from the 36 rotation views.

After pretraining, re-extract `vit_feat` with the new weights and re-run the
Combined(40D) recipe. If sigma genuinely adds information, the new `vit_feat`
should yield strictly higher R² than V-JEPA's.

Data
----
    sigma_profiles.npz contains (152, 50) / (32, 50) / (39, 50) 1D profiles
    aligned with cached_{train,val,test}.npz (same sample order).

Pretraining uses train+val samples (184 total). Test is held out.

Loss
----
    L = MSE(pred, sigma_target) + 0.1 * masked_view_reconstruction

    Masked-view term: randomly mask k out of 36 views during training; force the
    model to still predict the same sigma profile. Acts as a view-level dropout
    that encourages invariance to missing views.

Checkpoint output: lignos/checkpoints/sigma_ssl/vit_pretrained_sigma.pt
    (same encoder_state_dict format as vit_pretrained_vjepa.pt so that
    train_v4_plus_image.py:64-73 and extract_vit_features_per_conformer.py
    can load it as a drop-in replacement.)

Usage:
    python pretrain_sigma_ssl.py --epochs 100 --batch_size 8 --lr 1e-4
    python pretrain_sigma_ssl.py --init_from checkpoints/vjepa/vit_pretrained_vjepa.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5))

from data.dataset import COSMOBridgeV5Dataset  # noqa: E402
from models.multiview_vit import MultiViewViT  # noqa: E402


class SigmaReconstructionHead(nn.Module):
    """Small MLP that maps (B, 192) ViT embedding -> (B, 50) 1D sigma profile."""

    def __init__(self, embed_dim=192, sigma_bins=50, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, sigma_bins),
        )

    def forward(self, x):
        return self.net(x)


class SigmaSSLDataset(torch.utils.data.Dataset):
    """Wraps COSMOBridgeV5Dataset for a split and exposes (views, sigma_profile).

    Only needs the rotation views + the index-aligned sigma profile for
    each sample, so we skip the ion images and cached features.
    """

    def __init__(self, base_dataset, sigma_profiles):
        self.base = base_dataset
        self.sigma = sigma_profiles
        assert len(self.base) == len(self.sigma), (
            f"Dataset length {len(self.base)} != sigma rows {len(self.sigma)}"
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        return {
            "views": item["views"],                             # (36, 3, H, W)
            "sigma": torch.from_numpy(self.sigma[idx]).float(), # (50,)
        }


def build_dataset(split, n_views=36):
    return COSMOBridgeV5Dataset(
        cached_features_path=str(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz"),
        cosmo_images_dir=str(V5 / "data/cosmo_images"),
        ion_images_dir=str(V5 / "data/ion_images"),
        orig_cosmo_dir=str(PROJECT_ROOT / "data/pipeline/cosmo_images"),
        master_index_path=str(V5 / "data/master_index.csv"),
        n_views=n_views,
        image_size=224,
        view_sample_mode="all",
    )


def masked_view_batch(views, mask_ratio=0.3):
    """Randomly zero out some fraction of views per sample.

    views: (B, n_views, 3, H, W)
    Returns a new tensor with ~mask_ratio fraction of views zeroed.
    """
    B, V = views.shape[:2]
    mask = (torch.rand(B, V, device=views.device) < mask_ratio).float()
    # Expand to match view tensor dims
    mask = mask[:, :, None, None, None]
    return views * (1 - mask)


def load_vit(init_from, device):
    vit = MultiViewViT(n_views=36, embed_dim=192, dropout=0.1)
    if init_from and Path(init_from).exists():
        state = torch.load(init_from, map_location="cpu", weights_only=True)
        encoder_state = state.get("encoder_state_dict", state)
        missing, _ = vit.load_state_dict(encoder_state, strict=False)
        print(f"Initialized from {Path(init_from).name} ({len(missing)} missing keys)")
    else:
        print(f"Starting from random init (no {init_from})")
    return vit.to(device)


def encode(vit, views):
    """Wrap vit.encode_views_chunked and handle return type."""
    emb, _ = vit.encode_views_chunked(views, chunk_size=3)
    return emb  # (B, embed_dim)


def train_epoch(vit, head, loader, opt, device, mask_ratio, aux_weight):
    vit.train()
    head.train()
    total, n = 0.0, 0
    for batch in loader:
        views = batch["views"].to(device)
        sigma = batch["sigma"].to(device)

        # Main forward: full views
        emb = encode(vit, views)
        pred = head(emb)
        main_loss = F.mse_loss(pred, sigma)

        # Masked aux forward: same target, noisier views
        if aux_weight > 0:
            masked = masked_view_batch(views, mask_ratio=mask_ratio)
            emb_m = encode(vit, masked)
            pred_m = head(emb_m)
            aux_loss = F.mse_loss(pred_m, sigma)
            loss = main_loss + aux_weight * aux_loss
        else:
            loss = main_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(vit.parameters()) + list(head.parameters()), 1.0
        )
        opt.step()

        total += loss.item() * views.size(0)
        n += views.size(0)

    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(vit, head, loader, device):
    vit.eval()
    head.eval()
    preds, targs = [], []
    for batch in loader:
        views = batch["views"].to(device)
        sigma = batch["sigma"].to(device)
        pred = head(encode(vit, views))
        preds.append(pred.cpu())
        targs.append(sigma.cpu())
    preds = torch.cat(preds).numpy()
    targs = torch.cat(targs).numpy()
    mse = float(((preds - targs) ** 2).mean())
    # R² across all bins
    ss_res = ((targs - preds) ** 2).sum()
    ss_tot = ((targs - targs.mean(axis=0, keepdims=True)) ** 2).sum()
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    return mse, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--mask_ratio", type=float, default=0.3)
    ap.add_argument("--aux_weight", type=float, default=0.1)
    ap.add_argument("--init_from", type=str,
                    default=str(V5 / "checkpoints/vjepa/vit_pretrained_vjepa.pt"))
    ap.add_argument("--output_dir", type=str, default=str(V5 / "checkpoints/sigma_ssl"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load sigma profiles ──
    sigma = np.load(V5 / "data/sigma_profiles.npz")
    sig_tr = sigma["train"]   # (152, 50)
    sig_va = sigma["val"]     # (32, 50)
    sig_te = sigma["test"]    # (39, 50)
    print(f"Sigma profiles: train={sig_tr.shape}  val={sig_va.shape}  test={sig_te.shape}")

    # Normalize sigma (per-bin z-score using train statistics)
    mu = sig_tr.mean(axis=0, keepdims=True)
    std = sig_tr.std(axis=0, keepdims=True) + 1e-6
    sig_tr_n = (sig_tr - mu) / std
    sig_va_n = (sig_va - mu) / std
    sig_te_n = (sig_te - mu) / std

    np.savez(out_dir / "sigma_normalization.npz", mean=mu, std=std)

    # ── Datasets ──
    ds_tr = SigmaSSLDataset(build_dataset("train"), sig_tr_n)
    ds_va = SigmaSSLDataset(build_dataset("val"), sig_va_n)
    ds_te = SigmaSSLDataset(build_dataset("test"), sig_te_n)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # ── Model ──
    vit = load_vit(args.init_from, device)
    head = SigmaReconstructionHead(embed_dim=192, sigma_bins=50).to(device)

    opt = AdamW(
        list(vit.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    # ── Training loop ──
    history = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            vit, head, dl_tr, opt, device, args.mask_ratio, args.aux_weight
        )
        val_mse, val_r2 = eval_epoch(vit, head, dl_va, device)
        sched.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_mse,
            "val_r2": val_r2,
            "lr": opt.param_groups[0]["lr"],
        })
        print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}  "
              f"val_mse={val_mse:.4f}  val_r2={val_r2:.4f}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {"encoder_state_dict": vit.state_dict(),
                 "head_state_dict": head.state_dict(),
                 "epoch": epoch, "val_mse": val_mse, "val_r2": val_r2},
                out_dir / "vit_pretrained_sigma.pt",
            )

    # ── Final test linear-probe-ish report ──
    test_mse, test_r2 = eval_epoch(vit, head, dl_te, device)
    print(f"\nFinal test: MSE={test_mse:.4f}  R²={test_r2:.4f}")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump({
            "history": history,
            "best_val_mse": best_val,
            "test_mse": test_mse,
            "test_r2": test_r2,
            "args": vars(args),
        }, f, indent=2)

    print(f"\nSaved pretrained encoder to {out_dir / 'vit_pretrained_sigma.pt'}")
    print(f"Next step: run extract_vit_features_per_conformer.py with "
          f"--use_sigma_pretrained, then audit_residuals.py to see R² gain.")


if __name__ == "__main__":
    main()
