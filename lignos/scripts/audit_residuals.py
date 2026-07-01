#!/usr/bin/env python3
"""Stage 1 — Label cleanup via residual audit of the Combined(40D) baseline.

Reproduces the exact `slurm_combined_sigma.sh` recipe that scores avg R² ≈ 0.830
(PerPropHead residual on v4 router predictions using PCA-20(V-JEPA) + PCA-20(Supervised ViT)
features), then records per-sample residuals across 10 seeds for all three splits
and flags suspect labels.

Outputs (under lignos/results/residual_audit/):
    flagged_samples.csv      — samples where |z-score| > threshold on ≥2 properties
    per_sample_residuals.csv — raw per-sample per-property residuals (mean over 10 seeds)
    per_property_summary.json — R², MAE, top-10 worst residuals per property
    train_vs_val_gap.json    — overfitting diagnostic

Usage:
    python audit_residuals.py
    python audit_residuals.py --n_seeds 10 --z_threshold 2.5 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P",
         "lignin_wt"]
CORE_PROPS = PROPS[:7]  # original 7 for backward-compatible R² comparison


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def r2_per_prop(preds, targets, prop_names=None):
    """Compute per-property R², skipping NaN targets per column."""
    if prop_names is None:
        prop_names = PROPS[:targets.shape[1]]
    out = {}
    for i, n in enumerate(prop_names):
        if i >= targets.shape[1]:
            break
        mask = ~np.isnan(targets[:, i])
        if mask.sum() < 2:
            out[n] = float("nan")
            continue
        t = targets[mask, i]
        p = preds[mask, i]
        sr = ((t - p) ** 2).sum()
        st = ((t - t.mean()) ** 2).sum()
        out[n] = float(1 - sr / (st + 1e-8))
    valid = [out[p] for p in prop_names if p in out and not np.isnan(out[p])]
    out["avg"] = float(np.mean(valid)) if valid else float("nan")
    # Also compute avg over core 7 props only for comparison
    core_valid = [out[p] for p in CORE_PROPS if p in out and not np.isnan(out[p])]
    out["avg_core7"] = float(np.mean(core_valid)) if core_valid else float("nan")
    return out


class PerPropHead(nn.Module):
    """Per-property prediction head with gated features."""

    def __init__(self, nf, n_props=None, depth="shallow", wide_thermo=False,
                 deep_head_indices=None):
        super().__init__()
        n_props = n_props or len(PROPS)
        self.wide_thermo = wide_thermo
        self.gate = nn.Sequential(
            nn.Linear(5, 32), nn.GELU(), nn.Linear(32, nf), nn.Sigmoid()
        )
        ctx_dim = (25 + 3) if wide_thermo else 5
        head_in = nf + ctx_dim
        # deep_head_indices: list of property indices that get deep heads
        # (e.g., [7] for lignin only). All others get shallow heads.
        if deep_head_indices is None:
            deep_head_indices = list(range(n_props)) if depth == "deep" else []
        deep_set = set(deep_head_indices)
        heads = []
        for i in range(n_props):
            if i in deep_set:
                heads.append(nn.Sequential(
                    nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(64, 1),
                ))
            else:
                heads.append(nn.Sequential(
                    nn.Linear(head_in, 32), nn.GELU(), nn.Linear(32, 1)
                ))
        self.heads = nn.ModuleList(heads)
        self.alphas = nn.Parameter(torch.full((n_props,), -3.0))
        for h in self.heads:
            with torch.no_grad():
                h[-1].weight.mul_(0.01)
                h[-1].bias.zero_()

    def forward(self, v, i, t):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        if self.wide_thermo:
            # Interaction features: T×time, T×IL_conc, time×IL_conc
            interactions = torch.stack([
                t[:, 0] * t[:, 1],  # T × time
                t[:, 0] * t[:, 2],  # T × IL_conc
                t[:, 1] * t[:, 2],  # time × IL_conc
            ], dim=-1)
            ctx = torch.cat([t, interactions], -1)  # 25D + 3D = 28D
        else:
            ctx = tmp  # 5D
        inp = torch.cat([g, ctx], -1)
        res = torch.cat([h(inp) for h in self.heads], -1)
        return v + torch.sigmoid(self.alphas) * res


def load_split(split):
    """Load cached v4 features for a split. Returns dict of numpy arrays."""
    d = np.load(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.keys()}


def build_combined_40d():
    """Build the Combined(40D) feature matrices for train/val/test.

    Matches slurm_combined_sigma.sh exactly: PCA-20 fit on train only.
    """
    vj_tr = np.load(V5 / "data/cached_image_features_train.npz")["vit_feat"]
    vj_va = np.load(V5 / "data/cached_image_features_val.npz")["vit_feat"]
    vj_te = np.load(V5 / "data/cached_image_features_test.npz")["vit_feat"]

    sup = np.load(V5 / "data/supervised_vit_features.npz")["features"]
    sup_tr = sup[:152]
    sup_va = sup[152:152 + 32]
    sup_te = sup[152 + 32:]

    pca_vj = PCA(20).fit(vj_tr)
    pca_sup = PCA(20).fit(sup_tr)

    combo = {}
    for name, vj, sup_split in [
        ("train", vj_tr, sup_tr),
        ("val", vj_va, sup_va),
        ("test", vj_te, sup_te),
    ]:
        vj_p = pca_vj.transform(vj).astype(np.float32)
        sup_p = pca_sup.transform(sup_split).astype(np.float32)
        combo[name] = np.concatenate([vj_p, sup_p], axis=1)
    return combo


def _compute_prop_weights(targets):
    """Inverse-frequency weights so each property contributes equally to loss.

    Returns (n_props,) tensor where weight_j = total_valid / (n_props * valid_j).
    When all properties have the same count, all weights = 1.0.
    """
    valid_counts = (~torch.isnan(targets)).sum(dim=0).float()  # (n_props,)
    n_props = targets.shape[1]
    total_valid = valid_counts.sum()
    weights = total_valid / (n_props * valid_counts.clamp(min=1))
    return weights


def train_one_seed(seed, tr_v, tr_f, tr_th, tr_y, device, epochs=300, patience=50,
                   balance_props=True, depth="shallow", wide_thermo=False,
                   deep_head_indices=None):
    """Train a single PerPropHead seed. Returns the trained model."""
    set_seed(seed)
    n_props = tr_y.shape[1]
    model = PerPropHead(tr_f.shape[1], n_props=n_props, depth=depth,
                        wide_thermo=wide_thermo,
                        deep_head_indices=deep_head_indices).to(device)
    opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    tr_v_t = torch.from_numpy(tr_v).to(device)
    tr_f_t = torch.from_numpy(tr_f).to(device)
    tr_th_t = torch.from_numpy(tr_th).to(device)
    tr_y_t = torch.from_numpy(tr_y).to(device)

    prop_weights = _compute_prop_weights(tr_y_t).to(device) if balance_props else None

    ds = TensorDataset(tr_v_t.cpu(), tr_f_t.cpu(), tr_th_t.cpu(), tr_y_t.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    bad = 0

    for _ in range(epochs):
        model.train()
        for v, i, t, y in loader:
            v, i, t, y = v.to(device), i.to(device), t.to(device), y.to(device)
            pred = model(v, i, t)
            nan_mask = torch.isnan(y)
            valid = ~nan_mask
            if valid.sum() == 0:
                continue
            se = (pred - y.detach().clone().nan_to_num(0.0)) ** 2
            se[nan_mask] = 0.0
            if prop_weights is not None:
                se = se * prop_weights.unsqueeze(0)
            loss = se.sum() / valid.float().sum()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(tr_v_t, tr_f_t, tr_th_t)
            tr_y_clean = tr_y_t.detach().clone().nan_to_num(0.0)
            val_nan = torch.isnan(tr_y_t)
            val_valid = ~val_nan
            if val_valid.sum() == 0:
                continue
            val_se = (val_pred - tr_y_clean) ** 2
            val_se[val_nan] = 0.0
            if prop_weights is not None:
                val_se = val_se * prop_weights.unsqueeze(0)
            tl = (val_se.sum() / val_valid.float().sum()).item()
        if np.isfinite(tl) and tl < best_loss:
            best_loss = tl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, v, f, th, device):
    with torch.no_grad():
        return model(
            torch.from_numpy(v.astype(np.float32)).to(device),
            torch.from_numpy(f.astype(np.float32)).to(device),
            torch.from_numpy(th.astype(np.float32)).to(device),
        ).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--z_threshold", type=float, default=2.5,
                    help="Z-score threshold for flagging (per property)")
    ap.add_argument("--min_flagged_props", type=int, default=2,
                    help="Flag sample if |z| > threshold on at least this many properties")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_dir", type=str,
                    default=str(V5 / "results/residual_audit"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load all data ──
    print("Loading cached data...")
    tc = load_split("train")
    vc = load_split("val")
    tsc = load_split("test")

    # v4 base predictions per split
    def v4_base(c):
        return (0.4 * c["preds_fusion"] + 0.6 * c["preds_chemprop"]).astype(np.float32)

    v4_tr = v4_base(tc)
    # val and test use v4 ensemble predictions from seed_predictions
    # For simplicity we reuse train router on val/test via the cached preds;
    # the original script loads seed_predictions only for test. We do both.
    v4_va_path = PROJECT_ROOT / "cosmobridge_v4/results/seed_predictions"
    if v4_va_path.exists():
        sf = sorted(v4_va_path.glob("seed_*.npz"))
        if sf:
            sample = np.load(sf[0])
            key = "preds" if "preds" in sample else "predictions"
            v4_te = np.mean([np.load(f)[key] for f in sf], axis=0).astype(np.float32)
        else:
            v4_te = v4_base(tsc)
    else:
        v4_te = v4_base(tsc)
    v4_va = v4_base(vc)

    feats = build_combined_40d()

    tr_th = tc["thermo_feat"].astype(np.float32)
    va_th = vc["thermo_feat"].astype(np.float32)
    te_th = tsc["thermo_feat"].astype(np.float32)

    tr_y = tc["targets"].astype(np.float32)
    va_y = vc["targets"].astype(np.float32)
    te_y = tsc["targets"].astype(np.float32)

    n_tr, n_va, n_te = len(tr_y), len(va_y), len(te_y)
    print(f"  train={n_tr}  val={n_va}  test={n_te}  features={feats['train'].shape[1]}D")

    # ── Train n_seeds and collect per-sample predictions ──
    all_preds = {"train": [], "val": [], "test": []}
    for seed in range(args.n_seeds):
        print(f"Training seed {seed}...")
        model = train_one_seed(
            seed, v4_tr, feats["train"], tr_th, tr_y, device=device
        )
        all_preds["train"].append(predict(model, v4_tr, feats["train"], tr_th, device))
        all_preds["val"].append(predict(model, v4_va, feats["val"], va_th, device))
        all_preds["test"].append(predict(model, v4_te, feats["test"], te_th, device))

    # Ensemble mean
    mean_pred = {k: np.stack(all_preds[k]).mean(axis=0) for k in all_preds}

    # ── Metrics per split ──
    metrics = {
        "train": r2_per_prop(mean_pred["train"], tr_y),
        "val": r2_per_prop(mean_pred["val"], va_y),
        "test": r2_per_prop(mean_pred["test"], te_y),
    }
    print("\nEnsemble R² per split:")
    for s in ["train", "val", "test"]:
        print(f"  {s}: avg={metrics[s]['avg']:.4f}")
        for p in PROPS:
            print(f"    {p:8s}: {metrics[s][p]:.4f}")

    # ── Residuals + per-sample dataframe ──
    def residual_table(split_name, preds, targets, smiles, il_ids):
        rows = []
        resid = preds - targets  # (N, 7)
        # Z-score per property across this split
        z = (resid - resid.mean(axis=0, keepdims=True)) / (resid.std(axis=0, keepdims=True) + 1e-8)
        for i in range(len(targets)):
            row = {
                "split": split_name,
                "idx": i,
                "smiles": str(smiles[i]),
                "il_id": str(il_ids[i]) if i < len(il_ids) else "",
            }
            for j, p in enumerate(PROPS):
                row[f"{p}_target"] = float(targets[i, j])
                row[f"{p}_pred"] = float(preds[i, j])
                row[f"{p}_resid"] = float(resid[i, j])
                row[f"{p}_z"] = float(z[i, j])
            row["max_abs_z"] = float(np.max(np.abs(z[i])))
            row["n_props_flagged"] = int((np.abs(z[i]) > args.z_threshold).sum())
            rows.append(row)
        return rows

    all_rows = []
    all_rows += residual_table("train", mean_pred["train"], tr_y, tc["smiles"], tc["il_ids"])
    all_rows += residual_table("val", mean_pred["val"], va_y, vc["smiles"], vc["il_ids"])
    all_rows += residual_table("test", mean_pred["test"], te_y, tsc["smiles"], tsc["il_ids"])

    import csv
    resid_path = out_dir / "per_sample_residuals.csv"
    with open(resid_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {resid_path}")

    # ── Flagged samples (suspect labels) ──
    flagged = [r for r in all_rows if r["n_props_flagged"] >= args.min_flagged_props]
    flagged.sort(key=lambda r: -r["max_abs_z"])
    flag_path = out_dir / "flagged_samples.csv"
    with open(flag_path, "w", newline="") as f:
        if flagged:
            writer = csv.DictWriter(f, fieldnames=list(flagged[0].keys()))
            writer.writeheader()
            writer.writerows(flagged)
    print(f"Wrote {flag_path} ({len(flagged)} flagged samples)")

    # ── Per-property summary ──
    summary = {
        "n_seeds": args.n_seeds,
        "z_threshold": args.z_threshold,
        "min_flagged_props": args.min_flagged_props,
        "splits": {
            "train": {"n": n_tr, "r2": metrics["train"]},
            "val": {"n": n_va, "r2": metrics["val"]},
            "test": {"n": n_te, "r2": metrics["test"]},
        },
        "top_10_worst_per_property": {},
    }
    for j, p in enumerate(PROPS):
        ranked = sorted(all_rows, key=lambda r: -abs(r[f"{p}_z"]))[:10]
        summary["top_10_worst_per_property"][p] = [
            {
                "split": r["split"], "il_id": r["il_id"], "smiles": r["smiles"],
                "target": r[f"{p}_target"], "pred": r[f"{p}_pred"],
                "residual": r[f"{p}_resid"], "z": r[f"{p}_z"],
            }
            for r in ranked
        ]

    summary_path = out_dir / "per_property_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    # ── Train vs val overfitting diagnostic ──
    gap = {
        "train_r2": metrics["train"]["avg"],
        "val_r2": metrics["val"]["avg"],
        "test_r2": metrics["test"]["avg"],
        "train_minus_val": metrics["train"]["avg"] - metrics["val"]["avg"],
        "train_minus_test": metrics["train"]["avg"] - metrics["test"]["avg"],
        "interpretation": (
            "If train_minus_val > 0.10, head is overfitting and more regularization "
            "or SSL pretraining has the most runway. If gap < 0.05, label noise or "
            "representation ceiling is the bottleneck — Stage 2 and Stage 3 matter more."
        ),
    }
    gap_path = out_dir / "train_vs_val_gap.json"
    with open(gap_path, "w") as f:
        json.dump(gap, f, indent=2)
    print(f"Wrote {gap_path}")

    print("\nDone. Review flagged_samples.csv — samples appearing on multiple property top-10 lists")
    print("are likely label-noisy rather than model-noisy.")


if __name__ == "__main__":
    main()
