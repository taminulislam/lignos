"""Two-stage training: thermodynamic features → lignin transfer.

Stage 1: Train shallow model on all 8 properties (reproduces 0.843 best)
Stage 2: Freeze gate + core7 heads, train ONLY a fresh deep lignin head
"""
import json, sys, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import (PROPS, CORE_PROPS, PerPropHead, predict,
                              r2_per_prop, set_seed, train_one_seed,
                              _compute_prop_weights)


def load_split(split):
    d = np.load(V5 / "data/LignoIL" / f"cached_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    f, p = c.get("preds_fusion"), c.get("preds_chemprop")
    if f is not None and p is not None:
        return (0.4 * f + 0.6 * p).astype(np.float32)
    return np.zeros_like(c["targets"], dtype=np.float32)


def train_stage2_lignin(stage1_model, tr_v, tr_f, tr_th, tr_y, device,
                         epochs=300, patience=50, seed=0):
    """Freeze stage1 gate + core7 heads, train fresh deep lignin head."""
    set_seed(seed)
    model = copy.deepcopy(stage1_model).to(device)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Replace lignin head (index 7) with fresh deep head
    nf = model.gate[2].out_features  # gate output dim = nf
    head_in = nf + 5  # narrow thermo
    model.heads[7] = nn.Sequential(
        nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(64, 1),
    ).to(device)
    # Init small weights
    with torch.no_grad():
        model.heads[7][-1].weight.mul_(0.01)
        model.heads[7][-1].bias.zero_()

    # Unfreeze ONLY lignin head + its alpha
    for param in model.heads[7].parameters():
        param.requires_grad = True
    model.alphas.requires_grad = True  # allow alpha tuning for all props

    # Only optimize lignin head params + alphas
    opt_params = list(model.heads[7].parameters()) + [model.alphas]
    opt = AdamW(opt_params, lr=1e-3, weight_decay=1e-2)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    tr_v_t = torch.from_numpy(tr_v).to(device)
    tr_f_t = torch.from_numpy(tr_f).to(device)
    tr_th_t = torch.from_numpy(tr_th).to(device)
    tr_y_t = torch.from_numpy(tr_y).to(device)

    ds = TensorDataset(tr_v_t.cpu(), tr_f_t.cpu(), tr_th_t.cpu(), tr_y_t.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    bad = 0

    for epoch in range(epochs):
        model.train()
        for v, i, t, y in loader:
            v, i, t, y = v.to(device), i.to(device), t.to(device), y.to(device)
            pred = model(v, i, t)
            # Only compute loss on lignin column (index 7)
            lig_mask = ~torch.isnan(y[:, 7])
            if lig_mask.sum() == 0:
                continue
            loss = ((pred[lig_mask, 7] - y[lig_mask, 7].detach().clone().nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            opt.step()
        sched.step()

        # Validation on lignin
        model.eval()
        with torch.no_grad():
            val_pred = model(tr_v_t, tr_f_t, tr_th_t)
            lig_valid = ~torch.isnan(tr_y_t[:, 7])
            if lig_valid.sum() > 0:
                tl = ((val_pred[lig_valid, 7] - tr_y_t[lig_valid, 7].nan_to_num(0)) ** 2).mean().item()
            else:
                continue
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tr = load_split("train")
    te = load_split("test")

    pca = PCA(40).fit(tr["morgan_fp"])
    f_tr = pca.transform(tr["morgan_fp"]).astype(np.float32)
    f_te = pca.transform(te["morgan_fp"]).astype(np.float32)

    v4_tr, v4_te = v4_base(tr), v4_base(te)
    y_tr = tr["targets"].astype(np.float32)
    y_te = te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]

    results = []
    n_seeds = 10

    # ================================================================
    # Stage 1: Train shallow model (reproduces best config)
    # ================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: Shallow model on all 8 properties")
    print(f"{'='*60}")

    stage1_models = []
    stage1_r2s = []
    for seed in range(n_seeds):
        model = train_one_seed(seed, v4_tr, f_tr, th_tr, y_tr, device=device,
                                balance_props=False, depth="shallow", wide_thermo=False)
        stage1_models.append(model)
        te_pred = predict(model, v4_te, f_te, th_te, device)
        r2 = r2_per_prop(te_pred, y_te)
        stage1_r2s.append(r2)

    avg_core7 = float(np.mean([m["avg_core7"] for m in stage1_r2s]))
    std_core7 = float(np.std([m["avg_core7"] for m in stage1_r2s]))
    print(f"  Stage 1: R2_core7 = {avg_core7:.4f} +/- {std_core7:.4f}")
    per_prop_s1 = {}
    for p in PROPS:
        vals = [m.get(p) for m in stage1_r2s if m.get(p) is not None and not np.isnan(m.get(p, float("nan")))]
        per_prop_s1[p] = float(np.mean(vals)) if vals else float("nan")
        print(f"    {p}: {per_prop_s1[p]:.4f}")
    results.append({"name": "Stage1_shallow", "avg_r2_core7": avg_core7,
                     "std_r2_core7": std_core7, "per_prop": per_prop_s1})

    # ================================================================
    # Stage 2: Freeze gate+core7, train deep lignin head
    # ================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: Freeze gate+core7, train deep lignin head")
    print(f"{'='*60}")

    stage2_r2s = []
    for seed in range(n_seeds):
        # Take the stage1 model for this seed
        s1_model = stage1_models[seed]
        # Train stage2 on top
        s2_model = train_stage2_lignin(s1_model, v4_tr, f_tr, th_tr, y_tr,
                                        device=device, seed=seed + 100)
        te_pred = predict(s2_model, v4_te, f_te, th_te, device)
        r2 = r2_per_prop(te_pred, y_te)
        stage2_r2s.append(r2)

    avg_core7_s2 = float(np.mean([m["avg_core7"] for m in stage2_r2s]))
    std_core7_s2 = float(np.std([m["avg_core7"] for m in stage2_r2s]))
    print(f"  Stage 2: R2_core7 = {avg_core7_s2:.4f} +/- {std_core7_s2:.4f}")
    per_prop_s2 = {}
    for p in PROPS:
        vals = [m.get(p) for m in stage2_r2s if m.get(p) is not None and not np.isnan(m.get(p, float("nan")))]
        per_prop_s2[p] = float(np.mean(vals)) if vals else float("nan")
        print(f"    {p}: {per_prop_s2[p]:.4f}")
    results.append({"name": "Stage2_frozen_deep_lignin", "avg_r2_core7": avg_core7_s2,
                     "std_r2_core7": std_core7_s2, "per_prop": per_prop_s2})

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print("TWO-STAGE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Stage':<35} {'core7':>7} {'std':>7} {'lignin':>8}")
    print("-" * 60)
    for r in results:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<35} {r['avg_r2_core7']:>7.4f} {r['std_r2_core7']:>7.4f} {lig:>8.4f}")

    print(f"\nCore7 change after stage 2: {avg_core7_s2 - avg_core7:+.4f}")
    print(f"Lignin change after stage 2: {per_prop_s2.get('lignin_wt', 0) - per_prop_s1.get('lignin_wt', 0):+.4f}")

    out = V5 / "results" / "two_stage_comparison.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
