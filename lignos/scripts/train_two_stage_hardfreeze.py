"""Two-stage training with hard-freeze on core7 alphas.

Problem with the original stage2: `model.alphas` is a single Parameter of shape (8,);
the optimizer applies AdamW weight_decay to the full tensor, so alpha[0..6] drift
toward zero even though only lignin (index 7) has gradient. That shifts
sigmoid(alpha[0..6]) and therefore core7 residual gating — source of the -0.0037
core7 regression.

This script runs three variants head-to-head (same stage1 backbone):
  1) Stage2_original      — current behavior (for reproduction)
  2) Stage2_hardfreeze    — alpha in its own param group with weight_decay=0
                            + gradient hook zeroing alpha[0..6] (recommended)
  3) Stage2_no_alpha      — alpha entirely frozen (alpha[7] also fixed)

Goal: confirm hardfreeze preserves stage1 core7 exactly while keeping lignin gain.
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
from audit_residuals import (PROPS, CORE_PROPS, predict, r2_per_prop, set_seed,
                              train_one_seed)


def load_split(split):
    d = np.load(V5 / "data/LignoIL" / f"cached_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    f, p = c.get("preds_fusion"), c.get("preds_chemprop")
    if f is not None and p is not None:
        return (0.4 * f + 0.6 * p).astype(np.float32)
    return np.zeros_like(c["targets"], dtype=np.float32)


def _replace_lignin_head(model, device):
    nf = model.gate[2].out_features
    head_in = nf + 5
    model.heads[7] = nn.Sequential(
        nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(64, 1),
    ).to(device)
    with torch.no_grad():
        model.heads[7][-1].weight.mul_(0.01)
        model.heads[7][-1].bias.zero_()


def _train_lignin_loop(model, opt_params, opt, tr_v, tr_f, tr_th, tr_y, device,
                        epochs=300, patience=50):
    tr_v_t = torch.from_numpy(tr_v).to(device)
    tr_f_t = torch.from_numpy(tr_f).to(device)
    tr_th_t = torch.from_numpy(tr_th).to(device)
    tr_y_t = torch.from_numpy(tr_y).to(device)
    ds = TensorDataset(tr_v_t.cpu(), tr_f_t.cpu(), tr_th_t.cpu(), tr_y_t.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    bad = 0

    for epoch in range(epochs):
        model.train()
        for v, i, t, y in loader:
            v, i, t, y = v.to(device), i.to(device), t.to(device), y.to(device)
            pred = model(v, i, t)
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

        model.eval()
        with torch.no_grad():
            val_pred = model(tr_v_t, tr_f_t, tr_th_t)
            lig_valid = ~torch.isnan(tr_y_t[:, 7])
            if lig_valid.sum() == 0:
                continue
            tl = ((val_pred[lig_valid, 7] - tr_y_t[lig_valid, 7].nan_to_num(0)) ** 2).mean().item()
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


def train_stage2_original(stage1_model, tr_v, tr_f, tr_th, tr_y, device, seed=0):
    set_seed(seed)
    model = copy.deepcopy(stage1_model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    _replace_lignin_head(model, device)
    for p in model.heads[7].parameters():
        p.requires_grad = True
    model.alphas.requires_grad = True
    opt_params = list(model.heads[7].parameters()) + [model.alphas]
    opt = AdamW(opt_params, lr=1e-3, weight_decay=1e-2)
    return _train_lignin_loop(model, opt_params, opt, tr_v, tr_f, tr_th, tr_y, device)


def train_stage2_hardfreeze(stage1_model, tr_v, tr_f, tr_th, tr_y, device, seed=0):
    """Fix: alpha in separate param group with wd=0, plus grad-mask on alpha[0..6]."""
    set_seed(seed)
    model = copy.deepcopy(stage1_model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    _replace_lignin_head(model, device)
    for p in model.heads[7].parameters():
        p.requires_grad = True
    model.alphas.requires_grad = True

    # Gradient mask: zero out grads for alpha[0..6], keep alpha[7]
    mask = torch.zeros_like(model.alphas)
    mask[7] = 1.0
    model.alphas.register_hook(lambda g: g * mask)

    head7_params = list(model.heads[7].parameters())
    opt = AdamW([
        {"params": head7_params, "weight_decay": 1e-2},
        {"params": [model.alphas], "weight_decay": 0.0},
    ], lr=1e-3)
    opt_params = head7_params + [model.alphas]
    return _train_lignin_loop(model, opt_params, opt, tr_v, tr_f, tr_th, tr_y, device)


def train_stage2_no_alpha(stage1_model, tr_v, tr_f, tr_th, tr_y, device, seed=0):
    """Most conservative: alpha completely frozen, only head[7] trains."""
    set_seed(seed)
    model = copy.deepcopy(stage1_model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    _replace_lignin_head(model, device)
    for p in model.heads[7].parameters():
        p.requires_grad = True
    # alpha stays frozen
    opt_params = list(model.heads[7].parameters())
    opt = AdamW(opt_params, lr=1e-3, weight_decay=1e-2)
    return _train_lignin_loop(model, opt_params, opt, tr_v, tr_f, tr_th, tr_y, device)


def eval_seeds(stage2_fn, stage1_models, v4_tr, f_tr, th_tr, y_tr, v4_te, f_te, th_te, y_te, device, n_seeds):
    r2s = []
    for seed in range(n_seeds):
        s2_model = stage2_fn(stage1_models[seed], v4_tr, f_tr, th_tr, y_tr,
                              device=device, seed=seed + 100)
        te_pred = predict(s2_model, v4_te, f_te, th_te, device)
        r2s.append(r2_per_prop(te_pred, y_te))
    return r2s


def summarize(name, r2s):
    core7 = [r["avg_core7"] for r in r2s]
    out = {
        "name": name,
        "avg_r2_core7": float(np.mean(core7)),
        "std_r2_core7": float(np.std(core7)),
        "per_prop": {},
    }
    for p in PROPS:
        vals = [r.get(p) for r in r2s if r.get(p) is not None and not np.isnan(r.get(p, float("nan")))]
        out["per_prop"][p] = float(np.mean(vals)) if vals else float("nan")
    return out


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

    n_seeds = 10
    results = []

    print(f"\n{'='*60}\nSTAGE 1: shallow model on all 8 props (10 seeds)\n{'='*60}")
    stage1_models, stage1_r2s = [], []
    for seed in range(n_seeds):
        m = train_one_seed(seed, v4_tr, f_tr, th_tr, y_tr, device=device,
                            balance_props=False, depth="shallow", wide_thermo=False)
        stage1_models.append(m)
        stage1_r2s.append(r2_per_prop(predict(m, v4_te, f_te, th_te, device), y_te))
    s1 = summarize("Stage1_shallow", stage1_r2s)
    print(f"  Stage1 core7 = {s1['avg_r2_core7']:.4f} ± {s1['std_r2_core7']:.4f}")
    for p in PROPS:
        print(f"    {p:12s}: {s1['per_prop'][p]:.4f}")
    results.append(s1)

    for fn, label in [(train_stage2_original, "Stage2_original"),
                       (train_stage2_hardfreeze, "Stage2_hardfreeze"),
                       (train_stage2_no_alpha, "Stage2_no_alpha")]:
        print(f"\n{'='*60}\n{label}: 10-seed evaluation\n{'='*60}")
        r2s = eval_seeds(fn, stage1_models, v4_tr, f_tr, th_tr, y_tr,
                          v4_te, f_te, th_te, y_te, device, n_seeds)
        s = summarize(label, r2s)
        print(f"  {label}: core7 = {s['avg_r2_core7']:.4f} ± {s['std_r2_core7']:.4f}")
        for p in PROPS:
            print(f"    {p:12s}: {s['per_prop'][p]:.4f}")
        results.append(s)

    print(f"\n{'='*60}\nFINAL COMPARISON\n{'='*60}")
    print(f"{'Variant':<30}{'core7':>8}{'std':>8}{'lignin':>10}{'Δcore7':>10}")
    print("-" * 66)
    base_core7 = results[0]["avg_r2_core7"]
    for r in results:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        d = r["avg_r2_core7"] - base_core7
        print(f"{r['name']:<30}{r['avg_r2_core7']:>8.4f}{r['std_r2_core7']:>8.4f}{lig:>10.4f}{d:>+10.4f}")

    out_path = V5 / "results" / "two_stage_hardfreeze_comparison.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
