"""Two-stage training with hard-freeze on the EXTENDED LignoIL_unified cache.

Compares the prior hardfreeze winner against the same architecture fed the
new `[thermo_feat (25D), physchem_feat (12D)]` → 37D context. Runs both on
the same set of seeds so the comparison is apples-to-apples.

Variants in one job
-------------------
  1) Baseline_stage1        — shallow PerPropHead on the original LignoIL cache
  2) Baseline_stage2_hardfz — hardfreeze stage-2 transfer on the baseline
  3) Extended_stage1        — shallow PerPropHeadExt on LignoIL_unified with physchem
  4) Extended_stage2_hardfz — hardfreeze stage-2 transfer on the extended model

Key changes vs `train_two_stage_hardfreeze.py`
-----------------------------------------------
  * New model class `PerPropHeadExt` concatenates physchem features (gated by
    `has_physchem` to zero-fill rows without coverage) into the head context.
  * Physchem preprocessing: log1p on viscosity and conductivity (span ≥3
    decades); z-score with train-split statistics computed on rows that have
    physchem; applied to val/test consistently.
  * Hardfreeze stage-2: same alpha grad-mask + per-group wd=0 fix.
"""
from __future__ import annotations
import copy, json, sys
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
from audit_residuals import PROPS, CORE_PROPS, r2_per_prop, set_seed  # noqa: E402

BASELINE_DIR = V5 / "data" / "LignoIL"
EXTENDED_DIR = V5 / "data" / "LignoIL_unified"
PHYSCHEM_DIM = 12
N_SEEDS = 10


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_split(data_dir, split):
    d = np.load(data_dir / f"cached_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    f, p = c.get("preds_fusion"), c.get("preds_chemprop")
    return (0.4 * f + 0.6 * p).astype(np.float32)


def preprocess_physchem_train(phys_feat, has_physchem):
    """Fit log-transform + z-score on train rows with physchem. Returns (X_proc, stats)."""
    x = phys_feat.astype(np.float32).copy()
    # Log-transform viscosity (idx 3) and conductivity (idx 5) — they span ≥3 decades.
    x[:, 3] = np.log1p(np.maximum(x[:, 3], 0.0))
    x[:, 5] = np.log1p(np.maximum(x[:, 5], 0.0))
    covered = has_physchem.astype(bool)
    if covered.sum() > 0:
        mu = x[covered].mean(axis=0)
        sd = x[covered].std(axis=0) + 1e-6
    else:
        mu = np.zeros(x.shape[1], dtype=np.float32)
        sd = np.ones(x.shape[1], dtype=np.float32)
    z = (x - mu) / sd
    z = z * covered[:, None]
    return z.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def preprocess_physchem_apply(phys_feat, has_physchem, mu, sd):
    x = phys_feat.astype(np.float32).copy()
    x[:, 3] = np.log1p(np.maximum(x[:, 3], 0.0))
    x[:, 5] = np.log1p(np.maximum(x[:, 5], 0.0))
    covered = has_physchem.astype(bool)
    z = (x - mu) / sd
    z = z * covered[:, None]
    return z.astype(np.float32)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class PerPropHead(nn.Module):
    """Faithful copy of the baseline architecture (narrow thermo, no physchem)."""
    def __init__(self, nf, n_props=8, deep_indices=None):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(5, 32), nn.GELU(), nn.Linear(32, nf), nn.Sigmoid()
        )
        ctx_dim = 5
        head_in = nf + ctx_dim
        deep_set = set(deep_indices or [])
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

    def forward(self, v, i, t, phys=None):  # phys unused for baseline
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        ctx = tmp
        inp = torch.cat([g, ctx], -1)
        res = torch.cat([h(inp) for h in self.heads], -1)
        return v + torch.sigmoid(self.alphas) * res


class PerPropHeadExt(nn.Module):
    """Extended: ctx gets +12-D physchem appended."""
    def __init__(self, nf, n_props=8, physchem_dim=PHYSCHEM_DIM, deep_indices=None):
        super().__init__()
        self.physchem_dim = physchem_dim
        self.gate = nn.Sequential(
            nn.Linear(5, 32), nn.GELU(), nn.Linear(32, nf), nn.Sigmoid()
        )
        ctx_dim = 5 + physchem_dim
        head_in = nf + ctx_dim
        deep_set = set(deep_indices or [])
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

    def forward(self, v, i, t, phys):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        ctx = torch.cat([tmp, phys], -1)
        inp = torch.cat([g, ctx], -1)
        res = torch.cat([h(inp) for h in self.heads], -1)
        return v + torch.sigmoid(self.alphas) * res


# ----------------------------------------------------------------------
# Training + eval
# ----------------------------------------------------------------------
def _compute_prop_weights(targets):
    valid = (~torch.isnan(targets)).sum(dim=0).float()
    n_props = targets.shape[1]
    total = valid.sum()
    return total / (n_props * valid.clamp(min=1))


def train_one_seed(model_cls, seed, tr_v, tr_f, tr_th, tr_phys, tr_y, device,
                    epochs=300, patience=50, balance_props=False):
    set_seed(seed)
    n_props = tr_y.shape[1]
    if model_cls is PerPropHeadExt:
        model = model_cls(tr_f.shape[1], n_props=n_props).to(device)
    else:
        model = model_cls(tr_f.shape[1], n_props=n_props).to(device)
    opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    tr_v_t = torch.from_numpy(tr_v).to(device)
    tr_f_t = torch.from_numpy(tr_f).to(device)
    tr_th_t = torch.from_numpy(tr_th).to(device)
    tr_phys_t = torch.from_numpy(tr_phys).to(device) if tr_phys is not None else None
    tr_y_t = torch.from_numpy(tr_y).to(device)

    valid_mask = ~torch.isnan(tr_y_t)
    y_fill = torch.nan_to_num(tr_y_t, nan=0.0)
    weights = _compute_prop_weights(tr_y_t) if balance_props else torch.ones(n_props, device=device)

    ds = TensorDataset(
        tr_v_t.cpu(), tr_f_t.cpu(), tr_th_t.cpu(),
        (tr_phys_t.cpu() if tr_phys_t is not None else torch.zeros(len(tr_v_t), 1)),
        y_fill.cpu(), valid_mask.cpu()
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best_loss = float("inf"); best_state = None; bad = 0
    for ep in range(epochs):
        model.train()
        for v, im, t, p, y, m in loader:
            v, im, t, y, m = v.to(device), im.to(device), t.to(device), y.to(device), m.to(device)
            p = p.to(device) if tr_phys_t is not None else None
            pred = model(v, im, t, p)
            err2 = ((pred - y) ** 2) * m
            per_prop = err2.sum(0) / m.sum(0).clamp(min=1)
            loss = (per_prop * weights).sum() / weights.sum()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(tr_v_t, tr_f_t, tr_th_t, tr_phys_t)
            err2 = ((pred - y_fill) ** 2) * valid_mask
            per_prop = err2.sum(0) / valid_mask.sum(0).clamp(min=1)
            tl = (per_prop * weights).sum().item() / weights.sum().item()
        if np.isfinite(tl) and tl < best_loss:
            best_loss = tl
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_ext(model, v, f, th, phys, device):
    v_t = torch.from_numpy(v).to(device)
    f_t = torch.from_numpy(f).to(device)
    th_t = torch.from_numpy(th).to(device)
    p_t = torch.from_numpy(phys).to(device) if phys is not None else None
    model.eval()
    with torch.no_grad():
        out = model(v_t, f_t, th_t, p_t)
    return out.cpu().numpy()


def _build_deep_lignin_head(model, device, include_physchem):
    nf = model.gate[2].out_features
    ctx_dim = 5 + (PHYSCHEM_DIM if include_physchem else 0)
    head_in = nf + ctx_dim
    model.heads[7] = nn.Sequential(
        nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(64, 1),
    ).to(device)
    with torch.no_grad():
        model.heads[7][-1].weight.mul_(0.01)
        model.heads[7][-1].bias.zero_()


def train_stage2_hardfreeze(stage1_model, tr_v, tr_f, tr_th, tr_phys, tr_y, device, seed,
                             include_physchem, epochs=300, patience=50):
    set_seed(seed)
    model = copy.deepcopy(stage1_model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    _build_deep_lignin_head(model, device, include_physchem)
    for p in model.heads[7].parameters():
        p.requires_grad = True
    model.alphas.requires_grad = True
    mask = torch.zeros_like(model.alphas); mask[7] = 1.0
    model.alphas.register_hook(lambda g: g * mask)

    head7_params = list(model.heads[7].parameters())
    opt = AdamW([
        {"params": head7_params, "weight_decay": 1e-2},
        {"params": [model.alphas], "weight_decay": 0.0},
    ], lr=1e-3)
    opt_params = head7_params + [model.alphas]
    sched = CosineAnnealingLR(opt, T_max=epochs)

    tr_v_t = torch.from_numpy(tr_v).to(device)
    tr_f_t = torch.from_numpy(tr_f).to(device)
    tr_th_t = torch.from_numpy(tr_th).to(device)
    tr_phys_t = torch.from_numpy(tr_phys).to(device) if tr_phys is not None else None
    tr_y_t = torch.from_numpy(tr_y).to(device)

    ds = TensorDataset(
        tr_v_t.cpu(), tr_f_t.cpu(), tr_th_t.cpu(),
        (tr_phys_t.cpu() if tr_phys_t is not None else torch.zeros(len(tr_v_t), 1)),
        tr_y_t.cpu()
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best_loss = float("inf"); best_state = None; bad = 0
    for ep in range(epochs):
        model.train()
        for v, im, t, p, y in loader:
            v, im, t, y = v.to(device), im.to(device), t.to(device), y.to(device)
            p = p.to(device) if tr_phys_t is not None else None
            pred = model(v, im, t, p)
            lig = ~torch.isnan(y[:, 7])
            if lig.sum() == 0:
                continue
            loss = ((pred[lig, 7] - y[lig, 7].nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(tr_v_t, tr_f_t, tr_th_t, tr_phys_t)
            lig = ~torch.isnan(tr_y_t[:, 7])
            tl = ((pred[lig, 7] - tr_y_t[lig, 7].nan_to_num(0)) ** 2).mean().item() if lig.any() else float("inf")
        if np.isfinite(tl) and tl < best_loss:
            best_loss = tl
            best_state = {k: vv.clone() for k, vv in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def summarize(name, r2s):
    core7 = [r["avg_core7"] for r in r2s]
    out = {"name": name, "avg_r2_core7": float(np.mean(core7)),
           "std_r2_core7": float(np.std(core7)), "per_prop": {}}
    for p in PROPS:
        vals = [r.get(p) for r in r2s if r.get(p) is not None and not np.isnan(r.get(p, float("nan")))]
        out["per_prop"][p] = float(np.mean(vals)) if vals else float("nan")
    return out


def run_experiment(name, data_dir, model_cls, include_physchem, device):
    print(f"\n{'='*60}\nEXPERIMENT: {name}  (dir={data_dir.name})\n{'='*60}")
    tr = load_split(data_dir, "train")
    te = load_split(data_dir, "test")
    pca = PCA(40).fit(tr["morgan_fp"])
    f_tr = pca.transform(tr["morgan_fp"]).astype(np.float32)
    f_te = pca.transform(te["morgan_fp"]).astype(np.float32)
    v4_tr, v4_te = v4_base(tr), v4_base(te)
    y_tr = tr["targets"].astype(np.float32)
    y_te = te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]

    if include_physchem:
        p_tr, mu, sd = preprocess_physchem_train(tr["physchem_feat"], tr["has_physchem"])
        p_te = preprocess_physchem_apply(te["physchem_feat"], te["has_physchem"], mu, sd)
        print(f"  physchem: train rows with coverage = {int(tr['has_physchem'].sum())}/{len(tr['has_physchem'])}; "
              f"test coverage = {int(te['has_physchem'].sum())}/{len(te['has_physchem'])}")
    else:
        p_tr = p_te = None

    # Stage 1
    stage1_models, stage1_r2s = [], []
    for seed in range(N_SEEDS):
        m = train_one_seed(model_cls, seed, v4_tr, f_tr, th_tr, p_tr, y_tr, device=device,
                            balance_props=False)
        stage1_models.append(m)
        te_pred = predict_ext(m, v4_te, f_te, th_te, p_te, device)
        stage1_r2s.append(r2_per_prop(te_pred, y_te))
    s1 = summarize(f"{name}/Stage1", stage1_r2s)
    print(f"  Stage1 core7 = {s1['avg_r2_core7']:.4f} ± {s1['std_r2_core7']:.4f}")
    for p in PROPS:
        print(f"    {p:12s}: {s1['per_prop'][p]:.4f}")

    # Stage 2 hardfreeze
    stage2_r2s = []
    for seed in range(N_SEEDS):
        s2 = train_stage2_hardfreeze(stage1_models[seed], v4_tr, f_tr, th_tr, p_tr, y_tr,
                                      device=device, seed=seed + 100,
                                      include_physchem=include_physchem)
        te_pred = predict_ext(s2, v4_te, f_te, th_te, p_te, device)
        stage2_r2s.append(r2_per_prop(te_pred, y_te))
    s2s = summarize(f"{name}/Stage2_hardfreeze", stage2_r2s)
    print(f"  Stage2_hardfreeze core7 = {s2s['avg_r2_core7']:.4f} ± {s2s['std_r2_core7']:.4f}")
    for p in PROPS:
        print(f"    {p:12s}: {s2s['per_prop'][p]:.4f}")
    return s1, s2s


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    results = []

    b1, b2 = run_experiment("Baseline", BASELINE_DIR, PerPropHead,
                             include_physchem=False, device=device)
    results += [b1, b2]

    e1, e2 = run_experiment("Extended", EXTENDED_DIR, PerPropHeadExt,
                             include_physchem=True, device=device)
    results += [e1, e2]

    print(f"\n{'='*66}\nFINAL COMPARISON (Extended cache + physchem vs Baseline)\n{'='*66}")
    print(f"{'Variant':<38}{'core7':>9}{'std':>9}{'lignin':>10}")
    print("-" * 66)
    for r in results:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<38}{r['avg_r2_core7']:>9.4f}{r['std_r2_core7']:>9.4f}{lig:>10.4f}")

    out = V5 / "results" / "two_stage_extended_comparison.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
