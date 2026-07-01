"""A4: Kendall uncertainty-weighted multi-task loss on the current-winner setup.

Reproduces Stage2_hardfreeze on the Baseline LignoIL cache (the 0.8272 / 0.6222
winner) but swaps the equal-weighted per-property MSE mean for the Kendall
loss:

    L = Σⱼ [ (1/(2 σⱼ²)) · MSE(j) + log σⱼ ]
       = Σⱼ [ (1/2) · exp(-log_var_j) · MSE(j) + (1/2) · log_var_j ]

with `log_var_j` as 8 learnable scalar parameters (initialized at 0, so σ=1).
Rare tasks (γ₂, G^E, G_mix, H_vap with only 152 rows) should end up with
HIGHER σ (lower weight), preventing them from dominating gradients and
letting the abundant tasks (γ₁, H^E, P) converge cleanly.

Four variants, same 10 seeds:
  1. Baseline / Stage1                   — reproduces the winner Stage-1
  2. Baseline / Stage2_hardfreeze        — reproduces the winner Stage-2
  3. A4 / Stage1 (uncertainty-weighted)  — tests whether UW helps Stage-1
  4. A4 / Stage2_hardfreeze              — uses UW stage-1 backbone for transfer
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

CACHE = V5 / "data" / "LignoIL"
N_SEEDS = 10


def load_split(s):
    d = np.load(CACHE / f"cached_{s}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    return (0.4 * c["preds_fusion"] + 0.6 * c["preds_chemprop"]).astype(np.float32)


class PerPropHead(nn.Module):
    """Baseline shallow head (matches audit_residuals.PerPropHead narrow-thermo path)."""
    def __init__(self, nf, n_props=8):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, nf), nn.Sigmoid())
        ctx = 5; head_in = nf + ctx
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 32), nn.GELU(), nn.Linear(32, 1)) for _ in range(n_props)
        ])
        self.alphas = nn.Parameter(torch.full((n_props,), -3.0))
        for h in self.heads:
            with torch.no_grad():
                h[-1].weight.mul_(0.01); h[-1].bias.zero_()

    def forward(self, v, i, t):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        inp = torch.cat([g, tmp], -1)
        res = torch.cat([h(inp) for h in self.heads], -1)
        return v + torch.sigmoid(self.alphas) * res


def uncertainty_weighted_loss(pred, y, valid, log_var):
    """Kendall uncertainty-weighted multi-task loss.

    pred, y, valid: (N, K). log_var: (K,).
    For each task j: (1/2) * exp(-log_var_j) * MSE_j + (1/2) * log_var_j
    MSE_j is computed over VALID entries only (per-sample masking).
    """
    err2 = ((pred - y) ** 2) * valid
    per_prop_mse = err2.sum(0) / valid.sum(0).clamp(min=1)
    precision = torch.exp(-log_var)
    per_prop_loss = 0.5 * precision * per_prop_mse + 0.5 * log_var
    return per_prop_loss.sum()


def train_stage1(seed, tr_v, tr_f, tr_th, tr_y, device, use_uw,
                   epochs=300, patience=50):
    set_seed(seed)
    n_props = tr_y.shape[1]
    m = PerPropHead(tr_f.shape[1], n_props).to(device)
    log_var = (nn.Parameter(torch.zeros(n_props, device=device)) if use_uw
               else torch.zeros(n_props, device=device))
    params = list(m.parameters())
    if use_uw:
        params = params + [log_var]
    opt = AdamW(params, lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    v_t = torch.from_numpy(tr_v).to(device)
    f_t = torch.from_numpy(tr_f).to(device)
    t_t = torch.from_numpy(tr_th).to(device)
    y_t = torch.from_numpy(tr_y).to(device)
    valid = ~torch.isnan(y_t); yf = torch.nan_to_num(y_t, 0.0)
    ds = TensorDataset(v_t.cpu(), f_t.cpu(), t_t.cpu(), yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for vb, ib, tb, yb, vm in loader:
            vb, ib, tb, yb, vm = [x.to(device) for x in (vb, ib, tb, yb, vm)]
            pred = m(vb, ib, tb)
            if use_uw:
                loss = uncertainty_weighted_loss(pred, yb, vm.float(), log_var)
            else:
                err2 = ((pred - yb) ** 2) * vm.float()
                loss = (err2.sum(0) / vm.float().sum(0).clamp(min=1)).mean()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(v_t, f_t, t_t)
            err2 = ((pred - yf) ** 2) * valid.float()
            tl = (err2.sum(0) / valid.float().sum(0).clamp(min=1)).mean().item()
        if np.isfinite(tl) and tl < best:
            best, state, bad = tl, {k: vv.clone() for k, vv in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval()
    # Report learned log_var for visibility
    if use_uw:
        lv = log_var.detach().cpu().numpy()
        print(f"    learned log_var: {lv.round(2)}  sigma: {np.exp(lv/2).round(2)}")
    return m


def predict(m, v, f, t, device):
    vt, ft, tt = [torch.from_numpy(x).to(device) for x in (v, f, t)]
    m.eval()
    with torch.no_grad():
        return m(vt, ft, tt).cpu().numpy()


def train_stage2_hardfz(s1, tr_v, tr_f, tr_th, tr_y, device, seed,
                          epochs=300, patience=50):
    """Hardfreeze stage-2: freeze backbone + core7 heads, train deep lignin head.
    Lignin-only MSE — uncertainty weighting doesn't apply to a single-task stage."""
    set_seed(seed)
    m = copy.deepcopy(s1).to(device)
    for p in m.parameters(): p.requires_grad = False
    nf = m.gate[2].out_features
    m.heads[7] = nn.Sequential(
        nn.Linear(nf + 5, 128), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(64, 1),
    ).to(device)
    with torch.no_grad():
        m.heads[7][-1].weight.mul_(0.01); m.heads[7][-1].bias.zero_()
    for p in m.heads[7].parameters(): p.requires_grad = True
    m.alphas.requires_grad = True
    mask = torch.zeros_like(m.alphas); mask[7] = 1.0
    m.alphas.register_hook(lambda g: g * mask)
    hp = list(m.heads[7].parameters())
    opt = AdamW([{"params": hp, "weight_decay": 1e-2},
                  {"params": [m.alphas], "weight_decay": 0.0}], lr=1e-3)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    opts = hp + [m.alphas]

    v_t = torch.from_numpy(tr_v).to(device)
    f_t = torch.from_numpy(tr_f).to(device)
    t_t = torch.from_numpy(tr_th).to(device)
    y_t = torch.from_numpy(tr_y).to(device)
    ds = TensorDataset(v_t.cpu(), f_t.cpu(), t_t.cpu(), y_t.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for vb, ib, tb, yb in loader:
            vb, ib, tb, yb = [x.to(device) for x in (vb, ib, tb, yb)]
            pred = m(vb, ib, tb)
            lg = ~torch.isnan(yb[:, 7])
            if lg.sum() == 0: continue
            loss = ((pred[lg, 7] - yb[lg, 7].nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(opts, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(v_t, f_t, t_t)
            lg = ~torch.isnan(y_t[:, 7])
            tl = ((pred[lg, 7] - y_t[lg, 7].nan_to_num(0)) ** 2).mean().item() if lg.any() else float("inf")
        if np.isfinite(tl) and tl < best:
            best, state, bad = tl, {k: vv.clone() for k, vv in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval()
    return m


def summarize(name, r2s):
    c = [r["avg_core7"] for r in r2s]
    out = {"name": name, "avg_r2_core7": float(np.mean(c)),
           "std_r2_core7": float(np.std(c)), "per_prop": {}}
    for p in PROPS:
        vs = [r.get(p) for r in r2s if r.get(p) is not None and not np.isnan(r.get(p, float("nan")))]
        out["per_prop"][p] = float(np.mean(vs)) if vs else float("nan")
    return out


def run(label, use_uw, device, tr_cache, te_cache, f_tr, f_te):
    v4_tr, v4_te = v4_base(tr_cache), v4_base(te_cache)
    y_tr = tr_cache["targets"].astype(np.float32)
    y_te = te_cache["targets"].astype(np.float32)
    th_tr, th_te = tr_cache["thermo_feat"], te_cache["thermo_feat"]

    print(f"\n=== {label} === use_uw={use_uw}")
    s1_models, s1_r2 = [], []
    for seed in range(N_SEEDS):
        m = train_stage1(seed, v4_tr, f_tr, th_tr, y_tr, device, use_uw)
        s1_models.append(m)
        s1_r2.append(r2_per_prop(predict(m, v4_te, f_te, th_te, device), y_te))
    s1 = summarize(f"{label}/Stage1", s1_r2)
    print(f"  Stage1 core7={s1['avg_r2_core7']:.4f}±{s1['std_r2_core7']:.4f}  lignin={s1['per_prop']['lignin_wt']:.4f}")
    for p in PROPS:
        print(f"    {p:12s}: {s1['per_prop'][p]:.4f}")

    s2_r2 = []
    for seed in range(N_SEEDS):
        m2 = train_stage2_hardfz(s1_models[seed], v4_tr, f_tr, th_tr, y_tr,
                                   device, seed + 100)
        s2_r2.append(r2_per_prop(predict(m2, v4_te, f_te, th_te, device), y_te))
    s2s = summarize(f"{label}/Stage2_hardfreeze", s2_r2)
    print(f"  Stage2_hardfreeze core7={s2s['avg_r2_core7']:.4f}±{s2s['std_r2_core7']:.4f}  lignin={s2s['per_prop']['lignin_wt']:.4f}")
    for p in PROPS:
        print(f"    {p:12s}: {s2s['per_prop'][p]:.4f}")
    return [s1, s2s]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tr = load_split("train"); te = load_split("test")
    pca = PCA(40).fit(tr["morgan_fp"])
    f_tr = pca.transform(tr["morgan_fp"]).astype(np.float32)
    f_te = pca.transform(te["morgan_fp"]).astype(np.float32)

    results = []
    results += run("Baseline", use_uw=False, device=device,
                     tr_cache=tr, te_cache=te, f_tr=f_tr, f_te=f_te)
    results += run("A4_uncertainty", use_uw=True, device=device,
                     tr_cache=tr, te_cache=te, f_tr=f_tr, f_te=f_te)

    print(f"\n{'='*68}\nA4 COMPARISON — uncertainty-weighted vs equal-weight loss\n{'='*68}")
    print(f"{'Variant':<45}{'core7':>9}{'std':>9}{'lignin':>10}")
    print("-" * 73)
    for r in results:
        print(f"{r['name']:<45}{r['avg_r2_core7']:>9.4f}{r['std_r2_core7']:>9.4f}{r['per_prop']['lignin_wt']:>10.4f}")

    out = V5 / "results" / "a4_uncertainty.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
