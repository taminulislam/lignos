"""Track 1b: train on rows where has_physchem=True (matched coverage).

Restricts all splits (train/val/test) to the has_physchem=True subset:
    train: 262 rows across 19 original ILs
    val:    32 rows across  4 original ILs
    test:   39 rows across  5 original ILs
This eliminates the train/test distribution shift that wrecked the earlier
Extended experiment (train was 5% physchem-covered, test was 100%).

Compares Morgan-only vs Morgan+physchem on the same restricted data so the
physchem effect is isolated from cache size. Output variant with the better
lignin R² can ensemble with the full-data baseline hardfreeze for final
reporting (test rows are 100% physchem-covered, so the restricted model
can directly serve every test prediction).
"""
from __future__ import annotations
import argparse, copy, json, sys
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

EXT = V5 / "data" / "LignoIL_unified"
N_SEEDS = 10
PHYSCHEM_DIM = 12


def load_restricted(split):
    d = np.load(EXT / f"cached_{split}.npz", allow_pickle=True)
    has = d["has_physchem"].astype(bool)
    out = {}
    for k in d.files:
        arr = d[k]
        out[k] = arr[has] if len(arr) == len(has) else arr
    return out


def v4_base(c):
    return (0.4 * c["preds_fusion"] + 0.6 * c["preds_chemprop"]).astype(np.float32)


def preprocess_phys_train(x, has):
    x = x.astype(np.float32).copy()
    x[:, 3] = np.log1p(np.maximum(x[:, 3], 0.0))  # viscosity
    x[:, 5] = np.log1p(np.maximum(x[:, 5], 0.0))  # conductivity
    mu = x.mean(0); sd = x.std(0) + 1e-6
    return ((x - mu) / sd).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def preprocess_phys_apply(x, mu, sd):
    x = x.astype(np.float32).copy()
    x[:, 3] = np.log1p(np.maximum(x[:, 3], 0.0))
    x[:, 5] = np.log1p(np.maximum(x[:, 5], 0.0))
    return ((x - mu) / sd).astype(np.float32)


class HeadV(nn.Module):
    def __init__(self, nf, n_props=8, use_phys=False):
        super().__init__()
        self.use_phys = use_phys
        self.gate = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, nf), nn.Sigmoid())
        ctx_dim = 5 + (PHYSCHEM_DIM if use_phys else 0)
        head_in = nf + ctx_dim
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 32), nn.GELU(), nn.Linear(32, 1)) for _ in range(n_props)
        ])
        self.alphas = nn.Parameter(torch.full((n_props,), -3.0))
        for h in self.heads:
            with torch.no_grad():
                h[-1].weight.mul_(0.01); h[-1].bias.zero_()

    def forward(self, v, i, t, p=None):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        ctx = tmp if not self.use_phys else torch.cat([tmp, p], -1)
        res = torch.cat([h(torch.cat([g, ctx], -1)) for h in self.heads], -1)
        return v + torch.sigmoid(self.alphas) * res


def train_stage1(seed, tr_v, tr_f, tr_th, tr_p, tr_y, device, use_phys,
                  epochs=400, patience=60):
    set_seed(seed)
    n = tr_y.shape[1]
    m = HeadV(tr_f.shape[1], n, use_phys).to(device)
    opt = AdamW(m.parameters(), lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    tt = {k: torch.from_numpy(v).to(device) for k, v in
           dict(v=tr_v, f=tr_f, t=tr_th, y=tr_y).items()}
    pt = torch.from_numpy(tr_p).to(device) if tr_p is not None else None
    valid = ~torch.isnan(tt["y"]); yf = torch.nan_to_num(tt["y"], 0.0)
    ds = TensorDataset(tt["v"].cpu(), tt["f"].cpu(), tt["t"].cpu(),
                        (pt.cpu() if pt is not None else torch.zeros(len(tt["v"]), 1)),
                        yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for v, im, t, p, y, vm in loader:
            v, im, t, y, vm = [x.to(device) for x in (v, im, t, y, vm)]
            p = p.to(device) if pt is not None else None
            pred = m(v, im, t, p)
            e2 = ((pred - y) ** 2) * vm
            loss = (e2.sum(0) / vm.sum(0).clamp(min=1)).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(tt["v"], tt["f"], tt["t"], pt)
            e2 = ((pred - yf) ** 2) * valid
            tl = (e2.sum(0) / valid.sum(0).clamp(min=1)).mean().item()
        if np.isfinite(tl) and tl < best:
            best, state, bad = tl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval(); return m


def train_stage2_hardfz(s1, tr_v, tr_f, tr_th, tr_p, tr_y, device, seed, use_phys,
                          epochs=400, patience=60):
    set_seed(seed)
    m = copy.deepcopy(s1).to(device)
    for p in m.parameters(): p.requires_grad = False
    nf = m.gate[2].out_features
    ctx = 5 + (PHYSCHEM_DIM if use_phys else 0)
    m.heads[7] = nn.Sequential(
        nn.Linear(nf + ctx, 128), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(64, 1),
    ).to(device)
    with torch.no_grad():
        m.heads[7][-1].weight.mul_(0.01); m.heads[7][-1].bias.zero_()
    for p in m.heads[7].parameters(): p.requires_grad = True
    m.alphas.requires_grad = True
    mask = torch.zeros_like(m.alphas); mask[7] = 1.0
    m.alphas.register_hook(lambda g: g * mask)
    h7 = list(m.heads[7].parameters())
    opt = AdamW([{"params": h7, "weight_decay": 1e-2},
                  {"params": [m.alphas], "weight_decay": 0.0}], lr=1e-3)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    opts = h7 + [m.alphas]
    tt = {k: torch.from_numpy(v).to(device) for k, v in
           dict(v=tr_v, f=tr_f, t=tr_th, y=tr_y).items()}
    pt = torch.from_numpy(tr_p).to(device) if tr_p is not None else None
    ds = TensorDataset(tt["v"].cpu(), tt["f"].cpu(), tt["t"].cpu(),
                        (pt.cpu() if pt is not None else torch.zeros(len(tt["v"]), 1)),
                        tt["y"].cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for v, im, t, p, y in loader:
            v, im, t, y = [x.to(device) for x in (v, im, t, y)]
            p = p.to(device) if pt is not None else None
            pred = m(v, im, t, p)
            lg = ~torch.isnan(y[:, 7])
            if lg.sum() == 0: continue
            loss = ((pred[lg, 7] - y[lg, 7].nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(opts, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(tt["v"], tt["f"], tt["t"], pt)
            lg = ~torch.isnan(tt["y"][:, 7])
            tl = ((pred[lg, 7] - tt["y"][lg, 7].nan_to_num(0)) ** 2).mean().item() if lg.any() else float("inf")
        if np.isfinite(tl) and tl < best:
            best, state, bad = tl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval(); return m


def predict(m, v, f, t, p, device):
    tt = [torch.from_numpy(x).to(device) for x in (v, f, t)]
    pt = torch.from_numpy(p).to(device) if p is not None else None
    m.eval()
    with torch.no_grad():
        return m(*tt, pt).cpu().numpy()


def summarize(name, r2s):
    c = [r["avg_core7"] for r in r2s]
    out = {"name": name, "avg_r2_core7": float(np.mean(c)),
           "std_r2_core7": float(np.std(c)), "per_prop": {}}
    for p in PROPS:
        vs = [r.get(p) for r in r2s if r.get(p) is not None and not np.isnan(r.get(p, float("nan")))]
        out["per_prop"][p] = float(np.mean(vs)) if vs else float("nan")
    return out


def run(label, use_phys, device):
    tr = load_restricted("train"); te = load_restricted("test")
    v4_tr, v4_te = v4_base(tr), v4_base(te)
    y_tr = tr["targets"].astype(np.float32); y_te = te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]
    print(f"\n--- {label}: train={len(y_tr)}  test={len(y_te)} ---")
    pca = PCA(40).fit(tr["morgan_fp"])
    f_tr = pca.transform(tr["morgan_fp"]).astype(np.float32)
    f_te = pca.transform(te["morgan_fp"]).astype(np.float32)
    p_tr = p_te = None
    if use_phys:
        p_tr, mu, sd = preprocess_phys_train(tr["physchem_feat"], tr["has_physchem"])
        p_te = preprocess_phys_apply(te["physchem_feat"], mu, sd)
    s1_models, s1_r2, s2_r2 = [], [], []
    for seed in range(N_SEEDS):
        m1 = train_stage1(seed, v4_tr, f_tr, th_tr, p_tr, y_tr, device, use_phys)
        s1_models.append(m1)
        s1_r2.append(r2_per_prop(predict(m1, v4_te, f_te, th_te, p_te, device), y_te))
    for seed in range(N_SEEDS):
        m2 = train_stage2_hardfz(s1_models[seed], v4_tr, f_tr, th_tr, p_tr, y_tr,
                                   device, seed + 100, use_phys)
        s2_r2.append(r2_per_prop(predict(m2, v4_te, f_te, th_te, p_te, device), y_te))
    s1s = summarize(f"{label}/Stage1", s1_r2)
    s2s = summarize(f"{label}/Stage2_hardfreeze", s2_r2)
    for s in (s1s, s2s):
        print(f"{s['name']}: core7={s['avg_r2_core7']:.4f}±{s['std_r2_core7']:.4f}  lignin={s['per_prop']['lignin_wt']:.4f}")
    return [s1s, s2s]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    results = []
    results += run("Restricted_Morgan", use_phys=False, device=device)
    results += run("Restricted_Morgan+Physchem", use_phys=True, device=device)

    print(f"\n{'='*68}\nTrack 1b comparison\n{'='*68}")
    print(f"{'Variant':<45}{'core7':>9}{'std':>9}{'lignin':>10}")
    print("-" * 73)
    for r in results:
        print(f"{r['name']:<45}{r['avg_r2_core7']:>9.4f}{r['std_r2_core7']:>9.4f}{r['per_prop']['lignin_wt']:>10.4f}")
    out = V5 / "results" / "track1b_restricted.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
