"""Extended training on LignoIL_unified cache with masked physchem head.

Post-fix experiment (combines Priority #2 + #3):
  - Cache: data/LignoIL_unified/cached_*.npz (backfilled physchem + fixed
    thermo_feat for 110 Baran rows)
  - Arms:
      A) Baseline_unified        — no physchem, 5-D thermo ctx (same arch as
                                     today's 0.8272 baseline, run on unified cache
                                     to isolate the cache effect)
      B) Extended_unmasked       — current PerPropHeadExt (12-D physchem
                                     concat, no has_physchem signal to the head)
      C) Extended_masked         — 13-D physchem (appends `has_physchem`
                                     indicator as a 13th feature so the head
                                     can learn "missing" vs "zero-valued")

10 seeds each. All use the same shallow head, balanced-per-prop loss (matching
today's a4_uncertainty Baseline/Stage1 recipe). A 4th arm with unbalanced loss
is added IF the caller passes --add_unbalanced (for after #1's result).
"""
from __future__ import annotations
import argparse, json, sys
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
from audit_residuals import PROPS, r2_per_prop, set_seed  # noqa

CACHE = V5 / "data" / "LignoIL_unified"
PHYSCHEM_DIM = 12
N_SEEDS = 10


def load_split(split):
    d = np.load(CACHE / f"cached_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    return (0.4 * c["preds_fusion"] + 0.6 * c["preds_chemprop"]).astype(np.float32)


def preprocess_physchem(tr_phys, tr_has, te_phys, te_has):
    """Log-transform viscosity/conductivity, z-score on covered train rows, zero
    out rows without physchem coverage. Returns (tr_processed, te_processed)."""
    def apply(x, has, mu=None, sd=None):
        x = x.astype(np.float32).copy()
        x[:, 3] = np.log1p(np.maximum(x[:, 3], 0.0))
        x[:, 5] = np.log1p(np.maximum(x[:, 5], 0.0))
        covered = has.astype(bool)
        if mu is None:
            mu = x[covered].mean(axis=0) if covered.sum() else np.zeros(x.shape[1])
            sd = (x[covered].std(axis=0) + 1e-6) if covered.sum() else np.ones(x.shape[1])
        z = (x - mu) / sd
        z = z * covered[:, None]
        return z.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)

    tr_z, mu, sd = apply(tr_phys, tr_has)
    te_z, _, _ = apply(te_phys, te_has, mu, sd)
    return tr_z, te_z


class PerPropHead(nn.Module):
    """Shallow, 5-D thermo context (matches today's baseline)."""
    def __init__(self, nf, n_props=8):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, nf), nn.Sigmoid())
        head_in = nf + 5
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in range(n_props)
        ])
        self.alphas = nn.Parameter(torch.full((n_props,), -3.0))
        for h in self.heads:
            with torch.no_grad():
                h[-1].weight.mul_(0.01); h[-1].bias.zero_()

    def forward(self, v, i, t, phys=None, has_phys=None):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        inp = torch.cat([g, tmp], -1)
        res = torch.cat([h(inp) for h in self.heads], -1)
        return v + torch.sigmoid(self.alphas) * res


class PerPropHeadExt(nn.Module):
    """Extended: ctx = [thermo5, physchem12]. No mask signal."""
    def __init__(self, nf, n_props=8):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, nf), nn.Sigmoid())
        head_in = nf + 5 + PHYSCHEM_DIM
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in range(n_props)
        ])
        self.alphas = nn.Parameter(torch.full((n_props,), -3.0))
        for h in self.heads:
            with torch.no_grad():
                h[-1].weight.mul_(0.01); h[-1].bias.zero_()

    def forward(self, v, i, t, phys, has_phys=None):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        ctx = torch.cat([tmp, phys], -1)
        inp = torch.cat([g, ctx], -1)
        res = torch.cat([h(inp) for h in self.heads], -1)
        return v + torch.sigmoid(self.alphas) * res


class PerPropHeadMasked(nn.Module):
    """Masked: ctx = [thermo5, physchem12, has_physchem_indicator]. The 13th
    feature lets each head distinguish real-zero physchem from missing
    physchem and route gradients accordingly."""
    def __init__(self, nf, n_props=8):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, nf), nn.Sigmoid())
        head_in = nf + 5 + PHYSCHEM_DIM + 1  # +1 for has_physchem
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in range(n_props)
        ])
        self.alphas = nn.Parameter(torch.full((n_props,), -3.0))
        for h in self.heads:
            with torch.no_grad():
                h[-1].weight.mul_(0.01); h[-1].bias.zero_()

    def forward(self, v, i, t, phys, has_phys):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        hp = has_phys.float().unsqueeze(-1) if has_phys.ndim == 1 else has_phys.float()
        ctx = torch.cat([tmp, phys, hp], -1)
        inp = torch.cat([g, ctx], -1)
        res = torch.cat([h(inp) for h in self.heads], -1)
        return v + torch.sigmoid(self.alphas) * res


def train_one(seed, model_cls, tr_v, tr_f, tr_th, tr_phys, tr_has, tr_y, device,
              balance_props=True, epochs=300, patience=50):
    set_seed(seed)
    n_props = tr_y.shape[1]
    m = model_cls(tr_f.shape[1], n_props).to(device)
    opt = AdamW(m.parameters(), lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    v_t = torch.from_numpy(tr_v).to(device)
    f_t = torch.from_numpy(tr_f).to(device)
    t_t = torch.from_numpy(tr_th).to(device)
    y_t = torch.from_numpy(tr_y).to(device)
    phys_t = torch.from_numpy(tr_phys).to(device) if tr_phys is not None else None
    has_t = torch.from_numpy(tr_has).to(device) if tr_has is not None else None
    valid = ~torch.isnan(y_t); yf = torch.nan_to_num(y_t, 0.0)

    ds_tensors = [v_t.cpu(), f_t.cpu(), t_t.cpu(), yf.cpu(), valid.cpu()]
    if phys_t is not None:
        ds_tensors.insert(3, phys_t.cpu())
        ds_tensors.insert(4, has_t.cpu())
    ds = TensorDataset(*ds_tensors)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            if phys_t is None:
                vb, ib, tb, yb, vm = batch
                pred = m(vb, ib, tb)
            else:
                vb, ib, tb, pb, hb, yb, vm = batch
                pred = m(vb, ib, tb, pb, hb)
            err2 = ((pred - yb) ** 2) * vm.float()
            if balance_props:
                per_prop_mse = err2.sum(0) / vm.float().sum(0).clamp(min=1)
                loss = per_prop_mse.mean()
            else:
                loss = err2.sum() / vm.float().sum().clamp(min=1)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            if phys_t is None:
                pred = m(v_t, f_t, t_t)
            else:
                pred = m(v_t, f_t, t_t, phys_t, has_t)
            err2 = ((pred - yf) ** 2) * valid.float()
            tl = (err2.sum(0) / valid.float().sum(0).clamp(min=1)).mean().item()
        if np.isfinite(tl) and tl < best:
            best = tl; state = {k: v.clone() for k, v in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval()
    return m


def predict(m, v, f, t, phys, has_phys, device):
    xs = [torch.from_numpy(v).to(device), torch.from_numpy(f).to(device),
          torch.from_numpy(t).to(device)]
    if phys is not None:
        xs.extend([torch.from_numpy(phys).to(device), torch.from_numpy(has_phys).to(device)])
    m.eval()
    with torch.no_grad():
        return m(*xs).cpu().numpy()


def summarize(name, r2s):
    c = [r["avg_core7"] for r in r2s]
    out = {"name": name, "avg_r2_core7": float(np.mean(c)),
           "std_r2_core7": float(np.std(c)), "per_prop": {}}
    for p in PROPS:
        vs = [r.get(p) for r in r2s if r.get(p) is not None and not np.isnan(r.get(p, float("nan")))]
        out["per_prop"][p] = float(np.mean(vs)) if vs else float("nan")
    return out


def run_arm(name, model_cls, include_phys, balance, tr, te, f_tr, f_te, device):
    v4_tr, v4_te = v4_base(tr), v4_base(te)
    y_tr = tr["targets"].astype(np.float32)
    y_te = te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]
    if include_phys:
        p_tr, p_te = preprocess_physchem(
            tr["physchem_feat"], tr["has_physchem"],
            te["physchem_feat"], te["has_physchem"])
        h_tr, h_te = tr["has_physchem"].astype(np.float32), te["has_physchem"].astype(np.float32)
    else:
        p_tr = p_te = h_tr = h_te = None

    print(f"\n=== {name}  (balance={balance}) ===")
    r2s = []
    for seed in range(N_SEEDS):
        m = train_one(seed, model_cls, v4_tr, f_tr, th_tr, p_tr, h_tr, y_tr,
                      device, balance_props=balance)
        r2s.append(r2_per_prop(predict(m, v4_te, f_te, th_te, p_te, h_te, device), y_te))
    s = summarize(name, r2s)
    print(f"  core7 = {s['avg_r2_core7']:.4f} ± {s['std_r2_core7']:.4f}")
    for p in PROPS:
        print(f"    {p:12s}: {s['per_prop'][p]:.4f}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add_unbalanced", action="store_true",
                    help="Also run an unbalanced-loss variant (after #1's result)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Cache: {CACHE}")
    tr, te = load_split("train"), load_split("test")

    # Morgan PCA-40D on the shared train set
    pca = PCA(40).fit(tr["morgan_fp"])
    f_tr = pca.transform(tr["morgan_fp"]).astype(np.float32)
    f_te = pca.transform(te["morgan_fp"]).astype(np.float32)
    print(f"train={len(tr['smiles'])} rows, test={len(te['smiles'])} rows, physchem coverage: "
          f"train={int(tr['has_physchem'].sum())}/{len(tr['has_physchem'])}")

    results = []
    results.append(run_arm("A_baseline_unified",   PerPropHead,       False, True, tr, te, f_tr, f_te, device))
    results.append(run_arm("B_extended_unmasked",  PerPropHeadExt,    True,  True, tr, te, f_tr, f_te, device))
    results.append(run_arm("C_extended_masked",    PerPropHeadMasked, True,  True, tr, te, f_tr, f_te, device))
    if args.add_unbalanced:
        results.append(run_arm("D_extended_masked_unbal", PerPropHeadMasked, True, False, tr, te, f_tr, f_te, device))

    print(f"\n{'='*70}\nSUMMARY  (target: beat today's 0.8272 baseline)\n{'='*70}")
    print(f"{'Arm':<30}{'core7':>10}{'std':>10}{'lignin':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<30}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}"
              f"{r['per_prop'].get('lignin_wt', float('nan')):>10.4f}")

    out = V5 / "results" / "extended_masked_comparison.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
