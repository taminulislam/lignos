"""A1 combined experiment: Shallow + Unbalanced loss + Masked physchem head on
the A1 cache (LignoIL + physchem backfill + P-column denoise).

Baseline floor (same architecture on plain LignoIL with no physchem and
polluted P): 0.8316. The A1 goal is to beat that — projected 0.845-0.855.

Runs two arms for apples-to-apples comparison:
  1. A1_no_physchem — Shallow+Unbal on A1 cache with PerPropHead (thermo-5 only)
     — isolates the effect of P-column denoise alone
  2. A1_masked_physchem — Shallow+Unbal on A1 cache with PerPropHeadMasked
     (thermo-5 + physchem-12 + has_physchem indicator)

Test set = v4 cached_test (7-prop, 39 rows) aligned by position to A1 test.
"""
from __future__ import annotations
import json, sys
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

CACHE = V5 / "data" / "LignoIL_A1"
PHYSCHEM_DIM = 12
N_SEEDS = 10


def load_split(split):
    d = np.load(CACHE / f"cached_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    return (0.4 * c["preds_fusion"] + 0.6 * c["preds_chemprop"]).astype(np.float32)


class PerPropHead(nn.Module):
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


class PerPropHeadMasked(nn.Module):
    def __init__(self, nf, n_props=8):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, nf), nn.Sigmoid())
        head_in = nf + 5 + PHYSCHEM_DIM + 1
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
              balance_props=False, epochs=300, patience=50):
    """Unbalanced loss by default (balance_props=False). Matches peak recipe."""
    set_seed(seed)
    n_props = tr_y.shape[1]
    m = model_cls(tr_f.shape[1], n_props).to(device)
    opt = AdamW(m.parameters(), lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    v_t = torch.from_numpy(tr_v).to(device)
    f_t = torch.from_numpy(tr_f).to(device)
    t_t = torch.from_numpy(tr_th).to(device)
    y_t = torch.from_numpy(tr_y).to(device)
    use_phys = tr_phys is not None
    if use_phys:
        p_t = torch.from_numpy(tr_phys).to(device)
        h_t = torch.from_numpy(tr_has).to(device)
    valid = ~torch.isnan(y_t); yf = torch.nan_to_num(y_t, 0.0)

    if use_phys:
        ds = TensorDataset(v_t.cpu(), f_t.cpu(), t_t.cpu(), p_t.cpu(), h_t.cpu(), yf.cpu(), valid.cpu())
    else:
        ds = TensorDataset(v_t.cpu(), f_t.cpu(), t_t.cpu(), yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            if use_phys:
                vb, ib, tb, pb, hb, yb, vm = batch
                pred = m(vb, ib, tb, pb, hb)
            else:
                vb, ib, tb, yb, vm = batch
                pred = m(vb, ib, tb)
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
            if use_phys:
                pred = m(v_t, f_t, t_t, p_t, h_t)
            else:
                pred = m(v_t, f_t, t_t)
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


def run_arm(name, model_cls, use_phys, tr, te, f_tr, f_te, device):
    v4_tr, v4_te = v4_base(tr), v4_base(te)
    y_tr = tr["targets"].astype(np.float32)
    y_te = te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]
    if use_phys:
        p_tr = tr["physchem_feat"].astype(np.float32)
        p_te = te["physchem_feat"].astype(np.float32)
        h_tr = tr["has_physchem"].astype(np.float32)
        h_te = te["has_physchem"].astype(np.float32)
    else:
        p_tr = p_te = h_tr = h_te = None

    print(f"\n=== {name} ===")
    r2s = []
    for seed in range(N_SEEDS):
        m = train_one(seed, model_cls, v4_tr, f_tr, th_tr, p_tr, h_tr, y_tr, device)
        pred = predict(m, v4_te, f_te, th_te, p_te, h_te, device)
        r2s.append(r2_per_prop(pred, y_te))
    s = summarize(name, r2s)
    print(f"  core7 = {s['avg_r2_core7']:.4f} ± {s['std_r2_core7']:.4f}")
    for p in PROPS:
        print(f"    {p:12s}: {s['per_prop'][p]:.4f}")
    return s


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Cache: {CACHE}")
    tr = load_split("train")
    te = load_split("test")
    pca = PCA(40).fit(tr["morgan_fp"])
    f_tr = pca.transform(tr["morgan_fp"]).astype(np.float32)
    f_te = pca.transform(te["morgan_fp"]).astype(np.float32)

    print(f"train={len(tr['smiles'])} rows, physchem covered={int(tr['has_physchem'].sum())}")
    print(f"test={len(te['smiles'])} rows, physchem covered={int(te['has_physchem'].sum())}")
    for i, p in enumerate(PROPS):
        if i < tr["targets"].shape[1]:
            print(f"  train {p:10s} rows: {int((~np.isnan(tr['targets'][:, i])).sum())}")

    results = []
    results.append(run_arm("A1_no_physchem", PerPropHead, False, tr, te, f_tr, f_te, device))
    results.append(run_arm("A1_masked_physchem", PerPropHeadMasked, True, tr, te, f_tr, f_te, device))

    print(f"\n{'='*70}\nA1 SUMMARY  (floor: 0.8316 = Shallow+Unbal on plain LignoIL)\n{'='*70}")
    print(f"{'Arm':<30}{'core7':>10}{'std':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<30}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}")

    out = V5 / "results" / "a1_combined.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
