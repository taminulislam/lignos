"""A2 (multi-feature gated residual with ChemProp) + X (thermo G_mix>=0 hinge)
on the A1 cache (P-denoised). 4 arms, 10 seeds each.

Arms:
  1. A1_repro        — Shallow+Unbalanced, Morgan only (baseline, should reproduce 0.8350)
  2. A2_chemprop     — + ChemProp(40D) concat with zero-init gated residual
  3. X_gmix_hinge    — A1 baseline + soft hinge loss: ReLU(-pred_G_mix)^2 * lambda
  4. A2_plus_X       — ChemProp gated residual + G_mix hinge stacked

ChemProp features cover only 152 train rows (the 19 original ILs). PCA is fit
on those non-zero rows; feature vector is zeroed for the 5037 thermo
pre-training rows (same format as the other 5 cached features). The zero-init
gate on the ChemProp branch guarantees A2_chemprop starts identical to A1.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, r2_per_prop, set_seed  # noqa

CACHE = V5 / "data" / "LignoIL_A1"
N_SEEDS = 10
PROP_G_MIX = 4  # index of G_mix in PROPS
LAMBDA_HINGE = 0.1


def load_split(s):
    d = np.load(CACHE / f"cached_{s}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    return (0.4 * c["preds_fusion"] + 0.6 * c["preds_chemprop"]).astype(np.float32)


def build_chemprop_40d(tr_c, te_c):
    """PCA(40) on the non-zero train rows only. Zero rows stay zero."""
    nz = (tr_c != 0).any(axis=1)
    if nz.sum() < 2:
        dim = min(40, tr_c.shape[1])
        return np.zeros((len(tr_c), dim), dtype=np.float32), np.zeros((len(te_c), dim), dtype=np.float32)
    pca = PCA(min(40, tr_c.shape[1])).fit(tr_c[nz])
    tr_p = pca.transform(tr_c).astype(np.float32)
    te_p = pca.transform(te_c).astype(np.float32)
    # Zero out rows that were zero in original
    tr_p[~nz] = 0.0
    te_nz = (te_c != 0).any(axis=1)
    te_p[~te_nz] = 0.0
    return tr_p, te_p


class A1Head(nn.Module):
    """Shallow head matching A1_no_physchem. Context = thermo[:, :5]."""
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

    def forward(self, v, i, t, chemprop=None):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        inp = torch.cat([g, tmp], -1)
        res = torch.cat([h(inp) for h in self.heads], -1)
        return v + torch.sigmoid(self.alphas) * res


class A2Head(nn.Module):
    """A1 + ChemProp gated residual delta (zero-init so starts == A1)."""
    def __init__(self, nf, n_props=8, chemprop_dim=40):
        super().__init__()
        # Morgan path (same as A1Head)
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

        # ChemProp delta branch — zero-init projection & gate
        self.cp_proj = nn.Sequential(
            nn.Linear(chemprop_dim, 32), nn.GELU(), nn.Linear(32, 32)
        )
        with torch.no_grad():
            self.cp_proj[-1].weight.zero_()
            self.cp_proj[-1].bias.zero_()
        self.cp_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(32 + 5, 16), nn.GELU(), nn.Linear(16, 1))
            for _ in range(n_props)
        ])
        for h in self.cp_heads:
            with torch.no_grad():
                h[-1].weight.zero_(); h[-1].bias.zero_()
        # A separate learned per-prop gate on the chemprop branch
        self.cp_gate = nn.Parameter(torch.full((n_props,), -5.0))  # sigmoid ~ 0.0067

    def forward(self, v, i, t, chemprop):
        tmp = t[:, :5]
        # Morgan path
        g = i * self.gate(tmp)
        inp = torch.cat([g, tmp], -1)
        res = torch.cat([h(inp) for h in self.heads], -1)
        out = v + torch.sigmoid(self.alphas) * res
        # ChemProp delta (zero at init)
        cp_h = self.cp_proj(chemprop)
        cp_inp = torch.cat([cp_h, tmp], -1)
        cp_delta = torch.cat([h(cp_inp) for h in self.cp_heads], -1)
        return out + torch.sigmoid(self.cp_gate) * cp_delta


def train_one(seed, model_cls, tr_v, tr_f, tr_th, tr_cp, tr_y, device,
              use_hinge=False, lambda_hinge=LAMBDA_HINGE,
              epochs=300, patience=50):
    set_seed(seed)
    n_props = tr_y.shape[1]
    if model_cls is A2Head:
        m = model_cls(tr_f.shape[1], n_props, chemprop_dim=tr_cp.shape[1]).to(device)
    else:
        m = model_cls(tr_f.shape[1], n_props).to(device)
    opt = AdamW(m.parameters(), lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    v_t = torch.from_numpy(tr_v).to(device)
    f_t = torch.from_numpy(tr_f).to(device)
    t_t = torch.from_numpy(tr_th).to(device)
    y_t = torch.from_numpy(tr_y).to(device)
    use_cp = tr_cp is not None and model_cls is A2Head
    if use_cp:
        cp_t = torch.from_numpy(tr_cp).to(device)
    valid = ~torch.isnan(y_t); yf = torch.nan_to_num(y_t, 0.0)

    if use_cp:
        ds = TensorDataset(v_t.cpu(), f_t.cpu(), t_t.cpu(), cp_t.cpu(), yf.cpu(), valid.cpu())
    else:
        ds = TensorDataset(v_t.cpu(), f_t.cpu(), t_t.cpu(), yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            if use_cp:
                vb, ib, tb, cpb, yb, vm = batch
                pred = m(vb, ib, tb, cpb)
            else:
                vb, ib, tb, yb, vm = batch
                pred = m(vb, ib, tb)
            err2 = ((pred - yb) ** 2) * vm.float()
            mse_loss = err2.sum() / vm.float().sum().clamp(min=1)
            loss = mse_loss
            if use_hinge:
                # Hinge: penalize negative G_mix predictions (only where valid)
                # G_mix is already in z-score space so the "physical" threshold
                # is 0 in un-z-scored space. Approximate: penalize z < -0.5.
                g_mix_pred = pred[:, PROP_G_MIX]
                hinge = F.relu(-0.5 - g_mix_pred).pow(2).mean()
                loss = loss + lambda_hinge * hinge
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            if use_cp:
                pred = m(v_t, f_t, t_t, cp_t)
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


def predict(m, v, f, t, cp, device, use_cp):
    xs = [torch.from_numpy(v).to(device), torch.from_numpy(f).to(device),
          torch.from_numpy(t).to(device)]
    if use_cp:
        xs.append(torch.from_numpy(cp).to(device))
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


def run_arm(name, model_cls, use_hinge, use_cp, tr, te, f_tr, f_te, cp_tr, cp_te, device):
    v4_tr, v4_te = v4_base(tr), v4_base(te)
    y_tr = tr["targets"].astype(np.float32)
    y_te = te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]
    print(f"\n=== {name}  (model={model_cls.__name__}, hinge={use_hinge}, cp={use_cp}) ===")
    r2s = []
    for seed in range(N_SEEDS):
        m = train_one(seed, model_cls, v4_tr, f_tr, th_tr,
                      cp_tr if use_cp else None, y_tr, device, use_hinge=use_hinge)
        pred = predict(m, v4_te, f_te, th_te, cp_te if use_cp else None, device, use_cp=use_cp)
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
    tr, te = load_split("train"), load_split("test")
    pca = PCA(40).fit(tr["morgan_fp"])
    f_tr = pca.transform(tr["morgan_fp"]).astype(np.float32)
    f_te = pca.transform(te["morgan_fp"]).astype(np.float32)
    cp_tr, cp_te = build_chemprop_40d(tr["chemprop_fp"], te["chemprop_fp"])
    print(f"train={len(tr['smiles'])}, test={len(te['smiles'])}")
    print(f"chemprop nonzero train rows: {((cp_tr != 0).any(axis=1)).sum()}/{len(cp_tr)}")
    print(f"chemprop nonzero test rows:  {((cp_te != 0).any(axis=1)).sum()}/{len(cp_te)}")

    results = []
    results.append(run_arm("A1_repro", A1Head, False, False, tr, te, f_tr, f_te, cp_tr, cp_te, device))
    results.append(run_arm("A2_chemprop", A2Head, False, True, tr, te, f_tr, f_te, cp_tr, cp_te, device))
    results.append(run_arm("X_gmix_hinge", A1Head, True, False, tr, te, f_tr, f_te, cp_tr, cp_te, device))
    results.append(run_arm("A2_plus_X", A2Head, True, True, tr, te, f_tr, f_te, cp_tr, cp_te, device))

    print(f"\n{'='*70}\nA2+X SUMMARY  (A1 floor = 0.8350)\n{'='*70}")
    print(f"{'Arm':<20}{'core7':>10}{'std':>10}{'gamma2':>10}{'G_E':>10}{'G_mix':>10}{'H_vap':>10}")
    print("-" * 80)
    for r in results:
        pp = r["per_prop"]
        print(f"{r['name']:<20}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}"
              f"{pp['gamma2']:>10.4f}{pp['G_E']:>10.4f}{pp['G_mix']:>10.4f}{pp['H_vap']:>10.4f}")

    out = V5 / "results" / "a2_plus_x.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
