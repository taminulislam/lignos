"""A2_chemprop two-stage: Stage-1 core7 with Morgan+ChemProp gated residual
(A2Head) + Stage-2 hardfreeze with physchem-augmented deep lignin head.

Stage-1 recipe (A2_chemprop, replicated from train_a2_plus_x.py):
  - A2Head = A1Head + zero-init ChemProp(40D) gated residual branch
  - Shallow core head, 5-D thermo context
  - Morgan(40D) PCA features
  - Unbalanced loss, 300 epochs, early-stop patience 50
  - Expected: core7 ≈ 0.840, lignin ≈ 0.47 (single-stage, per a2_plus_x run)

Stage-2 recipe (new deep lignin head with hardfreeze):
  - Freeze entire A2Head backbone (Morgan + ChemProp + core heads + alphas + cp_gate)
  - REPLACE heads[7] path with a deep 3-layer head on
      ctx = [gated_Morgan(nf), thermo(5), physchem(12), has_physchem(1)] = nf+18 dim
  - Train only deep_lignin params + alpha_lignin
  - Loss: lignin-only MSE, 300 epochs

Expected:
  - core7 unchanged from Stage-1 (hardfreeze prevents drift), should reproduce ≈0.840
  - lignin climbs to 0.70–0.78 (matching the +0.21 gain pattern seen in A1 two-stage)
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
from audit_residuals import PROPS, r2_per_prop, set_seed  # noqa

CACHE = V5 / "data" / "LignoIL_A1"
N_SEEDS = 10


def load_split(s):
    d = np.load(CACHE / f"cached_{s}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    return (0.4 * c["preds_fusion"] + 0.6 * c["preds_chemprop"]).astype(np.float32)


def build_chemprop_40d(tr_c, te_c):
    """PCA(40) on non-zero train rows. Zero rows stay zero. Matches train_a2_plus_x."""
    nz = (tr_c != 0).any(axis=1)
    if nz.sum() < 2:
        dim = min(40, tr_c.shape[1])
        return (np.zeros((len(tr_c), dim), dtype=np.float32),
                np.zeros((len(te_c), dim), dtype=np.float32))
    pca = PCA(min(40, tr_c.shape[1])).fit(tr_c[nz])
    tr_p = pca.transform(tr_c).astype(np.float32)
    te_p = pca.transform(te_c).astype(np.float32)
    tr_p[~nz] = 0.0
    te_nz = (te_c != 0).any(axis=1)
    te_p[~te_nz] = 0.0
    return tr_p, te_p


class A2Head(nn.Module):
    """A1 + ChemProp gated residual delta (zero-init so starts == A1). Identical
    to train_a2_plus_x.A2Head."""
    def __init__(self, nf, n_props=8, chemprop_dim=40):
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
        self.cp_gate = nn.Parameter(torch.full((n_props,), -5.0))

    def forward(self, v, i, t, chemprop):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        inp = torch.cat([g, tmp], -1)
        res = torch.cat([h(inp) for h in self.heads], -1)
        out = v + torch.sigmoid(self.alphas) * res
        cp_h = self.cp_proj(chemprop)
        cp_inp = torch.cat([cp_h, tmp], -1)
        cp_delta = torch.cat([h(cp_inp) for h in self.cp_heads], -1)
        return out + torch.sigmoid(self.cp_gate) * cp_delta


class A2StageTwoLigninWrapper(nn.Module):
    """Wraps a trained A2Head. All params frozen; lignin column (index 7) is
    overridden by a deep 3-layer head fed with
        [gated_Morgan, thermo(5), physchem(12), has_physchem(1)]
    Only the deep_lignin weights + alpha_lignin are trainable.

    Note on lignin source: we build the Stage-2 residual on TOP of v[:, 7]
    (the v4_base prediction for lignin), NOT on top of the Stage-1 A2 lignin
    output. This mirrors train_a1_two_stage.py so the two-stage results are
    directly comparable.
    """
    def __init__(self, stage1_model: A2Head, physchem_dim=12):
        super().__init__()
        self.backbone = stage1_model
        for p in self.backbone.parameters():
            p.requires_grad = False

        nf = self.backbone.gate[2].out_features
        head_in = nf + 5 + physchem_dim + 1
        self.deep_lignin = nn.Sequential(
            nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        with torch.no_grad():
            self.deep_lignin[-1].weight.mul_(0.01); self.deep_lignin[-1].bias.zero_()

        self.alpha_lignin = nn.Parameter(self.backbone.alphas.data[7].clone())

    def forward(self, v, i, t, chemprop, phys, has_phys):
        # Frozen A2 forward for core-7 (columns 0..6)
        base = self.backbone(v, i, t, chemprop)
        # Deep lignin residual
        tmp = t[:, :5]
        g = i * self.backbone.gate(tmp)
        hp = has_phys.float().unsqueeze(-1) if has_phys.ndim == 1 else has_phys.float()
        ctx = torch.cat([g, tmp, phys, hp], -1)
        res_lignin = self.deep_lignin(ctx).squeeze(-1)
        out = base.clone()
        out[:, 7] = v[:, 7] + torch.sigmoid(self.alpha_lignin) * res_lignin
        return out


def train_stage1_a2(seed, v4_tr, f_tr, th_tr, cp_tr, y_tr, device,
                     epochs=300, patience=50):
    """Matches A2_chemprop arm from train_a2_plus_x.py."""
    set_seed(seed)
    n_props = y_tr.shape[1]
    m = A2Head(f_tr.shape[1], n_props, chemprop_dim=cp_tr.shape[1]).to(device)
    opt = AdamW(m.parameters(), lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    v_t = torch.from_numpy(v4_tr).to(device)
    f_t = torch.from_numpy(f_tr).to(device)
    t_t = torch.from_numpy(th_tr).to(device)
    cp_t = torch.from_numpy(cp_tr).to(device)
    y_t = torch.from_numpy(y_tr).to(device)
    valid = ~torch.isnan(y_t); yf = torch.nan_to_num(y_t, 0.0)

    ds = TensorDataset(v_t.cpu(), f_t.cpu(), t_t.cpu(), cp_t.cpu(), yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for vb, ib, tb, cpb, yb, vm in loader:
            vb, ib, tb, cpb, yb, vm = [x.to(device) for x in (vb, ib, tb, cpb, yb, vm)]
            pred = m(vb, ib, tb, cpb)
            err2 = ((pred - yb) ** 2) * vm.float()
            loss = err2.sum() / vm.float().sum().clamp(min=1)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(v_t, f_t, t_t, cp_t)
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


def train_stage2_lignin(stage1_model, v4_tr, f_tr, th_tr, cp_tr, phys_tr, has_tr, y_tr,
                         device, seed, epochs=300, patience=50):
    """Hardfreeze A2Head; train only the deep lignin head + alpha_lignin."""
    set_seed(seed)
    m = A2StageTwoLigninWrapper(copy.deepcopy(stage1_model)).to(device)
    train_params = [p for p in m.parameters() if p.requires_grad]
    opt = AdamW([
        {"params": m.deep_lignin.parameters(), "weight_decay": 1e-2},
        {"params": [m.alpha_lignin], "weight_decay": 0.0},
    ], lr=1e-3)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    v_t = torch.from_numpy(v4_tr).to(device)
    f_t = torch.from_numpy(f_tr).to(device)
    t_t = torch.from_numpy(th_tr).to(device)
    cp_t = torch.from_numpy(cp_tr).to(device)
    p_t = torch.from_numpy(phys_tr).to(device)
    h_t = torch.from_numpy(has_tr).to(device)
    y_t = torch.from_numpy(y_tr).to(device)

    ds = TensorDataset(v_t.cpu(), f_t.cpu(), t_t.cpu(), cp_t.cpu(),
                        p_t.cpu(), h_t.cpu(), y_t.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for vb, ib, tb, cpb, pb, hb, yb in loader:
            vb, ib, tb, cpb, pb, hb, yb = [x.to(device) for x in
                                            (vb, ib, tb, cpb, pb, hb, yb)]
            pred = m(vb, ib, tb, cpb, pb, hb)
            lg = ~torch.isnan(yb[:, 7])
            if lg.sum() == 0: continue
            loss = ((pred[lg, 7] - yb[lg, 7].nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(v_t, f_t, t_t, cp_t, p_t, h_t)
            lg = ~torch.isnan(y_t[:, 7])
            tl = ((pred[lg, 7] - y_t[lg, 7].nan_to_num(0)) ** 2).mean().item() if lg.any() else float("inf")
        if np.isfinite(tl) and tl < best:
            best = tl; state = {k: v.clone() for k, v in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval()
    return m


def preprocess_physchem(tr_phys, tr_has, te_phys, te_has):
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


def predict_stage1(m, v, f, t, cp, device):
    with torch.no_grad():
        return m(torch.from_numpy(v).to(device),
                 torch.from_numpy(f).to(device),
                 torch.from_numpy(t).to(device),
                 torch.from_numpy(cp).to(device)).cpu().numpy()


def predict_stage2(m, v, f, t, cp, phys, has_phys, device):
    with torch.no_grad():
        return m(torch.from_numpy(v).to(device),
                 torch.from_numpy(f).to(device),
                 torch.from_numpy(t).to(device),
                 torch.from_numpy(cp).to(device),
                 torch.from_numpy(phys).to(device),
                 torch.from_numpy(has_phys).to(device)).cpu().numpy()


def summarize(name, r2s):
    c = [r["avg_core7"] for r in r2s]
    out = {"name": name, "avg_r2_core7": float(np.mean(c)),
           "std_r2_core7": float(np.std(c)), "per_prop": {}}
    for p in PROPS:
        vs = [r.get(p) for r in r2s if r.get(p) is not None and not np.isnan(r.get(p, float("nan")))]
        out["per_prop"][p] = float(np.mean(vs)) if vs else float("nan")
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Cache: {CACHE}")
    tr, te = load_split("train"), load_split("test")

    # Morgan PCA(40D) — matches A1/A2 recipes
    pca = PCA(40).fit(tr["morgan_fp"])
    f_tr = pca.transform(tr["morgan_fp"]).astype(np.float32)
    f_te = pca.transform(te["morgan_fp"]).astype(np.float32)

    # ChemProp PCA(40D) — non-zero rows only
    cp_tr, cp_te = build_chemprop_40d(tr["chemprop_fp"], te["chemprop_fp"])
    nz_tr = int((tr["chemprop_fp"] != 0).any(axis=1).sum())
    nz_te = int((te["chemprop_fp"] != 0).any(axis=1).sum())
    print(f"ChemProp nonzero: train {nz_tr}/{len(tr['chemprop_fp'])}  test {nz_te}/{len(te['chemprop_fp'])}")

    # Physchem (log transforms + z-score, fit on covered train)
    p_tr, p_te = preprocess_physchem(tr["physchem_feat"], tr["has_physchem"],
                                      te["physchem_feat"], te["has_physchem"])
    h_tr = tr["has_physchem"].astype(np.float32)
    h_te = te["has_physchem"].astype(np.float32)

    v4_tr, v4_te = v4_base(tr), v4_base(te)
    y_tr, y_te = tr["targets"].astype(np.float32), te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]

    print(f"train={len(tr['smiles'])}, test={len(te['smiles'])}, "
          f"physchem_covered={int(h_tr.sum())}/{len(h_tr)}")

    stage1_r2s, stage2_r2s = [], []
    for seed in range(N_SEEDS):
        print(f"\n[seed {seed}] Stage-1 (A2_chemprop)...")
        s1 = train_stage1_a2(seed, v4_tr, f_tr, th_tr, cp_tr, y_tr, device)
        s1_pred = predict_stage1(s1, v4_te, f_te, th_te, cp_te, device)
        s1_r2 = r2_per_prop(s1_pred, y_te)
        stage1_r2s.append(s1_r2)
        print(f"  Stage-1 core7={s1_r2['avg_core7']:.4f} "
              f"lignin={s1_r2.get('lignin_wt', float('nan')):.4f}")

        print(f"[seed {seed}] Stage-2 (hardfreeze + deep lignin head + physchem)...")
        s2 = train_stage2_lignin(s1, v4_tr, f_tr, th_tr, cp_tr, p_tr, h_tr, y_tr,
                                  device, seed=seed + 100)
        s2_pred = predict_stage2(s2, v4_te, f_te, th_te, cp_te, p_te, h_te, device)
        s2_r2 = r2_per_prop(s2_pred, y_te)
        stage2_r2s.append(s2_r2)
        print(f"  Stage-2 core7={s2_r2['avg_core7']:.4f} "
              f"lignin={s2_r2.get('lignin_wt', float('nan')):.4f}")

    s1 = summarize("Stage1_A2_chemprop", stage1_r2s)
    s2 = summarize("Stage2_A2_deep_lignin_physchem", stage2_r2s)

    print(f"\n{'='*70}\nA2 two-stage SUMMARY\n{'='*70}")
    print(f"{'Stage':<40}{'core7':>10}{'std':>10}{'lignin':>10}")
    print("-" * 70)
    for r in [s1, s2]:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<40}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}{lig:>10.4f}")

    out = V5 / "results" / "a2_two_stage.json"
    json.dump([s1, s2], open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
