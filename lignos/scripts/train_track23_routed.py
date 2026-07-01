"""Track 2+3: use the v2 cache (real thermo on extension rows) + dual lignin
head routed by measurement_type.

Design
------
  * Cache v2 (`data/LignoIL_unified_v2`) has correct thermo_feat on the 191
    extension rows and a new `measurement_type` field (1=yield, 2=solubility).
  * Lignin "head 7" is replaced by TWO heads: `head_yield` and `head_sol`.
    Forward returns whichever one the row's measurement_type selects.
  * Stage-1 trains both heads with shared backbone; Stage-2 hardfreeze trains
    deep versions of both heads on top of the frozen core-7 + gate.

Test has 39 lignin rows, all solubility — so `head_sol` is what gets graded,
but `head_yield` still benefits the shared backbone during training.
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

V2 = V5 / "data" / "LignoIL_unified_v2"
N_SEEDS = 10


def load_split(s):
    d = np.load(V2 / f"cached_{s}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    return (0.4 * c["preds_fusion"] + 0.6 * c["preds_chemprop"]).astype(np.float32)


class RoutedHead(nn.Module):
    """Core 7 shallow heads + two routed deep/shallow sub-heads for lignin."""
    def __init__(self, nf, deep_lignin=False):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, nf), nn.Sigmoid())
        ctx = 5; head_in = nf + ctx
        self.heads_core = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, 32), nn.GELU(), nn.Linear(32, 1)) for _ in range(7)
        ])
        def mk(kind):
            if kind == "deep":
                return nn.Sequential(
                    nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(64, 1),
                )
            return nn.Sequential(nn.Linear(head_in, 32), nn.GELU(), nn.Linear(32, 1))
        kind = "deep" if deep_lignin else "shallow"
        self.head_yield = mk(kind)
        self.head_sol = mk(kind)
        self.alphas = nn.Parameter(torch.full((8,), -3.0))
        for h in list(self.heads_core) + [self.head_yield, self.head_sol]:
            with torch.no_grad():
                h[-1].weight.mul_(0.01); h[-1].bias.zero_()

    def forward(self, v, i, t, m_type):
        """m_type: int tensor (N,) with 1=yield, 2=solubility, 0=none.
        Returns prediction (N, 8) where col 7 is the routed lignin output."""
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        inp = torch.cat([g, tmp], -1)
        core = torch.cat([h(inp) for h in self.heads_core], -1)         # (N, 7)
        ly = self.head_yield(inp)                                         # (N, 1)
        ls = self.head_sol(inp)                                           # (N, 1)
        # Route: m_type==1 → yield, else solubility.
        is_yield = (m_type == 1).unsqueeze(-1).float()
        lignin = is_yield * ly + (1 - is_yield) * ls
        res = torch.cat([core, lignin], -1)                               # (N, 8)
        return v + torch.sigmoid(self.alphas) * res


def train_stage1(seed, tr_v, tr_f, tr_th, tr_mt, tr_y, device, epochs=400, patience=60):
    set_seed(seed)
    m = RoutedHead(tr_f.shape[1], deep_lignin=False).to(device)
    opt = AdamW(m.parameters(), lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    tt = {k: torch.from_numpy(v).to(device) for k, v in
           dict(v=tr_v, f=tr_f, t=tr_th, y=tr_y).items()}
    mt_t = torch.from_numpy(tr_mt).long().to(device)
    valid = ~torch.isnan(tt["y"]); yf = torch.nan_to_num(tt["y"], 0.0)
    ds = TensorDataset(tt["v"].cpu(), tt["f"].cpu(), tt["t"].cpu(),
                        mt_t.cpu(), yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for v, im, t, mt, y, vm in loader:
            v, im, t, mt, y, vm = [x.to(device) for x in (v, im, t, mt, y, vm)]
            pred = m(v, im, t, mt)
            e2 = ((pred - y) ** 2) * vm
            loss = (e2.sum(0) / vm.sum(0).clamp(min=1)).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(tt["v"], tt["f"], tt["t"], mt_t)
            e2 = ((pred - yf) ** 2) * valid
            tl = (e2.sum(0) / valid.sum(0).clamp(min=1)).mean().item()
        if np.isfinite(tl) and tl < best:
            best, state, bad = tl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval(); return m


def train_stage2_hardfz(s1, tr_v, tr_f, tr_th, tr_mt, tr_y, device, seed,
                          epochs=400, patience=60):
    set_seed(seed)
    m = copy.deepcopy(s1).to(device)
    for p in m.parameters(): p.requires_grad = False
    # Swap both lignin sub-heads for deep variants
    nf = m.gate[2].out_features
    hin = nf + 5
    deep = lambda: nn.Sequential(
        nn.Linear(hin, 128), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(64, 1),
    ).to(device)
    m.head_yield = deep(); m.head_sol = deep()
    for h in (m.head_yield, m.head_sol):
        with torch.no_grad():
            h[-1].weight.mul_(0.01); h[-1].bias.zero_()
        for p in h.parameters(): p.requires_grad = True
    m.alphas.requires_grad = True
    mask = torch.zeros_like(m.alphas); mask[7] = 1.0
    m.alphas.register_hook(lambda g: g * mask)
    hp = list(m.head_yield.parameters()) + list(m.head_sol.parameters())
    opt = AdamW([{"params": hp, "weight_decay": 1e-2},
                  {"params": [m.alphas], "weight_decay": 0.0}], lr=1e-3)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    opts = hp + [m.alphas]
    tt = {k: torch.from_numpy(v).to(device) for k, v in
           dict(v=tr_v, f=tr_f, t=tr_th, y=tr_y).items()}
    mt_t = torch.from_numpy(tr_mt).long().to(device)
    ds = TensorDataset(tt["v"].cpu(), tt["f"].cpu(), tt["t"].cpu(),
                        mt_t.cpu(), tt["y"].cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for v, im, t, mt, y in loader:
            v, im, t, mt, y = [x.to(device) for x in (v, im, t, mt, y)]
            pred = m(v, im, t, mt)
            lg = ~torch.isnan(y[:, 7])
            if lg.sum() == 0: continue
            loss = ((pred[lg, 7] - y[lg, 7].nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(opts, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(tt["v"], tt["f"], tt["t"], mt_t)
            lg = ~torch.isnan(tt["y"][:, 7])
            tl = ((pred[lg, 7] - tt["y"][lg, 7].nan_to_num(0)) ** 2).mean().item() if lg.any() else float("inf")
        if np.isfinite(tl) and tl < best:
            best, state, bad = tl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval(); return m


def predict(m, v, f, t, mt, device):
    tt = [torch.from_numpy(x).to(device) for x in (v, f, t)]
    mt_t = torch.from_numpy(mt).long().to(device)
    m.eval()
    with torch.no_grad():
        return m(tt[0], tt[1], tt[2], mt_t).cpu().numpy()


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
    tr = load_split("train"); te = load_split("test")
    v4_tr, v4_te = v4_base(tr), v4_base(te)
    y_tr = tr["targets"].astype(np.float32); y_te = te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]
    mt_tr = tr["measurement_type"].astype(np.int8)
    # Test: all solubility (confirmed by v2 build). Force to 2 for safety.
    mt_te = np.full(len(y_te), 2, dtype=np.int8)
    print(f"Train: {len(y_tr)} rows ({(mt_tr==1).sum()} yield, {(mt_tr==2).sum()} solubility)")
    print(f"Test:  {len(y_te)} rows ({(mt_te==2).sum()} solubility)")

    pca = PCA(40).fit(tr["morgan_fp"])
    f_tr = pca.transform(tr["morgan_fp"]).astype(np.float32)
    f_te = pca.transform(te["morgan_fp"]).astype(np.float32)

    # Stage 1
    s1_models, s1_r2 = [], []
    for seed in range(N_SEEDS):
        m = train_stage1(seed, v4_tr, f_tr, th_tr, mt_tr, y_tr, device)
        s1_models.append(m)
        s1_r2.append(r2_per_prop(predict(m, v4_te, f_te, th_te, mt_te, device), y_te))
    s1 = summarize("Track23/Stage1_routed", s1_r2)
    print(f"\nStage1 core7={s1['avg_r2_core7']:.4f}±{s1['std_r2_core7']:.4f}  lignin={s1['per_prop']['lignin_wt']:.4f}")
    for p in PROPS:
        print(f"  {p:12s}: {s1['per_prop'][p]:.4f}")

    # Stage 2
    s2_r2 = []
    for seed in range(N_SEEDS):
        m2 = train_stage2_hardfz(s1_models[seed], v4_tr, f_tr, th_tr, mt_tr, y_tr,
                                   device, seed + 100)
        s2_r2.append(r2_per_prop(predict(m2, v4_te, f_te, th_te, mt_te, device), y_te))
    s2s = summarize("Track23/Stage2_hardfz_routed", s2_r2)
    print(f"\nStage2 core7={s2s['avg_r2_core7']:.4f}±{s2s['std_r2_core7']:.4f}  lignin={s2s['per_prop']['lignin_wt']:.4f}")
    for p in PROPS:
        print(f"  {p:12s}: {s2s['per_prop'][p]:.4f}")

    results = [s1, s2s]
    out = V5 / "results" / "track23_routed.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
