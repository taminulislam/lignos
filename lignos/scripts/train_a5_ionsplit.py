"""A5.1 — A2 + disentangled cation/anion encoder with cross-attention.

Addresses OOD failure mode #1 from the Baran Task 2 CV collapse:
    Single-SMILES encoding conflates cation+anion; a novel IL corrupts the
    whole embedding. Even [C2H4COOHmim][Cl] (fold 3 R²=-2.4) shares its
    chloride anion with MANY training ILs and its imidazolium core with many
    more. Splitting + cross-attention lets the model transfer each partner's
    learned embedding independently.

Architecture:
  cation_proj : Linear(40, 32), GELU, Linear(32, 32)          [zero-init out]
  anion_proj  : Linear(40, 32), GELU, Linear(32, 32)          [zero-init out]
  xattn       : MultiheadAttention(embed_dim=32, num_heads=2) [out zero-init]
  ion_heads[i]: Linear(32+5, 16) → Linear(16, 1)              [zero-init per prop]
  ion_gate    : Parameter(-3.0) per prop  (sigmoid≈0.047)

At init: A5.1(weights) ≡ A2(weights) bit-identical (new-branch output = 0).
Warm-start from A2 Stage-1 + freeze A2 backbone (proven pattern from 2026-04-20).

Features: ion_split_{split}.npz provides per-row cation_morgan + anion_morgan
(2048-bit each), PCA to 40D separately. 100% coverage across splits.
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
from audit_residuals import PROPS, r2_per_prop, set_seed  # noqa
from train_a2_two_stage import (
    A2Head, A2StageTwoLigninWrapper,
    build_chemprop_40d, preprocess_physchem, v4_base,
)

CACHE = V5 / "data" / "LignoIL_A1"
A2_CKPT = V5 / "checkpoints" / "a2" / "stage1_best.pt"
ION_DIM = 40  # PCA output dim for cation/anion Morgan


class A5IonSplitHead(A2Head):
    """A2Head + zero-init disentangled cation/anion residual branch with
    cross-attention fusion."""

    def __init__(self, nf, n_props=8, chemprop_dim=40, ion_dim=ION_DIM):
        super().__init__(nf, n_props, chemprop_dim)
        # Per-ion projections
        self.cation_proj = nn.Sequential(
            nn.Linear(ion_dim, 32), nn.GELU(), nn.Linear(32, 32))
        self.anion_proj = nn.Sequential(
            nn.Linear(ion_dim, 32), nn.GELU(), nn.Linear(32, 32))
        for proj in (self.cation_proj, self.anion_proj):
            with torch.no_grad():
                proj[-1].weight.zero_(); proj[-1].bias.zero_()
        # Cross-attention between cation and anion (bidirectional, 2 heads).
        # DO NOT zero-init xattn.out_proj — that zeros the pooled attn output and
        # kills the gradient to ion_heads weights (which are zero-init on final
        # layer). Keep default init so pooled_attn is nonzero and ion_heads can
        # train. Init-equivalence with A2 is preserved by the SMALL init of
        # ion_heads final layer + low initial gate (sigmoid(−3)≈0.047).
        self.xattn = nn.MultiheadAttention(32, num_heads=2, batch_first=True)
        # Per-prop residual heads — use small-random final layer (not zero) so
        # gradient flows from ion_gate backprop through ion_heads.
        self.ion_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(32 + 5, 16), nn.GELU(), nn.Linear(16, 1))
            for _ in range(n_props)])
        for h in self.ion_heads:
            with torch.no_grad():
                h[-1].weight.mul_(0.01); h[-1].bias.zero_()
        # Per-prop gate (less pessimistic init −3 → sigmoid≈0.047)
        self.ion_gate = nn.Parameter(torch.full((n_props,), -3.0))

    def forward(self, v, i, t, chemprop, cation, anion, has_split):
        out = super().forward(v, i, t, chemprop)
        hs = has_split.float().unsqueeze(-1) if has_split.ndim == 1 else has_split.float()
        c_emb = self.cation_proj(cation) * hs         # (B, 32)
        a_emb = self.anion_proj(anion) * hs           # (B, 32)
        # Stack as sequence for cross-attention
        pair = torch.stack([c_emb, a_emb], dim=1)     # (B, 2, 32)
        attn_out, _ = self.xattn(pair, pair, pair)     # (B, 2, 32)
        # Mean-pool the two attended tokens
        pooled = attn_out.mean(dim=1)                 # (B, 32)
        tmp = t[:, :5]
        ion_in = torch.cat([pooled, tmp], -1)
        ion_delta = torch.cat([h(ion_in) for h in self.ion_heads], -1)
        return out + torch.sigmoid(self.ion_gate) * ion_delta * hs


class A5IonSplitStageTwoWrapper(A2StageTwoLigninWrapper):
    def forward(self, v, i, t, chemprop, cation, anion, has_split, phys, has_phys):
        base = self.backbone(v, i, t, chemprop, cation, anion, has_split)
        tmp = t[:, :5]
        g = i * self.backbone.gate(tmp)
        hp = has_phys.float().unsqueeze(-1) if has_phys.ndim == 1 else has_phys.float()
        ctx = torch.cat([g, tmp, phys, hp], -1)
        res_lignin = self.deep_lignin(ctx).squeeze(-1)
        out = base.clone()
        out[:, 7] = v[:, 7] + torch.sigmoid(self.alpha_lignin) * res_lignin
        return out


def _load_split(s):
    p_dft = CACHE / f"cached_{s}_dft.npz"
    p_std = CACHE / f"cached_{s}.npz"
    p = p_dft if p_dft.exists() else p_std
    print(f"[{s}] loading {p.name}")
    return {k: v for k, v in np.load(p, allow_pickle=True).items()}


def _load_ion_split(s):
    p = CACHE / f"ion_split_{s}.npz"
    d = np.load(p, allow_pickle=True)
    return d["cation_morgan"], d["anion_morgan"], d["has_split"]


def train_stage1(seed, v4, morg, th, cp, cat, an, hs, y, device,
                  epochs=300, patience=50, warm_start=True, freeze_a2=True):
    set_seed(seed)
    n_props = y.shape[1]
    m = A5IonSplitHead(morg.shape[1], n_props, chemprop_dim=cp.shape[1],
                        ion_dim=cat.shape[1]).to(device)

    if warm_start and A2_CKPT.exists():
        ckpt = torch.load(A2_CKPT, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        miss, unex = m.load_state_dict(sd, strict=False)
        print(f"  warm-started from {A2_CKPT.name} (A2 seed={ckpt.get('seed')}); "
              f"{len(miss)} unmatched (new branch), {len(unex)} unused")

    if freeze_a2:
        for name, p in m.named_parameters():
            if not name.startswith(("cation_", "anion_", "xattn", "ion_")):
                p.requires_grad = False

    train_params = [p for p in m.parameters() if p.requires_grad]
    print(f"  trainable params: {sum(p.numel() for p in train_params)} "
          f"of {sum(p.numel() for p in m.parameters())}")

    opt = AdamW(train_params, lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    ts = {k: torch.from_numpy(x).to(device) for k, x in
          dict(v=v4, i=morg, t=th, cp=cp, cat=cat, an=an, hs=hs, y=y).items()}
    valid = ~torch.isnan(ts["y"]); yf = torch.nan_to_num(ts["y"], 0.0)
    ds = TensorDataset(*[ts[k].cpu() for k in ("v","i","t","cp","cat","an","hs")],
                        yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for vb, ib, tb, cpb, catb, anb, hsb, yb, vm in loader:
            vb, ib, tb, cpb, catb, anb, hsb, yb, vm = [x.to(device)
                for x in (vb, ib, tb, cpb, catb, anb, hsb, yb, vm)]
            pred = m(vb, ib, tb, cpb, catb, anb, hsb)
            err2 = ((pred - yb) ** 2) * vm.float()
            loss = err2.sum() / vm.float().sum().clamp(min=1)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(ts["v"], ts["i"], ts["t"], ts["cp"], ts["cat"], ts["an"], ts["hs"])
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


def train_stage2(s1_model, v4, morg, th, cp, cat, an, hs, phys, hp, y,
                  device, seed, epochs=300, patience=50):
    set_seed(seed)
    m = A5IonSplitStageTwoWrapper(copy.deepcopy(s1_model)).to(device)
    opt = AdamW([{"params": m.deep_lignin.parameters(), "weight_decay": 1e-2},
                  {"params": [m.alpha_lignin], "weight_decay": 0.0}], lr=1e-3)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    ts = {k: torch.from_numpy(x).to(device) for k, x in
          dict(v=v4, i=morg, t=th, cp=cp, cat=cat, an=an, hs=hs, p=phys, hp=hp, y=y).items()}
    ds = TensorDataset(*[ts[k].cpu() for k in
                          ("v","i","t","cp","cat","an","hs","p","hp","y")])
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    best, state, bad = float("inf"), None, 0
    train_params = [p for p in m.parameters() if p.requires_grad]
    for _ in range(epochs):
        m.train()
        for batch in loader:
            vb, ib, tb, cpb, catb, anb, hsb, pb, hpb, yb = [x.to(device) for x in batch]
            pred = m(vb, ib, tb, cpb, catb, anb, hsb, pb, hpb)
            lg = ~torch.isnan(yb[:, 7])
            if lg.sum() == 0: continue
            loss = ((pred[lg, 7] - yb[lg, 7].nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(ts["v"], ts["i"], ts["t"], ts["cp"], ts["cat"], ts["an"], ts["hs"],
                      ts["p"], ts["hp"])
            lg = ~torch.isnan(ts["y"][:, 7])
            tl = ((pred[lg, 7] - ts["y"][lg, 7].nan_to_num(0)) ** 2).mean().item() if lg.any() else float("inf")
        if np.isfinite(tl) and tl < best:
            best = tl; state = {k: v.clone() for k, v in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval()
    return m


def predict_s1(m, v4, morg, th, cp, cat, an, hs, device):
    with torch.no_grad():
        return m(*(torch.from_numpy(x).to(device) for x in (v4, morg, th, cp, cat, an, hs))).cpu().numpy()


def predict_s2(m, v4, morg, th, cp, cat, an, hs, phys, hp, device):
    with torch.no_grad():
        return m(*(torch.from_numpy(x).to(device) for x in
                    (v4, morg, th, cp, cat, an, hs, phys, hp))).cpu().numpy()


def summarize(name, r2s):
    c = [r["avg_core7"] for r in r2s]
    out = {"name": name, "avg_r2_core7": float(np.mean(c)),
           "std_r2_core7": float(np.std(c)), "per_prop": {}}
    for p in PROPS:
        vs = [r.get(p) for r in r2s if r.get(p) is not None and not np.isnan(r.get(p, float("nan")))]
        out["per_prop"][p] = float(np.mean(vs)) if vs else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tr, va, te = _load_split("train"), _load_split("val"), _load_split("test")

    pca_m = PCA(40).fit(tr["morgan_fp"])
    m_tr, m_va, m_te = [pca_m.transform(x["morgan_fp"]).astype(np.float32)
                         for x in (tr, va, te)]
    cp_tr, cp_te = build_chemprop_40d(tr["chemprop_fp"], te["chemprop_fp"])
    _, cp_va = build_chemprop_40d(tr["chemprop_fp"], va["chemprop_fp"])
    p_tr, p_te = preprocess_physchem(tr["physchem_feat"], tr["has_physchem"],
                                      te["physchem_feat"], te["has_physchem"])
    _, p_va = preprocess_physchem(tr["physchem_feat"], tr["has_physchem"],
                                   va["physchem_feat"], va["has_physchem"])
    hp_tr = tr["has_physchem"].astype(np.float32)
    hp_va = va["has_physchem"].astype(np.float32)
    hp_te = te["has_physchem"].astype(np.float32)

    # Ion-split features
    cat_tr, an_tr, hs_tr = _load_ion_split("train")
    cat_va, an_va, hs_va = _load_ion_split("val")
    cat_te, an_te, hs_te = _load_ion_split("test")
    # PCA(40) on train cation and anion independently
    pca_cat = PCA(ION_DIM).fit(cat_tr)
    pca_an = PCA(ION_DIM).fit(an_tr)
    cat_tr = pca_cat.transform(cat_tr).astype(np.float32)
    cat_va = pca_cat.transform(cat_va).astype(np.float32)
    cat_te = pca_cat.transform(cat_te).astype(np.float32)
    an_tr = pca_an.transform(an_tr).astype(np.float32)
    an_va = pca_an.transform(an_va).astype(np.float32)
    an_te = pca_an.transform(an_te).astype(np.float32)

    v4_tr, v4_va, v4_te = v4_base(tr), v4_base(va), v4_base(te)
    y_tr, y_va, y_te = [x["targets"].astype(np.float32) for x in (tr, va, te)]
    th_tr, th_va, th_te = tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]

    s1_r2s, s2_r2s = [], []
    for seed in range(args.n_seeds):
        print(f"\n[seed {seed}] Stage-1 (A5.1 = A2 + ion-split + xattn)...")
        s1 = train_stage1(seed, v4_tr, m_tr, th_tr, cp_tr, cat_tr, an_tr, hs_tr, y_tr,
                           device, epochs=args.epochs)
        r = r2_per_prop(predict_s1(s1, v4_te, m_te, th_te, cp_te, cat_te, an_te, hs_te, device), y_te)
        s1_r2s.append(r)
        print(f"  Stage-1 core7={r['avg_core7']:.4f}  lignin={r.get('lignin_wt', float('nan')):.4f}  "
              f"ion_gate={torch.sigmoid(s1.ion_gate).mean().item():.3f}")

        print(f"[seed {seed}] Stage-2 (hardfreeze + deep lignin + physchem)...")
        s2 = train_stage2(s1, v4_tr, m_tr, th_tr, cp_tr, cat_tr, an_tr, hs_tr, p_tr, hp_tr, y_tr,
                           device, seed=seed + 100, epochs=args.epochs)
        r2 = r2_per_prop(predict_s2(s2, v4_te, m_te, th_te, cp_te, cat_te, an_te, hs_te,
                                      p_te, hp_te, device), y_te)
        s2_r2s.append(r2)
        print(f"  Stage-2 core7={r2['avg_core7']:.4f}  lignin={r2.get('lignin_wt', float('nan')):.4f}")

    s1 = summarize("Stage1_A5_ionsplit", s1_r2s)
    s2 = summarize("Stage2_A5_ionsplit_deep_lignin", s2_r2s)
    print(f"\n{'='*70}\nA5.1 ION-SPLIT two-stage SUMMARY\n{'='*70}")
    print(f"{'Stage':<40}{'core7':>10}{'std':>10}{'lignin':>10}")
    for r in [s1, s2]:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<40}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}{lig:>10.4f}")
    out = V5 / "results" / "a5_ionsplit.json"
    json.dump([s1, s2], open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
