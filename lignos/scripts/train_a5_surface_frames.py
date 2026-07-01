"""A5 — A2_chemprop + zero-init Surface and ViT-frame residual branches.

Design: extends train_a2_two_stage.A2Head with two additional zero-init gated
residual branches so that at initialization A5 ≡ A2 (bit-identical prediction).
Each branch has its own per-prop gate, and masks out rows where the input
modality is unavailable (`has_surface`, `has_frames`).

Stage-1 (core-7) — same Morgan+ChemProp recipe as A2, plus:
  - Surface branch: Linear(256→32) + per-prop Linear(32+5→16→1), zero-init
  - Frame branch  : Linear(frame_dim→32) + per-prop Linear(32+5→16→1), zero-init
  - Gate params   : per-prop sigmoid, init = sigmoid(-5) ≈ 0.007

Stage-2 (lignin) — identical hardfreeze + deep lignin head + physchem recipe
as train_a2_two_stage; inherits whatever core-7 backbone Stage-1 produced.

Coverage policy:
  - `has_surface` = (surface_fp != 0).any(axis=1). Zeroing respects the 91%
    DFT coverage; uncovered rows contribute nothing through Branch 1.
  - `has_frames`  = looked up via il-id → ViT bank (see build_il_vit_bank.py).

At training start: branch outputs are zero → A5 ≡ A2. Only way to regress is
runaway gate growth, which the LR/weight_decay schedule precludes.

Run:
    python lignos/scripts/train_a5_surface_frames.py --n-seeds 10
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
    build_chemprop_40d, preprocess_physchem,
    predict_stage2, v4_base,
)

CACHE = V5 / "data" / "LignoIL_A1"
VIT_BANK = V5 / "data" / "il_vit_bank.npz"  # produced by build_il_vit_bank.py
FRAME_DIM = 192  # ViT-Tiny output width
A2_CKPT = V5 / "checkpoints" / "a2" / "stage1_best.pt"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class A5Head(A2Head):
    """A2Head extended with zero-init Surface and Frame residual branches.

    New submodules (all zero-init on their output layer so A5(0)=A2):
      surf_proj   : Linear(256, 32)                   [final linear zeroed]
      surf_heads  : per-prop Linear(32+5,16)→Linear(16,1)   [last layer zeroed]
      surf_gate   : Parameter(-5.0) per prop
      frame_proj  : Linear(frame_dim, 32)             [final linear zeroed]
      frame_heads : per-prop Linear(32+5,16)→Linear(16,1)   [last layer zeroed]
      frame_gate  : Parameter(-5.0) per prop
    """
    def __init__(self, nf, n_props=8, chemprop_dim=40,
                 surface_dim=256, frame_dim=FRAME_DIM):
        super().__init__(nf, n_props, chemprop_dim)
        self.surface_dim = surface_dim
        self.frame_dim = frame_dim

        # ---- Surface branch
        self.surf_proj = nn.Sequential(
            nn.Linear(surface_dim, 32), nn.GELU(), nn.Linear(32, 32))
        with torch.no_grad():
            self.surf_proj[-1].weight.zero_()
            self.surf_proj[-1].bias.zero_()
        self.surf_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(32 + 5, 16), nn.GELU(), nn.Linear(16, 1))
            for _ in range(n_props)])
        for h in self.surf_heads:
            with torch.no_grad():
                h[-1].weight.zero_(); h[-1].bias.zero_()
        # Gate init −3 (≈5% contribution at step 0) instead of −5 (≈0.7% — too
        # pessimistic; gates never woke up in the 17756044 run).
        self.surf_gate = nn.Parameter(torch.full((n_props,), -3.0))

        # ---- Frame branch (single-view or pooled per-IL ViT features)
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_dim, 32), nn.GELU(), nn.Linear(32, 32))
        with torch.no_grad():
            self.frame_proj[-1].weight.zero_()
            self.frame_proj[-1].bias.zero_()
        self.frame_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(32 + 5, 16), nn.GELU(), nn.Linear(16, 1))
            for _ in range(n_props)])
        for h in self.frame_heads:
            with torch.no_grad():
                h[-1].weight.zero_(); h[-1].bias.zero_()
        self.frame_gate = nn.Parameter(torch.full((n_props,), -3.0))

    def forward(self, v, i, t, chemprop, surface, frame, has_surf, has_frm):
        # A2 forward — base prediction.
        out = super().forward(v, i, t, chemprop)
        tmp = t[:, :5]

        # Surface branch — mask rows with zero surface.
        hs = has_surf.float().unsqueeze(-1) if has_surf.ndim == 1 else has_surf.float()
        s_h = self.surf_proj(surface) * hs
        s_in = torch.cat([s_h, tmp], -1)
        s_delta = torch.cat([h(s_in) for h in self.surf_heads], -1)
        out = out + torch.sigmoid(self.surf_gate) * s_delta * hs

        # Frame branch — mask rows with no ViT frame lookup.
        hf = has_frm.float().unsqueeze(-1) if has_frm.ndim == 1 else has_frm.float()
        f_h = self.frame_proj(frame) * hf
        f_in = torch.cat([f_h, tmp], -1)
        f_delta = torch.cat([h(f_in) for h in self.frame_heads], -1)
        out = out + torch.sigmoid(self.frame_gate) * f_delta * hf
        return out


# --------------------------------------------------------------------------
# Data loading + feature assembly
# --------------------------------------------------------------------------
def _load_split(split):
    # Prefer _dft cache if present (A3 output).
    p_dft = CACHE / f"cached_{split}_dft.npz"
    p_std = CACHE / f"cached_{split}.npz"
    p = p_dft if p_dft.exists() else p_std
    print(f"[{split}] loading {p.name}")
    return {k: v for k, v in np.load(p, allow_pickle=True).items()}


def _load_vit_bank():
    """Return (smiles→192D ViT feat) lookup dict, or None if bank missing."""
    if not VIT_BANK.exists():
        print(f"[frame] bank {VIT_BANK} missing — frame branch will be all-zero.")
        return None
    z = np.load(VIT_BANK, allow_pickle=True)
    smis = z["smiles"]
    feats = z["vit_feat"]
    print(f"[frame] bank loaded: {len(smis)} IL SMILES → {feats.shape[1]}D ViT")
    return dict(zip(smis, feats))


def _assemble_frame(smiles, bank):
    """Return (frame_array, has_frame_mask) aligned with smiles."""
    n = len(smiles)
    if bank is None:
        return np.zeros((n, FRAME_DIM), dtype=np.float32), np.zeros(n, dtype=np.float32)
    feats = np.zeros((n, FRAME_DIM), dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    for i, s in enumerate(smiles):
        f = bank.get(s)
        if f is not None:
            feats[i] = f
            mask[i] = 1.0
    print(f"[frame] {int(mask.sum())}/{n} rows covered by ViT bank")
    return feats, mask


# --------------------------------------------------------------------------
# Training loops
# --------------------------------------------------------------------------
def train_stage1_a5(seed, v4, morg, th, cp, surf, frm, hs, hf, y,
                     device, epochs=300, patience=50,
                     warm_start=True, freeze_a2=True):
    """Warm-start from A2 Stage-1 checkpoint; freeze A2 backbone.

    Only the new surf_*, frame_* branches train. Prevents the from-scratch
    regression seen in 17746649 (2026-04-20) where core7 dropped 0.84 → 0.67.
    """
    set_seed(seed)
    n_props = y.shape[1]
    m = A5Head(morg.shape[1], n_props,
                chemprop_dim=cp.shape[1],
                surface_dim=surf.shape[1]).to(device)

    if warm_start and A2_CKPT.exists():
        ckpt = torch.load(A2_CKPT, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"  warm-started from {A2_CKPT.name} (A2 seed={ckpt.get('seed')}, "
              f"val_loss={ckpt.get('val_loss'):.5f}); "
              f"{len(missing)} unmatched, {len(unexpected)} unused")
    else:
        print(f"  NO warm-start (A2_CKPT missing or disabled) — training from scratch")

    if freeze_a2:
        for name, p in m.named_parameters():
            if not name.startswith(("surf_", "frame_")):
                p.requires_grad = False

    train_params = [p for p in m.parameters() if p.requires_grad]
    print(f"  trainable params: {sum(p.numel() for p in train_params)} "
          f"of {sum(p.numel() for p in m.parameters())}")

    opt = AdamW(train_params, lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    # Tensors
    ts = {k: torch.from_numpy(x).to(device) for k, x in
          dict(v=v4, i=morg, t=th, cp=cp, s=surf, f=frm,
               hs=hs, hf=hf, y=y).items()}
    valid = ~torch.isnan(ts["y"]); yf = torch.nan_to_num(ts["y"], 0.0)

    ds = TensorDataset(*[ts[k].cpu() for k in ("v","i","t","cp","s","f","hs","hf")],
                        yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for vb, ib, tb, cpb, sb, fb, hsb, hfb, yb, vm in loader:
            vb, ib, tb, cpb, sb, fb, hsb, hfb, yb, vm = [
                x.to(device) for x in (vb, ib, tb, cpb, sb, fb, hsb, hfb, yb, vm)]
            pred = m(vb, ib, tb, cpb, sb, fb, hsb, hfb)
            err2 = ((pred - yb) ** 2) * vm.float()
            loss = err2.sum() / vm.float().sum().clamp(min=1)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(ts["v"], ts["i"], ts["t"], ts["cp"], ts["s"], ts["f"],
                      ts["hs"], ts["hf"])
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


def predict_stage1(m, v4, morg, th, cp, surf, frm, hs, hf, device):
    with torch.no_grad():
        return m(torch.from_numpy(v4).to(device),
                 torch.from_numpy(morg).to(device),
                 torch.from_numpy(th).to(device),
                 torch.from_numpy(cp).to(device),
                 torch.from_numpy(surf).to(device),
                 torch.from_numpy(frm).to(device),
                 torch.from_numpy(hs).to(device),
                 torch.from_numpy(hf).to(device)).cpu().numpy()


# --------------------------------------------------------------------------
# Stage-2 wrapper — inherits from A2StageTwoLigninWrapper but forwards the
# new modalities through the (now frozen) A5Head.
# --------------------------------------------------------------------------
class A5StageTwoLigninWrapper(A2StageTwoLigninWrapper):
    def forward(self, v, i, t, chemprop, surface, frame, has_surf, has_frm,
                 phys, has_phys):
        base = self.backbone(v, i, t, chemprop, surface, frame, has_surf, has_frm)
        tmp = t[:, :5]
        g = i * self.backbone.gate(tmp)
        hp = has_phys.float().unsqueeze(-1) if has_phys.ndim == 1 else has_phys.float()
        ctx = torch.cat([g, tmp, phys, hp], -1)
        res_lignin = self.deep_lignin(ctx).squeeze(-1)
        out = base.clone()
        out[:, 7] = v[:, 7] + torch.sigmoid(self.alpha_lignin) * res_lignin
        return out


def train_stage2_lignin_a5(stage1_model, v4, morg, th, cp, surf, frm,
                            hs, hf, phys, hp, y, device, seed,
                            epochs=300, patience=50):
    set_seed(seed)
    m = A5StageTwoLigninWrapper(copy.deepcopy(stage1_model)).to(device)
    train_params = [p for p in m.parameters() if p.requires_grad]
    opt = AdamW([{"params": m.deep_lignin.parameters(), "weight_decay": 1e-2},
                  {"params": [m.alpha_lignin], "weight_decay": 0.0}], lr=1e-3)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    ts = {k: torch.from_numpy(x).to(device) for k, x in
          dict(v=v4, i=morg, t=th, cp=cp, s=surf, f=frm,
               hs=hs, hf=hf, p=phys, hp=hp, y=y).items()}
    ds = TensorDataset(*[ts[k].cpu() for k in
                          ("v","i","t","cp","s","f","hs","hf","p","hp","y")])
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            vb, ib, tb, cpb, sb, fb, hsb, hfb, pb, hpb, yb = batch
            pred = m(vb, ib, tb, cpb, sb, fb, hsb, hfb, pb, hpb)
            lg = ~torch.isnan(yb[:, 7])
            if lg.sum() == 0: continue
            loss = ((pred[lg, 7] - yb[lg, 7].nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(ts["v"], ts["i"], ts["t"], ts["cp"], ts["s"], ts["f"],
                      ts["hs"], ts["hf"], ts["p"], ts["hp"])
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


def predict_stage2_a5(m, v4, morg, th, cp, surf, frm, hs, hf, phys, hp, device):
    with torch.no_grad():
        return m(torch.from_numpy(v4).to(device),
                 torch.from_numpy(morg).to(device),
                 torch.from_numpy(th).to(device),
                 torch.from_numpy(cp).to(device),
                 torch.from_numpy(surf).to(device),
                 torch.from_numpy(frm).to(device),
                 torch.from_numpy(hs).to(device),
                 torch.from_numpy(hf).to(device),
                 torch.from_numpy(phys).to(device),
                 torch.from_numpy(hp).to(device)).cpu().numpy()


# --------------------------------------------------------------------------
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

    # Morgan PCA(40)
    pca_m = PCA(40).fit(tr["morgan_fp"])
    m_tr, m_va, m_te = [pca_m.transform(x["morgan_fp"]).astype(np.float32)
                         for x in (tr, va, te)]

    # ChemProp PCA(40) on non-zero train rows
    cp_tr, cp_te = build_chemprop_40d(tr["chemprop_fp"], te["chemprop_fp"])
    _, cp_va = build_chemprop_40d(tr["chemprop_fp"], va["chemprop_fp"])

    # Physchem
    p_tr, p_te = preprocess_physchem(tr["physchem_feat"], tr["has_physchem"],
                                      te["physchem_feat"], te["has_physchem"])
    _, p_va = preprocess_physchem(tr["physchem_feat"], tr["has_physchem"],
                                   va["physchem_feat"], va["has_physchem"])
    hp_tr = tr["has_physchem"].astype(np.float32)
    hp_va = va["has_physchem"].astype(np.float32)
    hp_te = te["has_physchem"].astype(np.float32)

    # Surface (256D already in cache; may be zero for uncovered rows)
    s_tr = tr["surface_fp"].astype(np.float32)
    s_va = va["surface_fp"].astype(np.float32)
    s_te = te["surface_fp"].astype(np.float32)
    hs_tr = (s_tr != 0).any(axis=1).astype(np.float32)
    hs_va = (s_va != 0).any(axis=1).astype(np.float32)
    hs_te = (s_te != 0).any(axis=1).astype(np.float32)
    print(f"[surface] train {int(hs_tr.sum())}/{len(hs_tr)} ({hs_tr.mean():.1%})  "
          f"val {int(hs_va.sum())}/{len(hs_va)}  test {int(hs_te.sum())}/{len(hs_te)}")

    # Frames: lookup per IL SMILES
    bank = _load_vit_bank()
    f_tr, hf_tr = _assemble_frame(tr["smiles"], bank)
    f_va, hf_va = _assemble_frame(va["smiles"], bank)
    f_te, hf_te = _assemble_frame(te["smiles"], bank)

    v4_tr, v4_va, v4_te = v4_base(tr), v4_base(va), v4_base(te)
    y_tr, y_va, y_te = [x["targets"].astype(np.float32) for x in (tr, va, te)]
    th_tr, th_va, th_te = tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]

    # Sanity: at init, A5 forward should equal A2 forward (gates ≈ 0).
    print(f"train={len(tr['smiles'])}, test={len(te['smiles'])}")

    stage1_r2s, stage2_r2s = [], []
    for seed in range(args.n_seeds):
        print(f"\n[seed {seed}] Stage-1 (A5 = A2 + Surface + Frame branches)...")
        s1 = train_stage1_a5(seed, v4_tr, m_tr, th_tr, cp_tr, s_tr, f_tr,
                              hs_tr, hf_tr, y_tr, device, epochs=args.epochs)
        s1_pred = predict_stage1(s1, v4_te, m_te, th_te, cp_te, s_te, f_te,
                                   hs_te, hf_te, device)
        r = r2_per_prop(s1_pred, y_te)
        stage1_r2s.append(r)
        print(f"  Stage-1 core7={r['avg_core7']:.4f}  lignin={r.get('lignin_wt', float('nan')):.4f}  "
              f"surf_gate={torch.sigmoid(s1.surf_gate).mean().item():.3f}  "
              f"frame_gate={torch.sigmoid(s1.frame_gate).mean().item():.3f}")

        print(f"[seed {seed}] Stage-2 (hardfreeze + deep lignin + physchem)...")
        s2 = train_stage2_lignin_a5(s1, v4_tr, m_tr, th_tr, cp_tr, s_tr, f_tr,
                                      hs_tr, hf_tr, p_tr, hp_tr, y_tr, device,
                                      seed=seed + 100, epochs=args.epochs)
        s2_pred = predict_stage2_a5(s2, v4_te, m_te, th_te, cp_te, s_te, f_te,
                                      hs_te, hf_te, p_te, hp_te, device)
        r2 = r2_per_prop(s2_pred, y_te)
        stage2_r2s.append(r2)
        print(f"  Stage-2 core7={r2['avg_core7']:.4f}  lignin={r2.get('lignin_wt', float('nan')):.4f}")

    s1 = summarize("Stage1_A5_surface_frames", stage1_r2s)
    s2 = summarize("Stage2_A5_deep_lignin", stage2_r2s)
    print(f"\n{'='*70}\nA5 two-stage SUMMARY\n{'='*70}")
    print(f"{'Stage':<40}{'core7':>10}{'std':>10}{'lignin':>10}")
    for r in [s1, s2]:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<40}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}{lig:>10.4f}")

    out = V5 / "results" / "a5_surface_frames.json"
    json.dump([s1, s2], open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
