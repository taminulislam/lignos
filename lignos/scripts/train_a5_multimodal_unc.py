"""A5.6 / A5.7 — Multimodal uncertainty-routed mean fusion.

Each modality (SMILES, ViT, DFT-Surface, COSMO-SAC) has its own (mean, logvar)
head. Predictions are combined by INVERSE-VARIANCE weighting:

    σ²_fused = 1 / Σ_k prec_k           prec_k = has_k · exp(-logvar_k)
    μ_fused  = σ²_fused · Σ_k μ_k · prec_k

Modality precision is masked by availability — a row without ViT frames gets
prec_ViT = 0 and that branch drops cleanly out of the fusion.

Training loss: Gaussian NLL on (μ_fused, logvar_fused, y, valid).

Start-state (inits):
  - SMILES branch: μ inherited from warm-started A2 (frozen), logvar head
    initialized to bias=0 → prec_SMILES ≈ 1 at init.
  - ViT / Surface / COSMO branches: mean heads zero-init (μ_k = 0), logvar
    heads initialized to bias=+5 → prec_k ≈ exp(-5) ≈ 0.007 at init. Their
    contribution is negligible at step 0; as training proceeds, the optimizer
    can lower their logvar to bring them into the fusion.

Optional (--tempered, A5.7): per-modality scalar τ_k ∈ R is learned,
logvar_k_eff = logvar_k + log τ_k. Absorbs systematic calibration bias per
modality.

Warm-start from lignos/checkpoints/a2/stage1_best.pt and freeze the
A2 backbone; only the new heads (and tau) train.

Expected (per A5.6 design brief):
  - core7 overall           : 0.85–0.86
  - conf-gated@50% core7    : ≥ 0.90 (closes the 0.90 gap)
  - Baran Task 2 OOD mean   : +0.3 to +0.5 when σ_SMILES grows on novel chem
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
import copy as _copy

CACHE = V5 / "data" / "LignoIL_A1"
A2_CKPT = V5 / "checkpoints" / "a2" / "stage1_best.pt"
VIT_BANK = V5 / "data" / "il_vit_bank.npz"
COSMO_BANK = V5 / "data" / "cosmo_sac_feat_bank.npz"
FRAME_DIM = 192
SURFACE_DIM = 256
COSMO_DIM = 20


class A5MultimodalUncHead(A2Head):
    """A2 backbone (frozen) + per-modality (mean, logvar) heads + inverse-variance fusion."""

    def __init__(self, nf, n_props=8, chemprop_dim=40,
                 surface_dim=SURFACE_DIM, frame_dim=FRAME_DIM, cosmo_dim=COSMO_DIM,
                 tempered=False):
        super().__init__(nf, n_props, chemprop_dim)
        self.tempered = tempered
        ctx_dim = nf + 5  # same context A2 uses: gated Morgan + thermo[:5]

        # SMILES branch — only logvar is new (mean inherited from A2 forward)
        self.logvar_smiles = self._mk_head(ctx_dim, n_props, final_bias=0.0)

        # Per-modality (mean, logvar) heads
        # Mean heads zero-init so at start the modality contributes μ=0 before
        # being weighted by its precision.
        self.mean_vit    = self._mk_head(frame_dim,   n_props, zero_final=True)
        self.logvar_vit  = self._mk_head(frame_dim,   n_props, final_bias=5.0)
        self.mean_surf   = self._mk_head(surface_dim, n_props, zero_final=True)
        self.logvar_surf = self._mk_head(surface_dim, n_props, final_bias=5.0)
        self.mean_cos    = self._mk_head(cosmo_dim,   n_props, zero_final=True)
        self.logvar_cos  = self._mk_head(cosmo_dim,   n_props, final_bias=5.0)

        # A5.7: learned per-modality scalar temperature (optional)
        if tempered:
            self.log_tau = nn.Parameter(torch.zeros(4))   # [SMILES, ViT, Surf, COSMO]
        else:
            self.register_parameter("log_tau", None)

    @staticmethod
    def _mk_head(in_dim, n_props, zero_final=False, final_bias=0.0):
        m = nn.Sequential(
            nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, n_props),
        )
        with torch.no_grad():
            if zero_final:
                m[-1].weight.zero_()
                m[-1].bias.zero_()
            else:
                m[-1].weight.mul_(0.01)
                m[-1].bias.fill_(final_bias)
        return m

    def _gated_ctx(self, i, t):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        return torch.cat([g, tmp], -1), tmp

    def forward(self, v, i, t, chemprop, vit, surface, cosmo,
                has_vit, has_surf, has_cos):
        # SMILES mean (frozen A2)
        mu_s = super().forward(v, i, t, chemprop)
        ctx, _ = self._gated_ctx(i, t)
        lv_s = self.logvar_smiles(ctx).clamp(-5, 5)

        # Per-modality branches
        mu_v = self.mean_vit(vit)
        lv_v = self.logvar_vit(vit).clamp(-5, 5)
        mu_p = self.mean_surf(surface)
        lv_p = self.logvar_surf(surface).clamp(-5, 5)
        mu_c = self.mean_cos(cosmo)
        lv_c = self.logvar_cos(cosmo).clamp(-5, 5)

        # Tempered variance (A5.7): per-modality scalar
        if self.log_tau is not None:
            lv_s = lv_s + self.log_tau[0]
            lv_v = lv_v + self.log_tau[1]
            lv_p = lv_p + self.log_tau[2]
            lv_c = lv_c + self.log_tau[3]

        # Availability masks broadcast to (B, n_props)
        def _mask(h):
            return h.float().unsqueeze(-1) if h.ndim == 1 else h.float()
        mv = _mask(has_vit); mp = _mask(has_surf); mc = _mask(has_cos)

        # Precisions (with masking for unavailable modalities)
        prec_s = torch.exp(-lv_s)                  # SMILES always available
        prec_v = torch.exp(-lv_v) * mv
        prec_p = torch.exp(-lv_p) * mp
        prec_c = torch.exp(-lv_c) * mc
        total = prec_s + prec_v + prec_p + prec_c + 1e-8

        mu_final = (mu_s * prec_s + mu_v * prec_v
                     + mu_p * prec_p + mu_c * prec_c) / total
        logvar_final = -torch.log(total)
        return mu_final, logvar_final


def _mask_gauss_nll(mu, lv, y, valid):
    # Tighter clamp (-5, 5) avoids NaN spirals when lv drives precision to
    # extremes (exp(10) ≈ 22k is already unstable in fp32 gradients).
    lv = lv.clamp(-5, 5)
    diff2 = (mu - y) ** 2
    nll = 0.5 * torch.exp(-lv) * diff2 + 0.5 * lv
    nll = nll * valid.float()
    return nll.sum() / valid.float().sum().clamp(min=1)


def _load_split(s):
    p_dft = CACHE / f"cached_{s}_dft.npz"
    p_std = CACHE / f"cached_{s}.npz"
    p = p_dft if p_dft.exists() else p_std
    print(f"[{s}] loading {p.name}")
    return {k: v for k, v in np.load(p, allow_pickle=True).items()}


def _canon(s):
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return s


def _assemble_bank(smiles, bank, dim):
    n = len(smiles)
    feats = np.zeros((n, dim), dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    for i, s in enumerate(smiles):
        cs = _canon(s)
        # Explicit None check — bank values are numpy arrays so `or` truthiness is illegal
        f = bank.get(cs)
        if f is None:
            f = bank.get(s)
        if f is not None:
            feats[i] = f
            mask[i] = 1.0
    return feats, mask


def _standardize(x, mask=None):
    """Z-score along axis 0, using `mask` to restrict which rows define mean/std."""
    if mask is not None and mask.sum() > 0:
        mu = x[mask > 0].mean(axis=0)
        sd = x[mask > 0].std(axis=0) + 1e-6
    else:
        mu = x.mean(axis=0)
        sd = x.std(axis=0) + 1e-6
    return (x - mu) / sd, mu, sd


def train_stage1(seed, v4, morg, th, cp, vit, surf, cos, hv, hp, hc, y,
                  device, tempered=False, epochs=300, patience=50):
    set_seed(seed)
    n_props = y.shape[1]
    m = A5MultimodalUncHead(morg.shape[1], n_props, chemprop_dim=cp.shape[1],
                               surface_dim=surf.shape[1], frame_dim=vit.shape[1],
                               cosmo_dim=cos.shape[1], tempered=tempered).to(device)

    if A2_CKPT.exists():
        ckpt = torch.load(A2_CKPT, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        miss, unex = m.load_state_dict(sd, strict=False)
        print(f"  warm-started A2 ckpt ({len(miss)} unmatched = new heads, {len(unex)} unused)")

    # Freeze A2 backbone; train all _smiles/_vit/_surf/_cos heads and log_tau
    _FREEZE_PREFIX_BLOCK = ("logvar_", "mean_vit", "mean_surf", "mean_cos", "log_tau")
    for name, p in m.named_parameters():
        if not any(name.startswith(pre) for pre in _FREEZE_PREFIX_BLOCK):
            p.requires_grad = False

    train_params = [p for p in m.parameters() if p.requires_grad]
    print(f"  trainable params: {sum(p.numel() for p in train_params)}"
          f"  (tempered={tempered})")

    opt = AdamW(train_params, lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    ts = {k: torch.from_numpy(x).to(device) for k, x in dict(
        v=v4, i=morg, t=th, cp=cp, vit=vit, surf=surf, cos=cos,
        hv=hv, hp=hp, hc=hc, y=y).items()}
    valid = ~torch.isnan(ts["y"]); yf = torch.nan_to_num(ts["y"], 0.0)

    ds = TensorDataset(*[ts[k].cpu() for k in
                          ("v","i","t","cp","vit","surf","cos","hv","hp","hc")],
                        yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            vb, ib, tb, cpb, vitb, surb, cosb, hvb, hpb, hcb, yb, vm = batch
            mu, lv = m(vb, ib, tb, cpb, vitb, surb, cosb, hvb, hpb, hcb)
            loss = _mask_gauss_nll(mu, lv, yb, vm)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            mu, lv = m(ts["v"], ts["i"], ts["t"], ts["cp"],
                        ts["vit"], ts["surf"], ts["cos"],
                        ts["hv"], ts["hp"], ts["hc"])
            tl = _mask_gauss_nll(mu, lv, yf, valid).item()
        if np.isfinite(tl) and tl < best:
            best = tl; state = {k: v.clone() for k, v in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval()
    return m


def predict_stage1(m, v4, morg, th, cp, vit, surf, cos, hv, hp, hc, device):
    with torch.no_grad():
        mu, lv = m(*(torch.from_numpy(x).to(device) for x in
                      (v4, morg, th, cp, vit, surf, cos, hv, hp, hc)))
    return mu.cpu().numpy(), lv.cpu().numpy()


# --------------------------------------------------------------------------
# Stage-2 — custom wrapper to match A5MultimodalUncHead's forward signature.
# --------------------------------------------------------------------------
class A5MMStage2Wrapper(nn.Module):
    """Freeze a multimodal Stage-1 backbone; attach a deep lignin head on top
    of the fused mean prediction (column 7)."""
    def __init__(self, stage1_model: A5MultimodalUncHead, physchem_dim=12):
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

    def forward(self, v, i, t, chemprop, vit, surf, cos, hv, hs, hc, phys, has_phys):
        base_mu, _ = self.backbone(v, i, t, chemprop, vit, surf, cos, hv, hs, hc)
        tmp = t[:, :5]
        g = i * self.backbone.gate(tmp)
        hp = has_phys.float().unsqueeze(-1) if has_phys.ndim == 1 else has_phys.float()
        ctx = torch.cat([g, tmp, phys, hp], -1)
        res_lignin = self.deep_lignin(ctx).squeeze(-1)
        out = base_mu.clone()
        out[:, 7] = v[:, 7] + torch.sigmoid(self.alpha_lignin) * res_lignin
        return out


def train_stage2_mm(stage1_model, v4, morg, th, cp, vit, surf, cos,
                     hv, hs, hc, phys, hp, y, device, seed,
                     epochs=300, patience=50):
    set_seed(seed)
    m = A5MMStage2Wrapper(_copy.deepcopy(stage1_model)).to(device)
    train_params = [p for p in m.parameters() if p.requires_grad]
    opt = AdamW([{"params": m.deep_lignin.parameters(), "weight_decay": 1e-2},
                  {"params": [m.alpha_lignin], "weight_decay": 0.0}], lr=1e-3)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    ts = {k: torch.from_numpy(x).to(device) for k, x in dict(
        v=v4, i=morg, t=th, cp=cp, vit=vit, surf=surf, cos=cos,
        hv=hv, hs=hs, hc=hc, p=phys, hp=hp, y=y).items()}
    ds = TensorDataset(*[ts[k].cpu() for k in
                          ("v","i","t","cp","vit","surf","cos",
                           "hv","hs","hc","p","hp","y")])
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            vb, ib, tb, cpb, vitb, surb, cosb, hvb, hsb, hcb, pb, hpb, yb = batch
            pred = m(vb, ib, tb, cpb, vitb, surb, cosb, hvb, hsb, hcb, pb, hpb)
            lg = ~torch.isnan(yb[:, 7])
            if lg.sum() == 0: continue
            loss = ((pred[lg, 7] - yb[lg, 7].nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(ts["v"], ts["i"], ts["t"], ts["cp"],
                      ts["vit"], ts["surf"], ts["cos"],
                      ts["hv"], ts["hs"], ts["hc"], ts["p"], ts["hp"])
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


def predict_stage2_mm(m, v4, morg, th, cp, vit, surf, cos, hv, hs, hc, phys, hp, device):
    with torch.no_grad():
        return m(*(torch.from_numpy(x).to(device) for x in
                    (v4, morg, th, cp, vit, surf, cos, hv, hs, hc, phys, hp))).cpu().numpy()


def confidence_gated_r2(pred, y, logvar, quantile=0.5):
    sigma = np.exp(0.5 * logvar).mean(axis=-1)
    thr = np.quantile(sigma, quantile)
    keep = sigma <= thr
    r = {}
    for i, p in enumerate(PROPS):
        valid = ~np.isnan(y[:, i]) & keep
        if valid.sum() < 2: continue
        yk = y[valid, i]; pk = pred[valid, i]
        ss_res = ((yk - pk) ** 2).sum()
        ss_tot = ((yk - yk.mean()) ** 2).sum() + 1e-12
        r[p] = 1.0 - ss_res / ss_tot
    r["avg_core7"] = float(np.mean([r[p] for p in PROPS[:7] if p in r and np.isfinite(r[p])]))
    r["keep_frac"] = float(keep.mean())
    r["quantile"] = quantile
    return r


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
    ap.add_argument("--tempered", action="store_true",
                    help="A5.7: learn per-modality scalar temperature τ_k.")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "tempered" if args.tempered else "vanilla"
    print(f"Device: {device}  tempered={args.tempered}")
    tr, va, te = _load_split("train"), _load_split("val"), _load_split("test")

    pca_m = PCA(40).fit(tr["morgan_fp"])
    m_tr, m_va, m_te = [pca_m.transform(x["morgan_fp"]).astype(np.float32)
                         for x in (tr, va, te)]
    cp_tr, cp_te = build_chemprop_40d(tr["chemprop_fp"], te["chemprop_fp"])
    _, cp_va = build_chemprop_40d(tr["chemprop_fp"], va["chemprop_fp"])
    p_tr, p_te = preprocess_physchem(tr["physchem_feat"], tr["has_physchem"],
                                      te["physchem_feat"], te["has_physchem"])
    hp_tr = tr["has_physchem"].astype(np.float32)
    hp_te = te["has_physchem"].astype(np.float32)

    # Surface: already in cache (surface_fp field)
    surf_tr = tr["surface_fp"].astype(np.float32)
    surf_va = va["surface_fp"].astype(np.float32)
    surf_te = te["surface_fp"].astype(np.float32)
    hs_tr = (surf_tr != 0).any(axis=1).astype(np.float32)
    hs_va = (surf_va != 0).any(axis=1).astype(np.float32)
    hs_te = (surf_te != 0).any(axis=1).astype(np.float32)

    # ViT bank
    vit_bank = {s: f for s, f in zip(*[np.load(VIT_BANK, allow_pickle=True)[k]
                                          for k in ("smiles", "vit_feat")])}
    vit_tr, hv_tr = _assemble_bank(tr["smiles"], vit_bank, FRAME_DIM)
    vit_va, hv_va = _assemble_bank(va["smiles"], vit_bank, FRAME_DIM)
    vit_te, hv_te = _assemble_bank(te["smiles"], vit_bank, FRAME_DIM)

    # COSMO-SAC bank (20-D σ-profile moments)
    cos_bank_raw = np.load(COSMO_BANK, allow_pickle=True)
    cos_bank = dict(zip(cos_bank_raw["smiles"], cos_bank_raw["cosmo_feat"]))
    cos_tr, hc_tr = _assemble_bank(tr["smiles"], cos_bank, COSMO_DIM)
    cos_va, hc_va = _assemble_bank(va["smiles"], cos_bank, COSMO_DIM)
    cos_te, hc_te = _assemble_bank(te["smiles"], cos_bank, COSMO_DIM)

    # Z-score ViT / surface / cosmo using train-covered rows
    vit_tr, mu_v, sd_v = _standardize(vit_tr, hv_tr)
    vit_va = ((vit_va - mu_v) / sd_v).astype(np.float32) * hv_va[:, None]
    vit_te = ((vit_te - mu_v) / sd_v).astype(np.float32) * hv_te[:, None]
    surf_tr, mu_p, sd_p = _standardize(surf_tr, hs_tr)
    surf_va = ((surf_va - mu_p) / sd_p).astype(np.float32) * hs_va[:, None]
    surf_te = ((surf_te - mu_p) / sd_p).astype(np.float32) * hs_te[:, None]
    cos_tr, mu_c, sd_c = _standardize(cos_tr, hc_tr)
    cos_va = ((cos_va - mu_c) / sd_c).astype(np.float32) * hc_va[:, None]
    cos_te = ((cos_te - mu_c) / sd_c).astype(np.float32) * hc_te[:, None]

    print(f"ViT  coverage: tr {hv_tr.mean():.1%}  va {hv_va.mean():.1%}  te {hv_te.mean():.1%}")
    print(f"Surf coverage: tr {hs_tr.mean():.1%}  va {hs_va.mean():.1%}  te {hs_te.mean():.1%}")
    print(f"COSMO coverage: tr {hc_tr.mean():.1%}  va {hc_va.mean():.1%}  te {hc_te.mean():.1%}")

    v4_tr, v4_va, v4_te = v4_base(tr), v4_base(va), v4_base(te)
    y_tr, y_te = tr["targets"].astype(np.float32), te["targets"].astype(np.float32)
    th_tr, th_va, th_te = tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]

    s1_r2s, s1_gated50, s1_gated25, s2_r2s = [], [], [], []
    for seed in range(args.n_seeds):
        print(f"\n[seed {seed}] Stage-1 (A5.{'7' if args.tempered else '6'} multimodal NLL fusion)...")
        s1 = train_stage1(seed, v4_tr, m_tr, th_tr, cp_tr,
                           vit_tr, surf_tr, cos_tr,
                           hv_tr, hs_tr, hc_tr, y_tr,
                           device, tempered=args.tempered, epochs=args.epochs)
        mu_te, lv_te = predict_stage1(s1, v4_te, m_te, th_te, cp_te,
                                        vit_te, surf_te, cos_te,
                                        hv_te, hs_te, hc_te, device)
        r = r2_per_prop(mu_te, y_te)
        g50 = confidence_gated_r2(mu_te, y_te, lv_te, quantile=0.5)
        g25 = confidence_gated_r2(mu_te, y_te, lv_te, quantile=0.25)
        s1_r2s.append(r); s1_gated50.append(g50); s1_gated25.append(g25)
        tau_str = ""
        if s1.log_tau is not None:
            tau_str = f"  log_tau={s1.log_tau.detach().cpu().numpy().round(2).tolist()}"
        print(f"  Stage-1 core7 ALL={r['avg_core7']:.4f}  gated@50%={g50['avg_core7']:.4f}  "
              f"gated@25%={g25['avg_core7']:.4f}  mean σ={np.exp(0.5*lv_te).mean():.3f}{tau_str}")

        print(f"[seed {seed}] Stage-2 (hardfreeze + deep lignin + physchem)...")
        s2 = train_stage2_mm(s1, v4_tr, m_tr, th_tr, cp_tr,
                              vit_tr, surf_tr, cos_tr,
                              hv_tr, hs_tr, hc_tr,
                              p_tr, hp_tr, y_tr,
                              device, seed=seed + 100, epochs=args.epochs)
        s2_pred = predict_stage2_mm(s2, v4_te, m_te, th_te, cp_te,
                                      vit_te, surf_te, cos_te,
                                      hv_te, hs_te, hc_te,
                                      p_te, hp_te, device)
        r2 = r2_per_prop(s2_pred, y_te)
        s2_r2s.append(r2)
        print(f"  Stage-2 core7={r2['avg_core7']:.4f}  lignin={r2.get('lignin_wt', float('nan')):.4f}")

    s1 = summarize(f"Stage1_A5_multimodal_{tag}", s1_r2s)
    sg50 = summarize(f"Stage1_A5_multimodal_gated50_{tag}", s1_gated50)
    sg25 = summarize(f"Stage1_A5_multimodal_gated25_{tag}", s1_gated25)
    s2 = summarize(f"Stage2_A5_multimodal_deep_lignin_{tag}", s2_r2s)
    print(f"\n{'='*70}\nA5.{'7' if args.tempered else '6'} MULTIMODAL UNC FUSION SUMMARY ({tag})\n{'='*70}")
    print(f"{'Stage':<50}{'core7':>10}{'std':>10}{'lignin':>10}")
    for r in [s1, sg50, sg25, s2]:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<50}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}{lig:>10.4f}")
    out = V5 / "results" / f"a5_multimodal_{tag}.json"
    json.dump([s1, sg50, sg25, s2], open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
