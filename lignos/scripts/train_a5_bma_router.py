"""A5.9 — Bayesian Model Averaging with V4-style anchored router.

Three specialists (frozen A2 backbone + specialty branches) predict (μ_k, σ²_k)
per target; a tiny per-molecule router learns a residual correction over the
inverse-variance BMA weights. This is V4's PerMoleculeRouter pattern
(cosmobridge_v4_router.py) generalized from binary sigmoid to K-way softmax,
and ANCHORED to BMA so the router starts from a thermodynamically-principled
prior and only learns corrections on top.

Specialists (trained jointly in Stage-A, then frozen for Stage-B routing):
  A : A2 (SMILES + ChemProp + thermo) + logvar_A
  B : A2 + Surface(256→32) + Frame(192→32) gated residual + logvar_B
  C : A2 + COSMO-SAC(20→32) gated residual + logvar_C

Stage-A joint loss (specialists trained together):
  L_A = gauss_NLL(μ_A, logvar_A, y)
  L_B = gauss_NLL(μ_B, logvar_B, y)
  L_C = gauss_NLL(μ_C, logvar_C, y)
  loss_A = L_A + L_B + L_C

  A2 backbone frozen; each specialty branch + logvar head trains.

Stage-B router training (specialists frozen):
  w_prior_k(x) = 1/σ²_k  (row-level, per-prop)
  logits_k(x)  = log(w_prior_k) + small_MLP(chemprop, surface, thermo)_k
  w_k(x)       = softmax_k(logits_k)
  μ_fused       = Σ_k w_k · μ_k
  σ²_fused      = 1 / Σ_k (1 / σ²_k)        (still inverse-variance, not learned)
  loss_B        = gauss_NLL(μ_fused, σ²_fused, y)

Router is ~6k params (MLP [321→64→24]); no risk of overfitting the 5189-row cache.
"""
from __future__ import annotations
import argparse, copy, json, sys
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
from train_a2_two_stage import (
    A2Head, build_chemprop_40d, preprocess_physchem, v4_base,
)

CACHE = V5 / "data" / "LignoIL_A1"
A2_CKPT = V5 / "checkpoints" / "a2" / "stage1_best.pt"
VIT_BANK = V5 / "data" / "il_vit_bank.npz"
COSMO_BANK = V5 / "data" / "cosmo_sac_feat_bank.npz"

FRAME_DIM = 192
SURFACE_DIM = 256
COSMO_DIM = 20
K_SPECIALISTS = 3
LV_CLAMP = (-5.0, 5.0)


class A5_BMA_Model(A2Head):
    """Three specialists + anchored router, all in one module for easy training."""

    def __init__(self, nf, n_props=8, chemprop_dim=40,
                 surface_dim=SURFACE_DIM, frame_dim=FRAME_DIM, cosmo_dim=COSMO_DIM):
        super().__init__(nf, n_props, chemprop_dim)
        self.n_props = n_props
        ctx_dim = nf + 5

        # ──── Specialist A: A2 + logvar
        self.logvar_A = self._mk(ctx_dim, n_props, bias=0.0)

        # ──── Specialist B: A2 + Surface + Frame branches (like A5_sf) + logvar
        self.surf_proj_B = nn.Sequential(nn.Linear(surface_dim, 32), nn.GELU(), nn.Linear(32, 32))
        self.frame_proj_B = nn.Sequential(nn.Linear(frame_dim, 32), nn.GELU(), nn.Linear(32, 32))
        for proj in (self.surf_proj_B, self.frame_proj_B):
            with torch.no_grad():
                proj[-1].weight.mul_(0.01); proj[-1].bias.zero_()
        self.delta_B_head = nn.Sequential(nn.Linear(ctx_dim + 64, 32), nn.GELU(), nn.Linear(32, n_props))
        with torch.no_grad():
            self.delta_B_head[-1].weight.zero_(); self.delta_B_head[-1].bias.zero_()
        self.gate_B = nn.Parameter(torch.full((n_props,), -3.0))
        self.logvar_B = self._mk(ctx_dim + 64, n_props, bias=0.0)

        # ──── Specialist C: A2 + COSMO-SAC branch (like A5_cosmo) + logvar
        self.cosmo_proj_C = nn.Sequential(nn.Linear(cosmo_dim, 32), nn.GELU(), nn.Linear(32, 32))
        with torch.no_grad():
            self.cosmo_proj_C[-1].weight.mul_(0.01); self.cosmo_proj_C[-1].bias.zero_()
        self.delta_C_head = nn.Sequential(nn.Linear(ctx_dim + 32, 32), nn.GELU(), nn.Linear(32, n_props))
        with torch.no_grad():
            self.delta_C_head[-1].weight.zero_(); self.delta_C_head[-1].bias.zero_()
        self.gate_C = nn.Parameter(torch.full((n_props,), -3.0))
        self.logvar_C = self._mk(ctx_dim + 32, n_props, bias=0.0)

        # ──── Router (V4 PerMoleculeRouter, generalized to K=3)
        # Input: [chemprop(40) ⊕ surface(256) ⊕ thermo(25)] = 321D
        router_in = chemprop_dim + surface_dim + 25
        self.router = nn.Sequential(
            nn.Linear(router_in, 64), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(64),
            nn.Linear(64, K_SPECIALISTS * n_props),
        )
        # Zero-init final → softmax(zeros) = uniform; with BMA anchor, the final
        # router logits == log(1/σ²_k) at step 0, i.e., pure inverse-variance.
        with torch.no_grad():
            self.router[-1].weight.zero_(); self.router[-1].bias.zero_()

    @staticmethod
    def _mk(in_dim, n_props, bias=0.0):
        m = nn.Sequential(nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, n_props))
        with torch.no_grad():
            m[-1].weight.mul_(0.01); m[-1].bias.fill_(bias)
        return m

    def _ctx(self, i, t):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        return torch.cat([g, tmp], -1), tmp

    def _specialists(self, v, i, t, chemprop, surface, vit, cos,
                      has_surf, has_vit, has_cos):
        ctx, tmp = self._ctx(i, t)
        mu_A_base = super().forward(v, i, t, chemprop)  # A2 prediction (frozen)
        lv_A = self.logvar_A(ctx).clamp(*LV_CLAMP)

        # Specialist B
        hv = (has_vit.float().unsqueeze(-1) if has_vit.ndim == 1 else has_vit.float())
        hs = (has_surf.float().unsqueeze(-1) if has_surf.ndim == 1 else has_surf.float())
        p_surf = self.surf_proj_B(surface) * hs
        p_frame = self.frame_proj_B(vit) * hv
        B_ctx = torch.cat([ctx, p_surf, p_frame], -1)
        delta_B = self.delta_B_head(B_ctx)
        mu_B = mu_A_base + torch.sigmoid(self.gate_B) * delta_B
        lv_B = self.logvar_B(B_ctx).clamp(*LV_CLAMP)

        # Specialist C
        hc = (has_cos.float().unsqueeze(-1) if has_cos.ndim == 1 else has_cos.float())
        p_cos = self.cosmo_proj_C(cos) * hc
        C_ctx = torch.cat([ctx, p_cos], -1)
        delta_C = self.delta_C_head(C_ctx)
        mu_C = mu_A_base + torch.sigmoid(self.gate_C) * delta_C
        lv_C = self.logvar_C(C_ctx).clamp(*LV_CLAMP)

        return (mu_A_base, lv_A), (mu_B, lv_B), (mu_C, lv_C)

    def _router_weights(self, chemprop, surface, thermo, lv_stack):
        """V4-style anchored router. lv_stack: (B, K, P)."""
        router_in = torch.cat([chemprop, surface, thermo], -1)
        corr = self.router(router_in).view(-1, K_SPECIALISTS, self.n_props)
        # Anchor prior: log(1/σ²_k) = -logvar_k → softmax gives BMA weights
        anchor = -lv_stack                           # (B, K, P)
        logits = anchor + corr                       # (B, K, P)
        return F.softmax(logits, dim=1)

    def forward(self, v, i, t, chemprop, surface, vit, cos,
                has_surf, has_vit, has_cos,
                return_components=False):
        (muA, lvA), (muB, lvB), (muC, lvC) = self._specialists(
            v, i, t, chemprop, surface, vit, cos, has_surf, has_vit, has_cos)

        mu_stack = torch.stack([muA, muB, muC], dim=1)   # (B, K, P)
        lv_stack = torch.stack([lvA, lvB, lvC], dim=1)   # (B, K, P)
        # Router uses FULL thermo_feat(25D) — matches V4 PerMoleculeRouter input
        w = self._router_weights(chemprop, surface, t, lv_stack)   # (B, K, P)
        mu_fused = (w * mu_stack).sum(dim=1)                           # (B, P)
        # σ²_fused = 1 / Σ_k 1/σ²_k  (unchanged; inverse-variance)
        prec_sum = torch.exp(-lv_stack).sum(dim=1) + 1e-8
        lv_fused = -torch.log(prec_sum)
        if return_components:
            return mu_fused, lv_fused, mu_stack, lv_stack, w
        return mu_fused, lv_fused


def _gauss_nll(mu, lv, y, valid):
    lv = lv.clamp(*LV_CLAMP)
    nll = 0.5 * torch.exp(-lv) * (mu - y) ** 2 + 0.5 * lv
    return (nll * valid.float()).sum() / valid.float().sum().clamp(min=1)


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
        f = bank.get(cs)
        if f is None:
            f = bank.get(s)
        if f is not None:
            feats[i] = f
            mask[i] = 1.0
    return feats, mask


def _standardize(x, mask=None):
    if mask is not None and mask.sum() > 0:
        mu = x[mask > 0].mean(axis=0); sd = x[mask > 0].std(axis=0) + 1e-6
    else:
        mu = x.mean(axis=0); sd = x.std(axis=0) + 1e-6
    return (x - mu) / sd, mu, sd


def train_joint(seed, v4, morg, th, cp, surf, vit, cos,
                hs, hv, hc, y, device, epochs=300, patience=50,
                lambda_spec=1.0):
    """Stage-A: joint training of 3 specialists + router on shared A2 backbone.

    Loss = NLL(μ_fused, lv_fused) + λ · [NLL(μ_A,lv_A) + NLL(μ_B,lv_B) + NLL(μ_C,lv_C)]

    Per-specialist auxiliary NLL keeps each specialist individually calibrated
    so the BMA anchor makes sense. λ=1.0 weights them equally with the fusion.
    """
    set_seed(seed)
    n_props = y.shape[1]
    m = A5_BMA_Model(morg.shape[1], n_props, chemprop_dim=cp.shape[1],
                       surface_dim=surf.shape[1], frame_dim=vit.shape[1],
                       cosmo_dim=cos.shape[1]).to(device)

    if A2_CKPT.exists():
        ckpt = torch.load(A2_CKPT, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        miss, unex = m.load_state_dict(sd, strict=False)
        print(f"  warm-started A2 ckpt ({len(miss)} unmatched specialist/router heads)")

    # Freeze A2 backbone; train all new specialist heads + router
    _A2_PARAMS = {"gate", "heads", "alphas", "cp_proj", "cp_heads", "cp_gate"}
    for name, p in m.named_parameters():
        root = name.split(".")[0]
        if root in _A2_PARAMS:
            p.requires_grad = False

    train_params = [p for p in m.parameters() if p.requires_grad]
    print(f"  trainable params: {sum(p.numel() for p in train_params)} "
          f"(of {sum(p.numel() for p in m.parameters())})")

    opt = AdamW(train_params, lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    ts = {k: torch.from_numpy(x).to(device) for k, x in dict(
        v=v4, i=morg, t=th, cp=cp, surf=surf, vit=vit, cos=cos,
        hs=hs, hv=hv, hc=hc, y=y).items()}
    valid = ~torch.isnan(ts["y"]); yf = torch.nan_to_num(ts["y"], 0.0)
    ds = TensorDataset(*[ts[k].cpu() for k in
                          ("v","i","t","cp","surf","vit","cos","hs","hv","hc")],
                        yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            vb, ib, tb, cpb, surb, vitb, cosb, hsb, hvb, hcb, yb, vm = batch
            mu_f, lv_f, mu_s, lv_s, w = m(vb, ib, tb, cpb, surb, vitb, cosb,
                                            hsb, hvb, hcb, return_components=True)
            L_fuse = _gauss_nll(mu_f, lv_f, yb, vm)
            L_spec = sum(_gauss_nll(mu_s[:, k], lv_s[:, k], yb, vm) for k in range(K_SPECIALISTS))
            loss = L_fuse + lambda_spec * L_spec / K_SPECIALISTS
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            mu_f, lv_f, _, _, _ = m(ts["v"], ts["i"], ts["t"], ts["cp"],
                                      ts["surf"], ts["vit"], ts["cos"],
                                      ts["hs"], ts["hv"], ts["hc"],
                                      return_components=True)
            tl = _gauss_nll(mu_f, lv_f, yf, valid).item()
        if np.isfinite(tl) and tl < best:
            best = tl; state = {k: v.clone() for k, v in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval()
    return m


def predict(m, v4, morg, th, cp, surf, vit, cos, hs, hv, hc, device):
    with torch.no_grad():
        mu_f, lv_f, mu_s, lv_s, w = m(
            *(torch.from_numpy(x).to(device) for x in
              (v4, morg, th, cp, surf, vit, cos, hs, hv, hc)),
            return_components=True)
    return (mu_f.cpu().numpy(), lv_f.cpu().numpy(),
            mu_s.cpu().numpy(), lv_s.cpu().numpy(), w.cpu().numpy())


def conf_gated_r2(pred, y, logvar, quantile=0.5):
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
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tr, va, te = _load_split("train"), _load_split("val"), _load_split("test")

    pca_m = PCA(40).fit(tr["morgan_fp"])
    m_tr, m_va, m_te = [pca_m.transform(x["morgan_fp"]).astype(np.float32)
                         for x in (tr, va, te)]
    cp_tr, cp_te = build_chemprop_40d(tr["chemprop_fp"], te["chemprop_fp"])
    _, cp_va = build_chemprop_40d(tr["chemprop_fp"], va["chemprop_fp"])

    surf_tr = tr["surface_fp"].astype(np.float32)
    surf_va = va["surface_fp"].astype(np.float32)
    surf_te = te["surface_fp"].astype(np.float32)
    hs_tr = (surf_tr != 0).any(axis=1).astype(np.float32)
    hs_va = (surf_va != 0).any(axis=1).astype(np.float32)
    hs_te = (surf_te != 0).any(axis=1).astype(np.float32)
    surf_tr, mu_p, sd_p = _standardize(surf_tr, hs_tr)
    surf_va = ((surf_va - mu_p) / sd_p).astype(np.float32) * hs_va[:, None]
    surf_te = ((surf_te - mu_p) / sd_p).astype(np.float32) * hs_te[:, None]

    vit_bank = {s: f for s, f in zip(*[np.load(VIT_BANK, allow_pickle=True)[k]
                                          for k in ("smiles", "vit_feat")])}
    vit_tr, hv_tr = _assemble_bank(tr["smiles"], vit_bank, FRAME_DIM)
    vit_va, hv_va = _assemble_bank(va["smiles"], vit_bank, FRAME_DIM)
    vit_te, hv_te = _assemble_bank(te["smiles"], vit_bank, FRAME_DIM)
    vit_tr, mu_v, sd_v = _standardize(vit_tr, hv_tr)
    vit_va = ((vit_va - mu_v) / sd_v).astype(np.float32) * hv_va[:, None]
    vit_te = ((vit_te - mu_v) / sd_v).astype(np.float32) * hv_te[:, None]

    cos_bank = dict(zip(*[np.load(COSMO_BANK, allow_pickle=True)[k]
                             for k in ("smiles", "cosmo_feat")]))
    cos_tr, hc_tr = _assemble_bank(tr["smiles"], cos_bank, COSMO_DIM)
    cos_va, hc_va = _assemble_bank(va["smiles"], cos_bank, COSMO_DIM)
    cos_te, hc_te = _assemble_bank(te["smiles"], cos_bank, COSMO_DIM)
    cos_tr, mu_c, sd_c = _standardize(cos_tr, hc_tr)
    cos_va = ((cos_va - mu_c) / sd_c).astype(np.float32) * hc_va[:, None]
    cos_te = ((cos_te - mu_c) / sd_c).astype(np.float32) * hc_te[:, None]

    print(f"Surf coverage: tr {hs_tr.mean():.1%} / te {hs_te.mean():.1%}  "
          f"ViT: tr {hv_tr.mean():.1%} / te {hv_te.mean():.1%}  "
          f"COSMO: tr {hc_tr.mean():.1%} / te {hc_te.mean():.1%}")

    v4_tr, v4_va, v4_te = v4_base(tr), v4_base(va), v4_base(te)
    y_tr, y_te = tr["targets"].astype(np.float32), te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]

    fuse_r2, g50_r2, g25_r2 = [], [], []
    specA_r2, specB_r2, specC_r2 = [], [], []
    for seed in range(args.n_seeds):
        print(f"\n[seed {seed}] Stage-A joint training (A2 frozen + 3 specialists + router)...")
        m = train_joint(seed, v4_tr, m_tr, th_tr, cp_tr,
                         surf_tr, vit_tr, cos_tr,
                         hs_tr, hv_tr, hc_tr, y_tr,
                         device, epochs=args.epochs)

        mu_f, lv_f, mu_s, lv_s, w_s = predict(m, v4_te, m_te, th_te, cp_te,
                                                surf_te, vit_te, cos_te,
                                                hs_te, hv_te, hc_te, device)
        r_f = r2_per_prop(mu_f, y_te)
        r_A = r2_per_prop(mu_s[:, 0], y_te)
        r_B = r2_per_prop(mu_s[:, 1], y_te)
        r_C = r2_per_prop(mu_s[:, 2], y_te)
        g50 = conf_gated_r2(mu_f, y_te, lv_f, quantile=0.5)
        g25 = conf_gated_r2(mu_f, y_te, lv_f, quantile=0.25)
        fuse_r2.append(r_f); specA_r2.append(r_A); specB_r2.append(r_B); specC_r2.append(r_C)
        g50_r2.append(g50); g25_r2.append(g25)
        w_avg = w_s.mean(axis=(0, 2))
        print(f"  Specialists core7: A={r_A['avg_core7']:.4f}  B={r_B['avg_core7']:.4f}  C={r_C['avg_core7']:.4f}")
        print(f"  Fused core7     : {r_f['avg_core7']:.4f}  (std across seeds pending)")
        print(f"  Gated@50% core7 : {g50['avg_core7']:.4f}   gated@25% core7: {g25['avg_core7']:.4f}")
        print(f"  Router weights  : A={w_avg[0]:.2f}  B={w_avg[1]:.2f}  C={w_avg[2]:.2f}")

    def _sum(name, rs): return summarize(name, rs)
    out = {
        "fused":            _sum("BMA_fused", fuse_r2),
        "fused_gated_50":   _sum("BMA_fused_gated50", g50_r2),
        "fused_gated_25":   _sum("BMA_fused_gated25", g25_r2),
        "specialist_A":     _sum("Specialist_A", specA_r2),
        "specialist_B":     _sum("Specialist_B", specB_r2),
        "specialist_C":     _sum("Specialist_C", specC_r2),
    }

    print(f"\n{'='*72}\nA5.9 BMA with V4-Anchored Router SUMMARY\n{'='*72}")
    print(f"{'Arm':<35}{'core7':>10}{'std':>10}")
    for k, v in out.items():
        print(f"{v['name']:<35}{v['avg_r2_core7']:>10.4f}{v['std_r2_core7']:>10.4f}")
    outf = V5 / "results" / "a5_bma_router.json"
    json.dump(out, open(outf, "w"), indent=2)
    print(f"\nSaved: {outf}")


if __name__ == "__main__":
    main()
