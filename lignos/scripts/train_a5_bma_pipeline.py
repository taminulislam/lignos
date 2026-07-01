"""A5.9 — Refactored sequential BMA pipeline (fixes joint-NLL runaway).

Workflow (matches V4 PerMoleculeRouter training pattern):

  Stage 1: Train Specialist A independently with NLL. Save best ckpt.
  Stage 2: Train Specialist B independently with NLL. Save best ckpt.
  Stage 3: Train Specialist C independently with NLL. Save best ckpt.
  Stage 4: Freeze all 3. Train ONLY the anchored router + fused NLL.

Key differences from the broken monolithic run (job 17774953):
  1. Each specialist is trained alone — no inter-specialist gradient coupling,
     so Specialist C can't be dragged into the runaway loop where its logvar
     clamps and μ_C drifts unbounded.
  2. Logvar heads init bias = +2 (σ² ≈ 7 at init, UNCONFIDENT). Must earn
     calibration. Prevents the "step-1 overconfidence → runaway" pathology.
  3. MSE warmup for the first 100 epochs, NLL afterwards — mean heads
     stabilize before logvar starts shaping gradients (standard VAE trick).
  4. Router trains on PRE-COMPUTED specialist predictions, not on end-to-end
     gradient — exactly how V4 did it.

Checkpoints are reusable: if `checkpoints/a5_bma/specialist_{A,B,C}.pt` exist,
we skip re-training and move to the router. Makes this script idempotent —
can resume after OOM, partial completion, or iterate the router without
retraining specialists.

Outputs:
  lignos/checkpoints/a5_bma/specialist_{A,B,C}.pt
  lignos/results/a5_bma_pipeline.json
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
BMA_DIR = V5 / "checkpoints" / "a5_bma"
BMA_DIR.mkdir(parents=True, exist_ok=True)
VIT_BANK = V5 / "data" / "il_vit_bank.npz"
COSMO_BANK = V5 / "data" / "cosmo_sac_feat_bank.npz"

FRAME_DIM = 192
SURFACE_DIM = 256
COSMO_DIM = 20
K_SPECIALISTS = 3
LV_CLAMP = (-3.0, 3.0)            # Tighter than A5.9's (-5, 5): precision ≤ 20
WARMUP_EPOCHS = 100                # MSE-only phase before NLL kicks in


# ==========================================================================
# Specialist model (flexible: A / B / C selected by --specialist)
# ==========================================================================
class A5_BMA_Specialist(A2Head):
    """A2 (frozen) + optional specialty branch + per-prop logvar head.

    Kind:
      A: no extra branch (SMILES only) — just A2 + logvar
      B: + Surface(256→32) + Frame(192→32) gated residual
      C: + COSMO-SAC(20→32) gated residual
    """

    def __init__(self, kind: str, nf, n_props=8, chemprop_dim=40,
                 surface_dim=SURFACE_DIM, frame_dim=FRAME_DIM, cosmo_dim=COSMO_DIM):
        super().__init__(nf, n_props, chemprop_dim)
        assert kind in ("A", "B", "C")
        self.kind = kind
        ctx_dim = nf + 5

        if kind == "B":
            self.surf_proj = nn.Sequential(nn.Linear(surface_dim, 32), nn.GELU(), nn.Linear(32, 32))
            self.frame_proj = nn.Sequential(nn.Linear(frame_dim, 32), nn.GELU(), nn.Linear(32, 32))
            for proj in (self.surf_proj, self.frame_proj):
                with torch.no_grad():
                    proj[-1].weight.mul_(0.01); proj[-1].bias.zero_()
            self.delta_head = self._mk(ctx_dim + 64, n_props, final_zero=True)
            self.gate_b = nn.Parameter(torch.full((n_props,), -3.0))
            self.logvar_head = self._mk(ctx_dim + 64, n_props, final_bias=2.0)
        elif kind == "C":
            self.cosmo_proj = nn.Sequential(nn.Linear(cosmo_dim, 32), nn.GELU(), nn.Linear(32, 32))
            with torch.no_grad():
                self.cosmo_proj[-1].weight.mul_(0.01); self.cosmo_proj[-1].bias.zero_()
            self.delta_head = self._mk(ctx_dim + 32, n_props, final_zero=True)
            self.gate_b = nn.Parameter(torch.full((n_props,), -3.0))
            self.logvar_head = self._mk(ctx_dim + 32, n_props, final_bias=2.0)
        else:  # A
            self.logvar_head = self._mk(ctx_dim, n_props, final_bias=2.0)

    @staticmethod
    def _mk(in_dim, n_props, final_zero=False, final_bias=0.0):
        m = nn.Sequential(nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, n_props))
        with torch.no_grad():
            if final_zero:
                m[-1].weight.zero_(); m[-1].bias.zero_()
            else:
                m[-1].weight.mul_(0.01); m[-1].bias.fill_(final_bias)
        return m

    def _ctx(self, i, t):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        return torch.cat([g, tmp], -1)

    def forward_with_lv(self, v, i, t, chemprop, surface=None, vit=None,
                          cos=None, has_surf=None, has_vit=None, has_cos=None):
        mu = super().forward(v, i, t, chemprop)         # A2 output
        ctx = self._ctx(i, t)

        if self.kind == "A":
            lv_in = ctx
        elif self.kind == "B":
            hv = (has_vit.float().unsqueeze(-1) if has_vit.ndim == 1 else has_vit.float())
            hs = (has_surf.float().unsqueeze(-1) if has_surf.ndim == 1 else has_surf.float())
            p_surf = self.surf_proj(surface) * hs
            p_frame = self.frame_proj(vit) * hv
            branch = torch.cat([ctx, p_surf, p_frame], -1)
            delta = self.delta_head(branch)
            mu = mu + torch.sigmoid(self.gate_b) * delta
            lv_in = branch
        else:  # C
            hc = (has_cos.float().unsqueeze(-1) if has_cos.ndim == 1 else has_cos.float())
            p_cos = self.cosmo_proj(cos) * hc
            branch = torch.cat([ctx, p_cos], -1)
            delta = self.delta_head(branch)
            mu = mu + torch.sigmoid(self.gate_b) * delta
            lv_in = branch

        lv = self.logvar_head(lv_in).clamp(*LV_CLAMP)
        return mu, lv


# ==========================================================================
# Loss helpers
# ==========================================================================
def _gauss_nll(mu, lv, y, valid):
    lv = lv.clamp(*LV_CLAMP)
    nll = 0.5 * torch.exp(-lv) * (mu - y) ** 2 + 0.5 * lv
    return (nll * valid.float()).sum() / valid.float().sum().clamp(min=1)


def _mse(mu, y, valid):
    err2 = ((mu - y) ** 2) * valid.float()
    return err2.sum() / valid.float().sum().clamp(min=1)


def _canon(s):
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return s


def _load_split(s):
    p_dft = CACHE / f"cached_{s}_dft.npz"
    p_std = CACHE / f"cached_{s}.npz"
    p = p_dft if p_dft.exists() else p_std
    return {k: v for k, v in np.load(p, allow_pickle=True).items()}


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
            feats[i] = f; mask[i] = 1.0
    return feats, mask


def _standardize(x, mask=None):
    if mask is not None and mask.sum() > 0:
        mu = x[mask > 0].mean(axis=0); sd = x[mask > 0].std(axis=0) + 1e-6
    else:
        mu = x.mean(axis=0); sd = x.std(axis=0) + 1e-6
    return (x - mu) / sd, mu, sd


# ==========================================================================
# Stage 1-3: train one specialist
# ==========================================================================
def train_specialist(kind, seed, feat_dict, y, device, epochs=300, patience=60,
                      warmup=WARMUP_EPOCHS):
    """Warm-start A2 backbone, freeze it, train specialty + logvar heads with
    MSE warmup → NLL.

    feat_dict contains: v4, morg, thermo, chemprop, surface, vit, cos, has_*
    """
    set_seed(seed)
    n_props = y.shape[1]
    m = A5_BMA_Specialist(kind, feat_dict["morg"].shape[1], n_props,
                            chemprop_dim=feat_dict["chemprop"].shape[1]).to(device)

    if A2_CKPT.exists():
        ckpt = torch.load(A2_CKPT, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        m.load_state_dict(sd, strict=False)

    # Freeze inherited A2 params
    _A2 = {"gate", "heads", "alphas", "cp_proj", "cp_heads", "cp_gate"}
    for name, p in m.named_parameters():
        if name.split(".")[0] in _A2:
            p.requires_grad = False

    train_params = [p for p in m.parameters() if p.requires_grad]
    opt = AdamW(train_params, lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    # Build training tensors
    keys = ("v", "i", "t", "cp", "surf", "vit", "cos", "hs", "hv", "hc")
    ts = {k: torch.from_numpy(feat_dict[full]).to(device)
          for k, full in zip(keys,
                              ("v4", "morg", "thermo", "chemprop",
                               "surface", "vit", "cos",
                               "has_surf", "has_vit", "has_cos"))}
    y_t = torch.from_numpy(y).to(device)
    valid = ~torch.isnan(y_t); yf = torch.nan_to_num(y_t, 0.0)

    ds = TensorDataset(*[ts[k].cpu() for k in keys], yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best_state, best_vl, bad = None, float("inf"), 0
    for ep in range(epochs):
        # Specialist A has no trainable mean parameters (A2 backbone frozen,
        # only logvar_head trains). MSE warmup would produce no gradient. Use
        # NLL from step 0 for A; B/C get the MSE warmup.
        use_nll = (ep >= warmup) or (kind == "A")
        m.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            vb, ib, tb, cpb, sub, vib, cob, hsb, hvb, hcb, yb, vm = batch
            mu, lv = m.forward_with_lv(vb, ib, tb, cpb, surface=sub, vit=vib,
                                         cos=cob, has_surf=hsb, has_vit=hvb, has_cos=hcb)
            loss = _gauss_nll(mu, lv, yb, vm) if use_nll else _mse(mu, yb, vm)
            if not torch.isfinite(loss): continue
            # Safety: skip batches where the loss doesn't connect to trainable
            # params (e.g., edge-case of frozen mean + MSE phase).
            if not loss.requires_grad: continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            mu, lv = m.forward_with_lv(ts["v"], ts["i"], ts["t"], ts["cp"],
                                         surface=ts["surf"], vit=ts["vit"],
                                         cos=ts["cos"], has_surf=ts["hs"],
                                         has_vit=ts["hv"], has_cos=ts["hc"])
            vl = (_gauss_nll(mu, lv, yf, valid) if use_nll
                  else _mse(mu, yf, valid)).item()
        if np.isfinite(vl) and vl < best_vl:
            best_vl = vl
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None: m.load_state_dict(best_state)
    m.eval()
    return m, best_vl


# ==========================================================================
# Router model
#   - MLP mode (V4 PerMoleculeRouter pattern): per-sample, per-prop learned
#     correction — ~21k params.
#   - Scalar mode (V5 simplicity, ablation): per-(specialist, prop) learned
#     scalar — K×P = 24 params. No context-dependent routing. Much less
#     overfit risk on the 5189-row cache. A/B test vs MLP variant.
# Both modes: anchored to log(1/σ²_k) so softmax at zero-correction ≡ pure BMA.
# ==========================================================================
class A5_BMA_Router(nn.Module):
    def __init__(self, feat_dim, n_props, K=K_SPECIALISTS, hidden=64,
                  mode="mlp"):
        super().__init__()
        assert mode in ("mlp", "scalar")
        self.K = K; self.P = n_props; self.mode = mode
        if mode == "mlp":
            self.net = nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.GELU(), nn.Dropout(0.3),
                nn.LayerNorm(hidden), nn.Linear(hidden, K * n_props),
            )
            with torch.no_grad():
                self.net[-1].weight.zero_(); self.net[-1].bias.zero_()
        else:  # scalar
            # Single per-(specialist, prop) learned scalar; no input dependence.
            self.scalar_corr = nn.Parameter(torch.zeros(K, n_props))

    def forward(self, chemprop, surface, thermo, lv_stack):
        """lv_stack: (B, K, P) — specialist logvars (already clamped)"""
        if self.mode == "mlp":
            x = torch.cat([chemprop, surface, thermo], -1)
            corr = self.net(x).view(-1, self.K, self.P)       # (B, K, P)
        else:  # scalar: broadcast the (K, P) correction across batch
            corr = self.scalar_corr.unsqueeze(0)                # (1, K, P)
        anchor = -lv_stack                        # (B, K, P) log(1/σ²_k)
        return F.softmax(anchor + corr, dim=1)    # (B, K, P)


# ==========================================================================
# Stage 4: train router on frozen specialists
# ==========================================================================
def train_router(specialists, feat_dict, y, device, epochs=200, patience=30,
                  mode="mlp"):
    """Freeze K specialists (inferred from list length); train only the router + fused-NLL.
    mode ∈ {'mlp', 'scalar'}: MLP corrector (V4 pattern) vs static scalar
    corrector (V5 simplicity) on top of the inverse-variance BMA anchor."""
    K = len(specialists)  # runtime K, handles 3 or 4 specialists
    for m in specialists:
        if isinstance(m, nn.Module):
            m.eval()
    n_props = y.shape[1]
    cp_dim = feat_dict["chemprop"].shape[1]
    surf_dim = feat_dict["surface"].shape[1]
    router = A5_BMA_Router(cp_dim + surf_dim + 25, n_props, K=K, mode=mode).to(device)

    opt = AdamW(router.parameters(), lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    # Pre-compute specialist predictions on train+val (they're frozen — cheap)
    def _predict_all(kind_keys=("v","i","t","cp","surf","vit","cos","hs","hv","hc")):
        pass  # done in-batch below

    keys = ("v","i","t","cp","surf","vit","cos","hs","hv","hc")
    ts = {k: torch.from_numpy(feat_dict[full]).to(device)
          for k, full in zip(keys,
                              ("v4","morg","thermo","chemprop",
                               "surface","vit","cos",
                               "has_surf","has_vit","has_cos"))}
    y_t = torch.from_numpy(y).to(device)
    valid = ~torch.isnan(y_t); yf = torch.nan_to_num(y_t, 0.0)

    ds = TensorDataset(*[ts[k].cpu() for k in keys], yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for ep in range(epochs):
        router.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            vb, ib, tb, cpb, sub, vib, cob, hsb, hvb, hcb, yb, vm = batch
            with torch.no_grad():
                mus, lvs = [], []
                for m in specialists:
                    mu, lv = m.forward_with_lv(vb, ib, tb, cpb, surface=sub,
                                                 vit=vib, cos=cob,
                                                 has_surf=hsb, has_vit=hvb, has_cos=hcb)
                    mus.append(mu); lvs.append(lv)
                mu_stack = torch.stack(mus, dim=1)     # (B, K, P)
                lv_stack = torch.stack(lvs, dim=1)
            w = router(cpb, sub, tb, lv_stack)           # (B, K, P)
            mu_fused = (w * mu_stack).sum(dim=1)
            prec = torch.exp(-lv_stack).sum(dim=1) + 1e-8
            lv_fused = -torch.log(prec)
            loss = _gauss_nll(mu_fused, lv_fused, yb, vm)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0); opt.step()
        sch.step()
        router.eval()
        with torch.no_grad():
            mus, lvs = [], []
            for m in specialists:
                mu, lv = m.forward_with_lv(ts["v"], ts["i"], ts["t"], ts["cp"],
                                             surface=ts["surf"], vit=ts["vit"],
                                             cos=ts["cos"], has_surf=ts["hs"],
                                             has_vit=ts["hv"], has_cos=ts["hc"])
                mus.append(mu); lvs.append(lv)
            mu_s = torch.stack(mus, dim=1)
            lv_s = torch.stack(lvs, dim=1)
            w = router(ts["cp"], ts["surf"], ts["t"], lv_s)
            mu_f = (w * mu_s).sum(dim=1)
            prec = torch.exp(-lv_s).sum(dim=1) + 1e-8
            lv_f = -torch.log(prec)
            tl = _gauss_nll(mu_f, lv_f, yf, valid).item()
        if np.isfinite(tl) and tl < best:
            best = tl
            state = {k: v.clone() for k, v in router.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: router.load_state_dict(state)
    router.eval()
    return router


def evaluate(specialists, router, feat_dict, y, device):
    keys = ("v","i","t","cp","surf","vit","cos","hs","hv","hc")
    tens = [torch.from_numpy(feat_dict[full]).to(device)
            for full in ("v4","morg","thermo","chemprop",
                          "surface","vit","cos",
                          "has_surf","has_vit","has_cos")]
    with torch.no_grad():
        mus, lvs = [], []
        for m in specialists:
            mu, lv = m.forward_with_lv(*tens[:4], surface=tens[4], vit=tens[5],
                                         cos=tens[6], has_surf=tens[7],
                                         has_vit=tens[8], has_cos=tens[9])
            mus.append(mu); lvs.append(lv)
        mu_s = torch.stack(mus, dim=1)
        lv_s = torch.stack(lvs, dim=1)
        w = router(tens[3], tens[4], tens[2], lv_s)
        mu_f = (w * mu_s).sum(dim=1)
        prec = torch.exp(-lv_s).sum(dim=1) + 1e-8
        lv_f = -torch.log(prec)
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
    core7 = [r[p] for p in PROPS[:7] if p in r and np.isfinite(r[p])]
    r["avg_core7"] = float(np.mean(core7)) if core7 else float("nan")
    return r


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds-per-specialist", type=int, default=3)
    ap.add_argument("--n-seeds-router", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--force-retrain", action="store_true",
                    help="Ignore existing specialist checkpoints and retrain.")
    ap.add_argument("--router-mode", choices=["mlp", "scalar"], default="mlp",
                    help="V4-style MLP corrector vs V5-style K×P scalar corrector.")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load and preprocess cache ---
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

    vit_bank = dict(zip(*[np.load(VIT_BANK, allow_pickle=True)[k]
                             for k in ("smiles", "vit_feat")]))
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

    v4_tr, v4_va, v4_te = v4_base(tr), v4_base(va), v4_base(te)
    y_tr, y_te = tr["targets"].astype(np.float32), te["targets"].astype(np.float32)
    th_tr, th_va, th_te = tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]

    print(f"Surf cov: tr {hs_tr.mean():.1%} / te {hs_te.mean():.1%}  "
          f"ViT: tr {hv_tr.mean():.1%} / te {hv_te.mean():.1%}  "
          f"COSMO: tr {hc_tr.mean():.1%} / te {hc_te.mean():.1%}")

    feats_tr = {"v4": v4_tr, "morg": m_tr, "thermo": th_tr, "chemprop": cp_tr,
                 "surface": surf_tr, "vit": vit_tr, "cos": cos_tr,
                 "has_surf": hs_tr, "has_vit": hv_tr, "has_cos": hc_tr}
    feats_te = {"v4": v4_te, "morg": m_te, "thermo": th_te, "chemprop": cp_te,
                 "surface": surf_te, "vit": vit_te, "cos": cos_te,
                 "has_surf": hs_te, "has_vit": hv_te, "has_cos": hc_te}

    # --- Stages 1-3: train specialists ---
    specialists = {}
    for kind in ("A", "B", "C"):
        ckpt_path = BMA_DIR / f"specialist_{kind}.pt"
        if ckpt_path.exists() and not args.force_retrain:
            print(f"\n[Specialist {kind}] loading cached ckpt → {ckpt_path.name}")
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = A5_BMA_Specialist(kind, m_tr.shape[1], y_tr.shape[1],
                                         chemprop_dim=cp_tr.shape[1]).to(device)
            model.load_state_dict(ck["state_dict"])
            specialists[kind] = model
            continue

        best = None
        for seed in range(args.n_seeds_per_specialist):
            print(f"\n[Specialist {kind}] seed {seed} (MSE warmup {WARMUP_EPOCHS} → NLL)...")
            m_k, vl = train_specialist(kind, seed, feats_tr, y_tr, device,
                                         epochs=args.epochs)
            m_te_pred, _ = m_k.forward_with_lv(
                *(torch.from_numpy(feats_te[full]).to(device)
                  for full in ("v4","morg","thermo","chemprop")),
                surface=torch.from_numpy(feats_te["surface"]).to(device),
                vit=torch.from_numpy(feats_te["vit"]).to(device),
                cos=torch.from_numpy(feats_te["cos"]).to(device),
                has_surf=torch.from_numpy(feats_te["has_surf"]).to(device),
                has_vit=torch.from_numpy(feats_te["has_vit"]).to(device),
                has_cos=torch.from_numpy(feats_te["has_cos"]).to(device))
            r = r2_per_prop(m_te_pred.detach().cpu().numpy(), y_te)
            print(f"  [Sp {kind}] seed {seed}: val_loss={vl:.4f}  test core7={r['avg_core7']:.4f}")
            if best is None or vl < best["val_loss"]:
                best = {"state_dict": {k: v.detach().cpu() for k, v in m_k.state_dict().items()},
                        "seed": seed, "val_loss": vl, "test_core7": r["avg_core7"]}
        torch.save(best, ckpt_path)
        print(f"[Specialist {kind}] saved best (seed={best['seed']}, "
               f"core7={best['test_core7']:.4f}) → {ckpt_path.name}")
        # Reload to fresh device model
        model = A5_BMA_Specialist(kind, m_tr.shape[1], y_tr.shape[1],
                                     chemprop_dim=cp_tr.shape[1]).to(device)
        model.load_state_dict(best["state_dict"])
        specialists[kind] = model

    # --- Stage 4: train router ---
    print(f"\n{'='*70}\nStage 4: Router (V4-anchored, specialists frozen)\n{'='*70}")
    router_r2_fused, router_r2_g50, router_r2_g25 = [], [], []
    r_a, r_b, r_c = [], [], []
    print(f"Router mode: {args.router_mode}")
    for seed in range(args.n_seeds_router):
        print(f"\n[Router] seed {seed} ({args.router_mode}) ...")
        set_seed(seed + 500)
        spec_list = [specialists["A"], specialists["B"], specialists["C"]]
        router = train_router(spec_list, feats_tr, y_tr, device,
                                epochs=args.epochs // 2, mode=args.router_mode)

        mu_f, lv_f, mu_s, lv_s, w = evaluate(spec_list, router, feats_te, y_te, device)
        rf = r2_per_prop(mu_f, y_te)
        ra = r2_per_prop(mu_s[:, 0], y_te)
        rb = r2_per_prop(mu_s[:, 1], y_te)
        rc = r2_per_prop(mu_s[:, 2], y_te)
        g50 = conf_gated_r2(mu_f, y_te, lv_f, quantile=0.5)
        g25 = conf_gated_r2(mu_f, y_te, lv_f, quantile=0.25)
        router_r2_fused.append(rf); router_r2_g50.append(g50); router_r2_g25.append(g25)
        r_a.append(ra); r_b.append(rb); r_c.append(rc)
        w_mean = w.mean(axis=(0, 2))
        print(f"  Specialists: A={ra['avg_core7']:.4f}  B={rb['avg_core7']:.4f}  C={rc['avg_core7']:.4f}")
        print(f"  Fused={rf['avg_core7']:.4f}  gated@50%={g50['avg_core7']:.4f}  gated@25%={g25['avg_core7']:.4f}")
        print(f"  Router weights (A, B, C): {w_mean.round(3).tolist()}")

    def _sum(name, rs):
        c = [r["avg_core7"] for r in rs]
        out = {"name": name, "avg_r2_core7": float(np.mean(c)),
                "std_r2_core7": float(np.std(c)), "per_prop": {}}
        for p in PROPS:
            vs = [r.get(p) for r in rs if r.get(p) is not None and not np.isnan(r.get(p, float("nan")))]
            out["per_prop"][p] = float(np.mean(vs)) if vs else float("nan")
        return out

    summary = {
        "fused": _sum("BMA_fused", router_r2_fused),
        "gated_50": _sum("BMA_fused_gated50", router_r2_g50),
        "gated_25": _sum("BMA_fused_gated25", router_r2_g25),
        "specialist_A": _sum("Specialist_A", r_a),
        "specialist_B": _sum("Specialist_B", r_b),
        "specialist_C": _sum("Specialist_C", r_c),
    }
    print(f"\n{'='*72}\nA5.9 BMA Pipeline Summary (sequential training, warmup+NLL)\n{'='*72}")
    print(f"{'Arm':<35}{'core7':>10}{'std':>10}")
    for k, v in summary.items():
        print(f"{v['name']:<35}{v['avg_r2_core7']:>10.4f}{v['std_r2_core7']:>10.4f}")

    outf = V5 / "results" / f"a5_bma_pipeline_{args.router_mode}.json"
    json.dump(summary, open(outf, "w"), indent=2)
    print(f"\nSaved: {outf}")


if __name__ == "__main__":
    main()
