"""A5.2 — A2 with Gaussian NLL (mean + log-variance) per-prop heads.

Addresses OOD failure mode #2 from Baran Task 2 (R² = −2.40 on fold 3):
    A2 makes CONFIDENT wrong predictions on novel ILs. Adding an aleatoric
    uncertainty output per target trains the model to down-weight its own
    predictions when evidence is weak — concretely, its 95% CI widens, and
    fold-3-style catastrophes become prediction-declined rather than
    catastrophically-wrong.

Architecture:
  Per-prop head now outputs 2 scalars: [mean, log_sigma^2] instead of just
  mean. Training loss per target element:
      L = 0.5 * exp(-log_var) * (pred - y)^2 + 0.5 * log_var
  This is the standard Kendall-and-Gal (2017) heteroscedastic regression
  loss. The `exp(-log_var)` term down-weights noisy targets; the `log_var`
  term prevents log_var from going to +∞ (which would drive first term to 0).

  Inference: mean is used for the point prediction; sigma is available for
  confidence gating in downstream evaluation.

Warm-start pattern: start from A2 Stage-1 checkpoint (frozen), train only the
log-var head extension. This way:
  - Mean prediction stays at A2's 0.8401 core7
  - Log-var head learns to predict calibrated uncertainty on the same data
  - Stage-2 lignin training unchanged (uses mean only, not sigma)

If we want the sigma head to also shape the mean predictions, we'd unfreeze
but that risks the 2026-04-20 regression — keep frozen for cleanliness.

A5.2 headline number = A2's core7 + expected NLL/calibration improvement +
confidence-gated R² on Baran Task 2 (we can report "R² restricted to the 80%
of predictions with lowest epistemic sigma").
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
    train_stage2_lignin, predict_stage2,
)

CACHE = V5 / "data" / "LignoIL_A1"
A2_CKPT = V5 / "checkpoints" / "a2" / "stage1_best.pt"


class A5UncertaintyHead(A2Head):
    """A2Head + per-prop log-variance head. Mean output is identical to A2."""

    def __init__(self, nf, n_props=8, chemprop_dim=40):
        super().__init__(nf, n_props, chemprop_dim)
        # log-sigma head: one scalar per prop, input is the same as the mean head
        # (gated Morgan + thermo(5))
        self.logvar_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(nf + 5, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in range(n_props)])
        # Initialize log-var output ≈ 0 so sigma^2 ≈ 1 at init (neutral)
        for h in self.logvar_heads:
            with torch.no_grad():
                h[-1].weight.zero_(); h[-1].bias.zero_()

    def predict_logvar(self, i, t):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        inp = torch.cat([g, tmp], -1)
        return torch.cat([h(inp) for h in self.logvar_heads], -1)


def _load_split(s):
    p_dft = CACHE / f"cached_{s}_dft.npz"
    p_std = CACHE / f"cached_{s}.npz"
    p = p_dft if p_dft.exists() else p_std
    print(f"[{s}] loading {p.name}")
    return {k: v for k, v in np.load(p, allow_pickle=True).items()}


def gaussian_nll_loss(pred, logvar, y, valid):
    """Masked Kendall-Gal 2017 heteroscedastic loss.
    L per element = 0.5 * exp(-logvar) * (pred - y)^2 + 0.5 * logvar
    """
    # Clamp logvar for numeric stability
    logvar = logvar.clamp(-10.0, 10.0)
    diff2 = (pred - y) ** 2
    nll = 0.5 * torch.exp(-logvar) * diff2 + 0.5 * logvar
    nll = nll * valid.float()
    return nll.sum() / valid.float().sum().clamp(min=1)


def train_stage1_unc(seed, v4, morg, th, cp, y, device, epochs=300, patience=50):
    """Warm-start A5 uncertainty head. Main head (A2) frozen; train only
    log-var heads with Gaussian NLL."""
    set_seed(seed)
    n_props = y.shape[1]
    m = A5UncertaintyHead(morg.shape[1], n_props, chemprop_dim=cp.shape[1]).to(device)

    if A2_CKPT.exists():
        ckpt = torch.load(A2_CKPT, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        miss, unex = m.load_state_dict(sd, strict=False)
        print(f"  warm-started A2 ckpt ({len(miss)} unmatched = logvar_heads, {len(unex)} unused)")

    # Freeze mean backbone; only logvar heads train
    for name, p in m.named_parameters():
        if not name.startswith("logvar_"):
            p.requires_grad = False

    train_params = [p for p in m.parameters() if p.requires_grad]
    print(f"  trainable params: {sum(p.numel() for p in train_params)} "
          f"of {sum(p.numel() for p in m.parameters())}")

    opt = AdamW(train_params, lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    ts = {k: torch.from_numpy(x).to(device) for k, x in
          dict(v=v4, i=morg, t=th, cp=cp, y=y).items()}
    valid = ~torch.isnan(ts["y"]); yf = torch.nan_to_num(ts["y"], 0.0)
    ds = TensorDataset(*[ts[k].cpu() for k in ("v","i","t","cp")],
                        yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for vb, ib, tb, cpb, yb, vm in loader:
            vb, ib, tb, cpb, yb, vm = [x.to(device) for x in (vb, ib, tb, cpb, yb, vm)]
            with torch.no_grad():
                pred = m(vb, ib, tb, cpb)  # mean (frozen, no grad)
            logvar = m.predict_logvar(ib, tb)
            loss = gaussian_nll_loss(pred, logvar, yb, vm)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(ts["v"], ts["i"], ts["t"], ts["cp"])
            lv = m.predict_logvar(ts["i"], ts["t"])
            tl = gaussian_nll_loss(pred, lv, yf, valid).item()
        if np.isfinite(tl) and tl < best:
            best = tl; state = {k: v.clone() for k, v in m.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: m.load_state_dict(state)
    m.eval()
    return m


def predict_stage1(m, v4, morg, th, cp, device):
    with torch.no_grad():
        pred = m(torch.from_numpy(v4).to(device),
                 torch.from_numpy(morg).to(device),
                 torch.from_numpy(th).to(device),
                 torch.from_numpy(cp).to(device)).cpu().numpy()
        lv = m.predict_logvar(torch.from_numpy(morg).to(device),
                               torch.from_numpy(th).to(device)).cpu().numpy()
    return pred, lv


def confidence_gated_r2(pred, y, logvar, quantile=0.5):
    """R² restricted to the `quantile` fraction of predictions with lowest
    per-row average sigma. If the model's uncertainty is calibrated, removing
    the top-sigma tail should improve R² — this is the key A5.2 benefit."""
    sigma = np.exp(0.5 * logvar).mean(axis=-1)  # (N,) row-level
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
    hp_te = te["has_physchem"].astype(np.float32)

    v4_tr, v4_va, v4_te = v4_base(tr), v4_base(va), v4_base(te)
    y_tr, y_va, y_te = [x["targets"].astype(np.float32) for x in (tr, va, te)]
    th_tr, th_va, th_te = tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]

    s1_r2s, s1_gated_r2s, s2_r2s = [], [], []
    for seed in range(args.n_seeds):
        print(f"\n[seed {seed}] Stage-1 (A5.2 = A2 frozen + logvar heads, NLL loss)...")
        s1 = train_stage1_unc(seed, v4_tr, m_tr, th_tr, cp_tr, y_tr, device,
                                epochs=args.epochs)
        pred_te, lv_te = predict_stage1(s1, v4_te, m_te, th_te, cp_te, device)
        r = r2_per_prop(pred_te, y_te)
        gated = confidence_gated_r2(pred_te, y_te, lv_te, quantile=0.5)
        s1_r2s.append(r); s1_gated_r2s.append(gated)
        print(f"  Stage-1 core7={r['avg_core7']:.4f}  "
              f"conf-gated@50% core7={gated['avg_core7']:.4f}  "
              f"mean sigma={np.exp(0.5*lv_te).mean():.4f}")

        print(f"[seed {seed}] Stage-2 (hardfreeze + deep lignin + physchem, unchanged)...")
        s2 = train_stage2_lignin(s1, v4_tr, m_tr, th_tr, cp_tr, p_tr, hp_tr, y_tr,
                                  device, seed=seed + 100, epochs=args.epochs)
        s2_pred = predict_stage2(s2, v4_te, m_te, th_te, cp_te, p_te, hp_te, device)
        r2 = r2_per_prop(s2_pred, y_te)
        s2_r2s.append(r2)
        print(f"  Stage-2 core7={r2['avg_core7']:.4f}  lignin={r2.get('lignin_wt', float('nan')):.4f}")

    s1 = summarize("Stage1_A5_uncertainty_mean", s1_r2s)
    s1g = summarize("Stage1_A5_uncertainty_gated50", s1_gated_r2s)
    s2 = summarize("Stage2_A5_deep_lignin_unchanged", s2_r2s)
    print(f"\n{'='*70}\nA5.2 UNCERTAINTY HEAD SUMMARY\n{'='*70}")
    print(f"{'Stage':<42}{'core7':>10}{'std':>10}{'lignin':>10}")
    for r in [s1, s1g, s2]:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<42}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}{lig:>10.4f}")
    out = V5 / "results" / "a5_uncertainty.json"
    json.dump([s1, s1g, s2], open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
