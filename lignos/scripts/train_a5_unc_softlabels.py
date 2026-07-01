"""A5.2 + A5.4 stack — Uncertainty head + COSMO-SAC soft labels combined.

Components:
  - Frozen A2 backbone (from checkpoint)
  - cosmo_corr head (A5.4): per-prop correction, zero-init output, gated
  - logvar_heads (A5.2): per-prop log-variance, zero-bias init

Training losses:
  L_main = Gaussian NLL on (corrected_mean, logvar, y) where y is not NaN
  L_aux  = MSE(corrected_mean[G_E/γ2], cosmo_soft_label) on rows where real
            y is NaN AND the IL has a COSMO-SAC prediction
  loss   = L_main + λ_aux * L_aux

Expected benefits (compound):
  - From A5.2: confidence-gated@50% R² ≈ 0.88 on core7
  - From A5.4: soft-labels give the 5147 unlabeled rows gradient pathway →
    cosmo_corr head can learn thermodynamic consistency adjustments
  - Combined: representation shaped by physics + calibrated uncertainty on
    top of a quality-preserved A2 backbone

At inference, the output mean is the corrected mean; log_var gives sigma for
confidence gating.
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
from train_a5_uncertainty import gaussian_nll_loss
from train_a5_cosmosac_softlabels import load_soft_label_bank, assemble_soft_labels

CACHE = V5 / "data" / "LignoIL_A1"
A2_CKPT = V5 / "checkpoints" / "a2" / "stage1_best.pt"
IDX_GAMMA2 = PROPS.index("gamma2")
IDX_G_E = PROPS.index("G_E")


class A5UncSoftHead(A2Head):
    """A2Head + cosmo-correction head + logvar heads."""

    def __init__(self, nf, n_props=8, chemprop_dim=40):
        super().__init__(nf, n_props, chemprop_dim)
        # Correction head from A5.4
        self.cosmo_corr = nn.Sequential(
            nn.Linear(nf + 5, 32), nn.GELU(), nn.Linear(32, n_props))
        with torch.no_grad():
            self.cosmo_corr[-1].weight.zero_()
            self.cosmo_corr[-1].bias.zero_()
        self.corr_gate = nn.Parameter(torch.full((n_props,), -3.0))
        # Logvar heads from A5.2
        self.logvar_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(nf + 5, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in range(n_props)])
        for h in self.logvar_heads:
            with torch.no_grad():
                h[-1].weight.zero_(); h[-1].bias.zero_()

    def _gated_context(self, i, t):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        return torch.cat([g, tmp], -1)

    def forward(self, v, i, t, chemprop):
        out = super().forward(v, i, t, chemprop)
        ctx = self._gated_context(i, t)
        delta = self.cosmo_corr(ctx)
        return out + torch.sigmoid(self.corr_gate) * delta

    def predict_logvar(self, i, t):
        ctx = self._gated_context(i, t)
        return torch.cat([h(ctx) for h in self.logvar_heads], -1)


def _load_split(s):
    p_dft = CACHE / f"cached_{s}_dft.npz"
    p_std = CACHE / f"cached_{s}.npz"
    p = p_dft if p_dft.exists() else p_std
    print(f"[{s}] loading {p.name}")
    return {k: v for k, v in np.load(p, allow_pickle=True).items()}


def train_stage1(seed, v4, morg, th, cp, ln_gw, g_e, has_soft, y, device,
                  lambda_aux=0.01, epochs=300, patience=50):
    set_seed(seed)
    n_props = y.shape[1]
    m = A5UncSoftHead(morg.shape[1], n_props, chemprop_dim=cp.shape[1]).to(device)

    if A2_CKPT.exists():
        ckpt = torch.load(A2_CKPT, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        miss, unex = m.load_state_dict(sd, strict=False)
        print(f"  warm-started A2 ckpt ({len(miss)} unmatched, {len(unex)} unused)")

    # Freeze A2 backbone; train cosmo_corr + corr_gate + logvar_heads
    for name, p in m.named_parameters():
        if not name.startswith(("cosmo_corr", "corr_gate", "logvar_")):
            p.requires_grad = False

    train_params = [p for p in m.parameters() if p.requires_grad]
    print(f"  trainable params: {sum(p.numel() for p in train_params)}")

    opt = AdamW(train_params, lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    ts = {k: torch.from_numpy(x).to(device) for k, x in
          dict(v=v4, i=morg, t=th, cp=cp, lg=ln_gw, ge=g_e, hs=has_soft, y=y).items()}
    valid = ~torch.isnan(ts["y"]); yf = torch.nan_to_num(ts["y"], 0.0)
    ds = TensorDataset(*[ts[k].cpu() for k in ("v","i","t","cp","lg","ge","hs")],
                        yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        for vb, ib, tb, cpb, lgb, geb, hsb, yb, vm in loader:
            vb, ib, tb, cpb, lgb, geb, hsb, yb, vm = [x.to(device)
                for x in (vb, ib, tb, cpb, lgb, geb, hsb, yb, vm)]
            pred = m(vb, ib, tb, cpb)
            logvar = m.predict_logvar(ib, tb)
            # NLL main loss
            main = gaussian_nll_loss(pred, logvar, yb, vm)
            # Soft-label aux loss on unlabeled rows
            aux_mask_g_e = hsb * (~vm[:, IDX_G_E]).float()
            aux_mask_g2 = hsb * (~vm[:, IDX_GAMMA2]).float()
            aux_g_e = ((pred[:, IDX_G_E] - geb) ** 2 * aux_mask_g_e).sum() / aux_mask_g_e.sum().clamp(min=1)
            aux_g2 = ((pred[:, IDX_GAMMA2] - lgb) ** 2 * aux_mask_g2).sum() / aux_mask_g2.sum().clamp(min=1)
            loss = main + lambda_aux * (aux_g_e + aux_g2)
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


def predict(m, v4, morg, th, cp, device):
    with torch.no_grad():
        pred = m(torch.from_numpy(v4).to(device),
                 torch.from_numpy(morg).to(device),
                 torch.from_numpy(th).to(device),
                 torch.from_numpy(cp).to(device)).cpu().numpy()
        lv = m.predict_logvar(torch.from_numpy(morg).to(device),
                               torch.from_numpy(th).to(device)).cpu().numpy()
    return pred, lv


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
    ap.add_argument("--lambda-aux", type=float, default=0.01)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  lambda_aux={args.lambda_aux}")
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

    bank = load_soft_label_bank()
    lg_tr, ge_tr, hs_tr = assemble_soft_labels(tr["smiles"], bank)
    lg_va, ge_va, hs_va = assemble_soft_labels(va["smiles"], bank)
    lg_te, ge_te, hs_te = assemble_soft_labels(te["smiles"], bank)
    # Standardize soft labels to z-score (match target scaling)
    y_tr_orig = tr["targets"].astype(np.float32)
    for idx, soft in [(IDX_GAMMA2, lg_tr), (IDX_G_E, ge_tr)]:
        real = y_tr_orig[:, idx]
        mu = np.nanmean(real); sd = np.nanstd(real) + 1e-6
        soft -= mu; soft /= sd
    for arr_pair in [(lg_va, IDX_GAMMA2), (lg_te, IDX_GAMMA2), (ge_va, IDX_G_E), (ge_te, IDX_G_E)]:
        arr, idx = arr_pair
        mu = np.nanmean(y_tr_orig[:, idx]); sd = np.nanstd(y_tr_orig[:, idx]) + 1e-6
        arr -= mu; arr /= sd
        arr[np.isnan(arr)] = 0.0

    v4_tr, v4_va, v4_te = v4_base(tr), v4_base(va), v4_base(te)
    y_tr, y_te = tr["targets"].astype(np.float32), te["targets"].astype(np.float32)
    th_tr, th_va, th_te = tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]

    s1_r2s, s1_gated50_r2s, s2_r2s = [], [], []
    for seed in range(args.n_seeds):
        print(f"\n[seed {seed}] Stage-1 (A5.2+A5.4 stack)...")
        s1 = train_stage1(seed, v4_tr, m_tr, th_tr, cp_tr,
                           lg_tr, ge_tr, hs_tr, y_tr,
                           device, lambda_aux=args.lambda_aux,
                           epochs=args.epochs)
        pred_te, lv_te = predict(s1, v4_te, m_te, th_te, cp_te, device)
        r = r2_per_prop(pred_te, y_te)
        gated = confidence_gated_r2(pred_te, y_te, lv_te, quantile=0.5)
        s1_r2s.append(r); s1_gated50_r2s.append(gated)
        print(f"  Stage-1 core7={r['avg_core7']:.4f}  gated@50%={gated['avg_core7']:.4f}  "
              f"lignin={r.get('lignin_wt', float('nan')):.4f}")

        print(f"[seed {seed}] Stage-2 (hardfreeze + deep lignin)...")
        s2 = train_stage2_lignin(s1, v4_tr, m_tr, th_tr, cp_tr, p_tr, hp_tr, y_tr,
                                  device, seed=seed + 100, epochs=args.epochs)
        s2_pred = predict_stage2(s2, v4_te, m_te, th_te, cp_te, p_te, hp_te, device)
        r2 = r2_per_prop(s2_pred, y_te)
        s2_r2s.append(r2)
        print(f"  Stage-2 core7={r2['avg_core7']:.4f}  lignin={r2.get('lignin_wt', float('nan')):.4f}")

    tag = f"lambda{args.lambda_aux}"
    s1 = summarize(f"Stage1_A5_unc_softlabels_{tag}", s1_r2s)
    s1g = summarize(f"Stage1_A5_unc_softlabels_gated50_{tag}", s1_gated50_r2s)
    s2 = summarize(f"Stage2_A5_deep_lignin_{tag}", s2_r2s)
    print(f"\n{'='*70}\nA5.2 + A5.4 STACK SUMMARY\n{'='*70}")
    print(f"{'Stage':<50}{'core7':>10}{'std':>10}{'lignin':>10}")
    for r in [s1, s1g, s2]:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<50}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}{lig:>10.4f}")
    out = V5 / "results" / f"a5_unc_softlabels_{tag}.json"
    json.dump([s1, s1g, s2], open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
