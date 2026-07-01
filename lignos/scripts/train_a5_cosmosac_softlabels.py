"""A5.4 — A2 + COSMO-SAC soft labels on unlabeled rows.

The correct "Plan B intent": use first-principles COSMO-SAC predictions as
pseudo-labels on the 5147 zero-target ILThermo pre-training rows. Aux loss
flows gradient through the main A2 backbone, shaping its representation via
thermodynamically consistent physics on rows that otherwise contribute zero
training signal.

Soft labels (per IL, computed at T=298 K, x=0.5 binary IL+water):
  G_E_cosmo  : excess Gibbs of mixing (J/mol)  ← used as G_E aux target
  ln_gamma_w : ln γ of water in the IL         ← used as γ₂ aux target

Only applied to rows where the REAL label is NaN. For rows with real data,
the main MSE loss dominates (no aux interference).

Training loop:
  L_main = masked_MSE(pred, y_real)        on rows where y is not NaN
  L_aux_G_E    = MSE(pred[G_E], G_E_cosmo)  on rows where y[G_E] IS NaN and
                                             the IL has a cosmo soft label
  L_aux_gamma2 = MSE(pred[gamma2], ln γ_w)  (similarly)
  loss = L_main + lambda_aux * (L_aux_G_E + L_aux_gamma2)

Warm-start + freeze A2 backbone so aux loss can't destabilize Stage-1 (per
the 2026-04-20 collapse diagnostic). But we WANT the aux loss to shape the
representation — so we need a different pattern: train from scratch with
the COSMO-SAC aux directing the residual prediction toward thermodynamic
consistency.

This script uses BOTH warm-start backbone AND trains a small post-A2
'cosmo_correction_head' that adjusts predictions toward COSMO-SAC on unlabeled
rows.
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
SOFT_LABEL_BANK = V5 / "data" / "cosmo_sac_soft_labels.npz"
A2_CKPT = V5 / "checkpoints" / "a2" / "stage1_best.pt"

# Target indices
IDX_GAMMA2 = PROPS.index("gamma2")
IDX_G_E = PROPS.index("G_E")


class A5CosmoSACHead(A2Head):
    """A2Head + per-prop correction head that can be nudged by COSMO-SAC aux loss."""

    def __init__(self, nf, n_props=8, chemprop_dim=40):
        super().__init__(nf, n_props, chemprop_dim)
        # Small correction head on gated Morgan+thermo, zero-init
        self.cosmo_corr = nn.Sequential(
            nn.Linear(nf + 5, 32), nn.GELU(), nn.Linear(32, n_props))
        with torch.no_grad():
            self.cosmo_corr[-1].weight.zero_()
            self.cosmo_corr[-1].bias.zero_()
        # Per-prop gate, init -3 (≈5% contribution)
        self.corr_gate = nn.Parameter(torch.full((n_props,), -3.0))

    def forward(self, v, i, t, chemprop):
        out = super().forward(v, i, t, chemprop)
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        inp = torch.cat([g, tmp], -1)
        delta = self.cosmo_corr(inp)
        return out + torch.sigmoid(self.corr_gate) * delta


def load_soft_label_bank():
    z = np.load(SOFT_LABEL_BANK, allow_pickle=True)
    return dict(zip(z["smiles"], zip(z["ln_gamma_water"], z["G_E_cosmo"])))


def _canon(s):
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return s


def assemble_soft_labels(smiles, bank):
    """Return per-row (ln γ_w, G_E_cosmo, has_soft) arrays aligned with smiles."""
    n = len(smiles)
    ln_gw = np.zeros(n, dtype=np.float32)
    g_e = np.zeros(n, dtype=np.float32)
    has_soft = np.zeros(n, dtype=np.float32)
    for i, s in enumerate(smiles):
        cs = _canon(s)
        if cs in bank:
            ln_gw[i], g_e[i] = bank[cs]
            has_soft[i] = 1.0
        elif s in bank:
            ln_gw[i], g_e[i] = bank[s]
            has_soft[i] = 1.0
    print(f"[soft] {int(has_soft.sum())}/{n} rows with COSMO-SAC soft labels")
    return ln_gw, g_e, has_soft


def _load_split(s):
    p_dft = CACHE / f"cached_{s}_dft.npz"
    p_std = CACHE / f"cached_{s}.npz"
    p = p_dft if p_dft.exists() else p_std
    print(f"[{s}] loading {p.name}")
    return {k: v for k, v in np.load(p, allow_pickle=True).items()}


def train_stage1_softlabel(seed, v4, morg, th, cp, ln_gw, g_e, has_soft, y,
                             device, lambda_aux=0.01, epochs=300, patience=50):
    """Train correction head with main MSE + aux soft-label MSE on unlabeled rows.

    The A2 backbone is warm-started and frozen; only `cosmo_corr` + `corr_gate`
    train. This keeps Stage-1 core7 ≥ A2 baseline by construction.
    """
    set_seed(seed)
    n_props = y.shape[1]
    m = A5CosmoSACHead(morg.shape[1], n_props, chemprop_dim=cp.shape[1]).to(device)

    if A2_CKPT.exists():
        ckpt = torch.load(A2_CKPT, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        miss, unex = m.load_state_dict(sd, strict=False)
        print(f"  warm-started A2 ckpt ({len(miss)} unmatched, {len(unex)} unused)")

    for name, p in m.named_parameters():
        if not name.startswith(("cosmo_corr", "corr_gate")):
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
            err2 = ((pred - yb) ** 2) * vm.float()
            main = err2.sum() / vm.float().sum().clamp(min=1)
            # Aux: apply soft-label MSE only on rows where real label is NaN
            # and we have a cosmo-sac prediction for that IL
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
            err2 = ((pred - yf) ** 2) * valid.float()
            # Use MAIN loss only for early-stopping (not aux — aux on unlabeled is
            # intrinsically a moving target if COSMO is biased)
            tl = (err2.sum(0) / valid.float().sum(0).clamp(min=1)).mean().item()
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
        return m(torch.from_numpy(v4).to(device),
                 torch.from_numpy(morg).to(device),
                 torch.from_numpy(th).to(device),
                 torch.from_numpy(cp).to(device)).cpu().numpy()


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
    # Standardize soft labels to be in same scale as targets (z-score using
    # train real labels for that column)
    y_tr = tr["targets"].astype(np.float32)
    for idx, soft in [(IDX_GAMMA2, lg_tr), (IDX_G_E, ge_tr)]:
        real = y_tr[:, idx]
        mu = np.nanmean(real); sd = np.nanstd(real) + 1e-6
        soft -= mu; soft /= sd  # in-place
    # Same for val/te
    lg_va = (lg_va - np.nanmean(tr["targets"][:, IDX_GAMMA2])) / (np.nanstd(tr["targets"][:, IDX_GAMMA2]) + 1e-6)
    ge_va = (ge_va - np.nanmean(tr["targets"][:, IDX_G_E])) / (np.nanstd(tr["targets"][:, IDX_G_E]) + 1e-6)
    lg_te = (lg_te - np.nanmean(tr["targets"][:, IDX_GAMMA2])) / (np.nanstd(tr["targets"][:, IDX_GAMMA2]) + 1e-6)
    ge_te = (ge_te - np.nanmean(tr["targets"][:, IDX_G_E])) / (np.nanstd(tr["targets"][:, IDX_G_E]) + 1e-6)
    for a in (lg_va, ge_va, lg_te, ge_te):
        a[np.isnan(a)] = 0.0

    v4_tr, v4_va, v4_te = v4_base(tr), v4_base(va), v4_base(te)
    y_te = te["targets"].astype(np.float32)
    th_tr, th_va, th_te = tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]

    s1_r2s, s2_r2s = [], []
    for seed in range(args.n_seeds):
        print(f"\n[seed {seed}] Stage-1 (A5.4 = A2 + COSMO-SAC soft labels)...")
        s1 = train_stage1_softlabel(seed, v4_tr, m_tr, th_tr, cp_tr,
                                      lg_tr, ge_tr, hs_tr, y_tr,
                                      device, lambda_aux=args.lambda_aux,
                                      epochs=args.epochs)
        r = r2_per_prop(predict_stage1(s1, v4_te, m_te, th_te, cp_te, device), y_te)
        s1_r2s.append(r)
        print(f"  Stage-1 core7={r['avg_core7']:.4f}  lignin={r.get('lignin_wt', float('nan')):.4f}  "
              f"corr_gate={torch.sigmoid(s1.corr_gate).mean().item():.3f}")

        print(f"[seed {seed}] Stage-2 (hardfreeze + deep lignin + physchem)...")
        s2 = train_stage2_lignin(s1, v4_tr, m_tr, th_tr, cp_tr, p_tr, hp_tr, y_tr,
                                  device, seed=seed + 100, epochs=args.epochs)
        s2_pred = predict_stage2(s2, v4_te, m_te, th_te, cp_te, p_te, hp_te, device)
        r2 = r2_per_prop(s2_pred, y_te)
        s2_r2s.append(r2)
        print(f"  Stage-2 core7={r2['avg_core7']:.4f}  lignin={r2.get('lignin_wt', float('nan')):.4f}")

    tag = f"lambda{args.lambda_aux}"
    s1 = summarize(f"Stage1_A5_cosmosac_softlabels_{tag}", s1_r2s)
    s2 = summarize(f"Stage2_A5_cosmosac_deep_lignin_{tag}", s2_r2s)
    print(f"\n{'='*70}\nA5.4 COSMO-SAC SOFT LABELS SUMMARY\n{'='*70}")
    print(f"{'Stage':<50}{'core7':>10}{'std':>10}{'lignin':>10}")
    for r in [s1, s2]:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<50}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}{lig:>10.4f}")
    out = V5 / "results" / f"a5_cosmosac_softlabels_{tag}.json"
    json.dump([s1, s2], open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
