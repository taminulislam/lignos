"""Plan B — A2 + COSMO-SAC σ-profile reconstruction AUXILIARY LOSS.

Rationale: A5_cosmo (feature-as-input) showed that σ-profile features as a
passive input signal don't generate enough gradient to open the gate. Plan B
flips the direction: the model's existing gated Morgan+ChemProp representation
must PREDICT the σ-profile (reconstruction head), and the aux MSE loss
backprops through the main backbone. This gives σ-profile physics a direct
gradient pathway to shape the representation, which is exactly what we want
for OOD generalization: rows whose target labels are NaN (5,147 ILThermo
pre-training rows) STILL contribute gradient via the σ-profile reconstruction
auxiliary task.

Architecture:
  A2Head  +  aux_recon_head(nf+5 -> 20)   [parallel, added to forward()]
  Total loss = L_main (masked target MSE) + lambda * L_aux (σ-profile MSE on
               rows where has_cosmo=1)

Unlike A5_cosmo (feature branch), the aux head does NOT affect the core-7
prediction at inference — it only shapes the training gradient. So the
headline-reporting step uses the A2Head output unchanged; aux head is
discarded at eval.

Expected outcomes (per 2026-04-20 brainstorm):
  - core7 +0.01 to +0.02 (5147 unlabeled rows now contribute via aux loss)
  - OOD lift: σ-profile physics generalizes to new IL chemistries by DFT
  - Biggest impact on G_E, H_E, G_mix (mixing props most σ-profile-informed)
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
    predict_stage1 as predict_stage1_a2,
    predict_stage2 as predict_stage2_a2,
    train_stage2_lignin,
)

CACHE = V5 / "data" / "LignoIL_A1"
COSMO_BANK = V5 / "data" / "cosmo_sac_feat_bank.npz"
COSMO_DIM = 20
LAMBDA_AUX = 0.05  # weight on aux loss — tune {0.02, 0.05, 0.1}
A2_CKPT = V5 / "checkpoints" / "a2" / "stage1_best.pt"


class A2WithCosmoRecon(A2Head):
    """A2Head + a small head that RECONSTRUCTS σ-profile features from the
    gated Morgan+thermo representation. Training-only branch; inference ignores it.
    """
    def __init__(self, nf, n_props=8, chemprop_dim=40, cosmo_dim=COSMO_DIM):
        super().__init__(nf, n_props, chemprop_dim)
        # Aux head: [gated_morgan(nf) + thermo(5)] -> cosmo_dim
        self.aux_recon = nn.Sequential(
            nn.Linear(nf + 5, 64), nn.GELU(),
            nn.Linear(64, cosmo_dim),
        )

    def predict_cosmo(self, i, t):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        inp = torch.cat([g, tmp], -1)
        return self.aux_recon(inp)


def load_cosmo_bank():
    z = np.load(COSMO_BANK, allow_pickle=True)
    return dict(zip(z["smiles"], z["cosmo_feat"]))


def _canon(s):
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return s


def assemble_cosmo(smiles, bank):
    n = len(smiles)
    feats = np.zeros((n, COSMO_DIM), dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    for i, s in enumerate(smiles):
        cs = _canon(s)
        f = bank.get(cs)
        if f is None:
            f = bank.get(s)
        if f is not None:
            feats[i] = f
            mask[i] = 1.0
    print(f"[cosmo] {int(mask.sum())}/{n} rows covered by COSMO-SAC bank")
    return feats, mask


def _load_split(s):
    p_dft = CACHE / f"cached_{s}_dft.npz"
    p_std = CACHE / f"cached_{s}.npz"
    p = p_dft if p_dft.exists() else p_std
    print(f"[{s}] loading {p.name}")
    return {k: v for k, v in np.load(p, allow_pickle=True).items()}


def _standardize_cosmo(feats, mask):
    """Z-score per feature dim, over rows where mask==1. Zero rows stay zero."""
    cov = mask.astype(bool)
    mu = feats[cov].mean(0) if cov.sum() else np.zeros(feats.shape[1])
    sd = (feats[cov].std(0) + 1e-6) if cov.sum() else np.ones(feats.shape[1])
    out = (feats - mu) / sd
    out = out * mask[:, None]
    return out.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def train_stage1_aux(seed, v4, morg, th, cp, cosmo_z, hc, y, device,
                      lambda_aux=LAMBDA_AUX, epochs=300, patience=50,
                      warm_start=False, freeze_backbone=False):
    """Stage-1 trainer.

    - lambda_aux=0   → no aux loss (baseline diagnostic)
    - warm_start     → load A2 Stage-1 checkpoint before training
    - freeze_backbone → only aux_recon trains; main A2Head stays at loaded values
                        (useful to isolate: "does Stage-2 on frozen A2 + aux head
                        alone reproduce the 0.77 lignin?")
    """
    set_seed(seed)
    n_props = y.shape[1]
    m = A2WithCosmoRecon(morg.shape[1], n_props, chemprop_dim=cp.shape[1]).to(device)

    if warm_start and A2_CKPT.exists():
        ckpt = torch.load(A2_CKPT, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        miss, unex = m.load_state_dict(sd, strict=False)
        print(f"  warm-started from {A2_CKPT.name} (A2 seed={ckpt.get('seed')}); "
              f"{len(miss)} unmatched (aux_recon only), {len(unex)} unused")

    if freeze_backbone:
        for name, p in m.named_parameters():
            if not name.startswith("aux_recon"):
                p.requires_grad = False

    train_params = [p for p in m.parameters() if p.requires_grad]
    print(f"  trainable params: {sum(p.numel() for p in train_params)} "
          f"of {sum(p.numel() for p in m.parameters())}")

    opt = AdamW(train_params, lr=5e-4, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    ts = {k: torch.from_numpy(x).to(device) for k, x in
          dict(v=v4, i=morg, t=th, cp=cp, c=cosmo_z, hc=hc, y=y).items()}
    valid = ~torch.isnan(ts["y"]); yf = torch.nan_to_num(ts["y"], 0.0)

    ds = TensorDataset(*[ts[k].cpu() for k in ("v","i","t","cp","c","hc")],
                        yf.cpu(), valid.cpu())
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for ep in range(epochs):
        m.train()
        for vb, ib, tb, cpb, cb, hcb, yb, vm in loader:
            vb, ib, tb, cpb, cb, hcb, yb, vm = [x.to(device)
                for x in (vb, ib, tb, cpb, cb, hcb, yb, vm)]
            pred = m(vb, ib, tb, cpb)
            err2 = ((pred - yb) ** 2) * vm.float()
            main = err2.sum() / vm.float().sum().clamp(min=1)
            # Aux loss: reconstruct σ-profile from gated Morgan+thermo
            recon = m.predict_cosmo(ib, tb)
            aux_err = ((recon - cb) ** 2).mean(-1) * hcb  # (B,)
            aux = aux_err.sum() / hcb.sum().clamp(min=1)
            loss = main + lambda_aux * aux
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pred = m(ts["v"], ts["i"], ts["t"], ts["cp"])
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
    ap.add_argument("--lambda-aux", type=float, default=LAMBDA_AUX)
    ap.add_argument("--warm-start", action="store_true",
                    help="Load A2 Stage-1 checkpoint before training.")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="Freeze A2Head; only aux_recon trainable.")
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
    _, p_va = preprocess_physchem(tr["physchem_feat"], tr["has_physchem"],
                                   va["physchem_feat"], va["has_physchem"])
    hp_tr = tr["has_physchem"].astype(np.float32)
    hp_va = va["has_physchem"].astype(np.float32)
    hp_te = te["has_physchem"].astype(np.float32)

    bank = load_cosmo_bank()
    c_tr, hc_tr = assemble_cosmo(tr["smiles"], bank)
    c_va, hc_va = assemble_cosmo(va["smiles"], bank)
    c_te, hc_te = assemble_cosmo(te["smiles"], bank)
    # Standardize using train-covered rows as the reference
    c_tr_z, mu, sd = _standardize_cosmo(c_tr, hc_tr)
    c_va_z = ((c_va - mu) / sd) * hc_va[:, None]
    c_te_z = ((c_te - mu) / sd) * hc_te[:, None]
    c_tr_z, c_va_z, c_te_z = [x.astype(np.float32) for x in (c_tr_z, c_va_z, c_te_z)]

    v4_tr, v4_va, v4_te = v4_base(tr), v4_base(va), v4_base(te)
    y_tr, y_va, y_te = [x["targets"].astype(np.float32) for x in (tr, va, te)]
    th_tr, th_va, th_te = tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]

    stage1_r2s, stage2_r2s = [], []
    for seed in range(args.n_seeds):
        print(f"\n[seed {seed}] Stage-1 (A2 + σ-profile aux loss λ={args.lambda_aux})...")
        s1 = train_stage1_aux(seed, v4_tr, m_tr, th_tr, cp_tr, c_tr_z, hc_tr, y_tr,
                                device, lambda_aux=args.lambda_aux, epochs=args.epochs,
                                warm_start=args.warm_start,
                                freeze_backbone=args.freeze_backbone)
        # Stage-1 eval uses only the main forward (drops aux_recon)
        s1_pred = predict_stage1_a2(s1, v4_te, m_te, th_te, cp_te, device)
        r = r2_per_prop(s1_pred, y_te)
        stage1_r2s.append(r)
        print(f"  Stage-1 core7={r['avg_core7']:.4f}  lignin={r.get('lignin_wt', float('nan')):.4f}")

        print(f"[seed {seed}] Stage-2 (hardfreeze + deep lignin + physchem)...")
        # Reuse A2 Stage-2 trainer; backbone is our A2WithCosmoRecon (extra aux
        # head is ignored — its params aren't touched by Stage-2 optimizer).
        s2 = train_stage2_lignin(s1, v4_tr, m_tr, th_tr, cp_tr, p_tr, hp_tr, y_tr,
                                  device, seed=seed + 100, epochs=args.epochs)
        s2_pred = predict_stage2_a2(s2, v4_te, m_te, th_te, cp_te, p_te, hp_te, device)
        r2 = r2_per_prop(s2_pred, y_te)
        stage2_r2s.append(r2)
        print(f"  Stage-2 core7={r2['avg_core7']:.4f}  lignin={r2.get('lignin_wt', float('nan')):.4f}")

    tag = f"lambda{args.lambda_aux}"
    if args.warm_start: tag += "_ws"
    if args.freeze_backbone: tag += "_fz"
    s1 = summarize(f"Stage1_A2_cosmo_auxloss_{tag}", stage1_r2s)
    s2 = summarize(f"Stage2_A2_aux_deep_lignin_{tag}", stage2_r2s)
    print(f"\n{'='*70}\nA2 + σ-profile AUX LOSS (Plan B) SUMMARY\n{'='*70}")
    print(f"{'Stage':<45}{'core7':>10}{'std':>10}{'lignin':>10}")
    for r in [s1, s2]:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<45}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}{lig:>10.4f}")
    out = V5 / "results" / f"a2_cosmo_auxloss_{tag}.json"
    json.dump([s1, s2], open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
