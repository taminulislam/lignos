"""Tier 2 ablation — push lignin past 0.706 baseline.

Two lignin-head techniques, in a 2x2 factorial:

  (5) Feed the model's OWN fused core7 predictions (μ_fused[:, :7]) into the
      deep lignin head as an auxiliary latent summary. Distinct from v2 A'/AB'
      which fed raw process-thermo dims — those were noise; these are
      model-learned per-IL chemistry vectors.
  (6) Gaussian jitter on thermo (dim 0..4) + physchem at training time
      (σ=0.05). A small-sample regularizer — SMILES randomization doesn't
      apply here because Morgan FP is rotation-invariant on the molecule
      graph, so all randomized SMILES map to the same features.

We run 4 configs × 10 seeds:
  - baseline (no #5, no #6)  — reference sanity-check
  - #5 only (pred_mu)
  - #6 only (aug_noise=0.05)
  - #5 + #6 combined

Each config reports core7 (pre- and post-Stage-2) and lignin R².

Output: results/a5_bma_tier2.json
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
from train_a2_two_stage import build_chemprop_40d, preprocess_physchem, v4_base
from train_a5_bma_pipeline import (
    A5_BMA_Specialist, A5_BMA_Router, train_router,
    _load_split, _standardize, _assemble_bank,
    FRAME_DIM, COSMO_DIM, BMA_DIR,
    VIT_BANK, COSMO_BANK,
)


class A5_BMA_Stage2_Tier2(nn.Module):
    """Subclass adding `use_pred_mu`: when True, deep_lignin ctx also receives
    μ_fused[:, :7] (the model's own core7 predictions for each row).

    Backbone identical to A5_BMA_Stage2; only the head differs."""

    def __init__(self, specialists_dict, router, nf, n_props=8, physchem_dim=12,
                  use_pred_mu=False):
        super().__init__()
        self.spec_A = specialists_dict["A"]
        self.spec_B = specialists_dict["B"]
        self.spec_C = specialists_dict["C"]
        self.router = router
        for m in (self.spec_A, self.spec_B, self.spec_C, self.router):
            for p in m.parameters():
                p.requires_grad = False

        self.gate_fn = self.spec_A.gate
        self.alpha_lignin = nn.Parameter(self.spec_A.alphas.data[7].clone())

        self.use_pred_mu = use_pred_mu
        head_in = nf + 5 + physchem_dim + 1 + (7 if use_pred_mu else 0)
        self.deep_lignin = nn.Sequential(
            nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        with torch.no_grad():
            self.deep_lignin[-1].weight.mul_(0.01); self.deep_lignin[-1].bias.zero_()

    def _fused(self, v, i, t, chemprop, surface, vit, cos, hs, hv, hc):
        mus, lvs = [], []
        with torch.no_grad():
            for m in (self.spec_A, self.spec_B, self.spec_C):
                mu, lv = m.forward_with_lv(v, i, t, chemprop,
                                             surface=surface, vit=vit, cos=cos,
                                             has_surf=hs, has_vit=hv, has_cos=hc)
                mus.append(mu); lvs.append(lv)
            mu_s = torch.stack(mus, dim=1)
            lv_s = torch.stack(lvs, dim=1)
            w = self.router(chemprop, surface, t, lv_s)
            mu_f = (w * mu_s).sum(dim=1)
            prec = torch.exp(-lv_s).sum(dim=1) + 1e-8
            lv_f = -torch.log(prec)
        return mu_f, lv_f

    def forward(self, v, i, t, chemprop, surface, vit, cos, hs, hv, hc,
                phys, has_phys):
        mu_f, lv_f = self._fused(v, i, t, chemprop, surface, vit, cos, hs, hv, hc)
        tmp = t[:, :5]
        g = i * self.gate_fn(tmp)
        hp = has_phys.float().unsqueeze(-1) if has_phys.ndim == 1 else has_phys.float()
        parts = [g, tmp, phys, hp]
        if self.use_pred_mu:
            # Detach to prevent gradients flowing back through frozen backbone
            parts.append(mu_f[:, :7].detach())
        ctx = torch.cat(parts, -1)
        res_lignin = self.deep_lignin(ctx).squeeze(-1)
        out = mu_f.clone()
        out[:, 7] = v[:, 7] + torch.sigmoid(self.alpha_lignin) * res_lignin
        return out, lv_f


def train_stage2_tier2(model, feats, y, device, seed, epochs=300, patience=50,
                         aug_noise=0.0):
    """Trains ONLY deep_lignin + alpha_lignin. Optional Gaussian noise on
    thermo (dims 0..4) and physchem at each training batch."""
    set_seed(seed)
    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW([{"params": model.deep_lignin.parameters(), "weight_decay": 1e-2},
                  {"params": [model.alpha_lignin], "weight_decay": 0.0}], lr=1e-3)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    keys = ("v","i","t","cp","surf","vit","cos","hs","hv","hc","p","hp")
    full_keys = ("v4","morg","thermo","chemprop","surface","vit","cos",
                  "has_surf","has_vit","has_cos","physchem","has_physchem")
    ts = {k: torch.from_numpy(feats[full]).to(device) for k, full in zip(keys, full_keys)}
    ts["y"] = torch.from_numpy(y).to(device)

    ds = TensorDataset(*[ts[k].cpu() for k in ("v","i","t","cp","surf","vit","cos",
                                                   "hs","hv","hc","p","hp","y")])
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best_tl, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            vb, ib, tb, cpb, sub, vib, cob, hsb, hvb, hcb, pb, hpb, yb = batch
            # --- Tier 2 #6: input jitter ---
            if aug_noise > 0:
                tb = tb.clone()
                tb[:, :5] = tb[:, :5] + aug_noise * torch.randn_like(tb[:, :5])
                pb = pb + aug_noise * torch.randn_like(pb) * hpb.unsqueeze(-1)
            pred, _ = model(vb, ib, tb, cpb, sub, vib, cob, hsb, hvb, hcb, pb, hpb)
            lg = ~torch.isnan(yb[:, 7])
            if lg.sum() == 0: continue
            loss = ((pred[lg, 7] - yb[lg, 7].nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            pred, _ = model(ts["v"], ts["i"], ts["t"], ts["cp"],
                              ts["surf"], ts["vit"], ts["cos"],
                              ts["hs"], ts["hv"], ts["hc"], ts["p"], ts["hp"])
            lg = ~torch.isnan(ts["y"][:, 7])
            tl = ((pred[lg, 7] - ts["y"][lg, 7].nan_to_num(0)) ** 2).mean().item() if lg.any() else float("inf")
        if np.isfinite(tl) and tl < best_tl:
            best_tl = tl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, feats, device):
    keys = ("v4","morg","thermo","chemprop","surface","vit","cos",
             "has_surf","has_vit","has_cos","physchem","has_physchem")
    tens = [torch.from_numpy(feats[k]).to(device) for k in keys]
    with torch.no_grad():
        mu, lv = model(*tens)
    return mu.cpu().numpy(), lv.cpu().numpy()


def r2_per(pred, y):
    r = {}
    for i, p in enumerate(PROPS):
        v = ~np.isnan(y[:, i])
        if v.sum() < 2: continue
        yk, pk = y[v, i], pred[v, i]
        ss_res = ((yk - pk) ** 2).sum()
        ss_tot = ((yk - yk.mean()) ** 2).sum() + 1e-12
        r[p] = float(1.0 - ss_res / ss_tot)
    c = [r[p] for p in PROPS[:7] if p in r and np.isfinite(r[p])]
    r["avg_core7"] = float(np.mean(c)) if c else float("nan")
    return r


def gated_r2(pred, y, logvar, quantile=0.5):
    sigma = np.exp(0.5 * logvar).mean(axis=-1)
    thr = np.quantile(sigma, quantile)
    keep = sigma <= thr
    r = {}
    for i, p in enumerate(PROPS):
        v = ~np.isnan(y[:, i]) & keep
        if v.sum() < 2: continue
        yk, pk = y[v, i], pred[v, i]
        ss_res = ((yk - pk) ** 2).sum()
        ss_tot = ((yk - yk.mean()) ** 2).sum() + 1e-12
        r[p] = float(1.0 - ss_res / ss_tot)
    c = [r[p] for p in PROPS[:7] if p in r and np.isfinite(r[p])]
    r["avg_core7"] = float(np.mean(c)) if c else float("nan")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--aug-noise", type=float, default=0.05,
                     help="Gaussian std for thermo+physchem jitter (Tier 2 #6).")
    ap.add_argument("--router-mode", choices=["mlp", "scalar"], default="scalar")
    ap.add_argument("--configs", nargs="+", default=None,
                     help="Subset of configs to run: tier2_mu{0,1}_aug{0,1}. "
                          "Default = all four.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                     help="Explicit seed indices to run. If set, overrides "
                          "--n-seeds. Useful for resuming missing seeds.")
    ap.add_argument("--save-rowpreds", type=str, default=None,
                     help="If set, dump per-seed test-set per-row lignin "
                          "predictions to this npz path (keys: lignin_preds "
                          "[n_seeds, n_test], y_true [n_test], seeds [n_seeds]).")
    args = ap.parse_args()
    seed_iter = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  n_seeds={args.n_seeds}  aug_noise={args.aug_noise}")

    # ----- Data (matches pipeline exactly) -----
    tr, va, te = _load_split("train"), _load_split("val"), _load_split("test")
    pca_m = PCA(40).fit(tr["morgan_fp"])
    m_tr, m_va, m_te = [pca_m.transform(x["morgan_fp"]).astype(np.float32)
                         for x in (tr, va, te)]
    cp_tr, cp_te = build_chemprop_40d(tr["chemprop_fp"], te["chemprop_fp"])
    _, cp_va = build_chemprop_40d(tr["chemprop_fp"], va["chemprop_fp"])

    surf_tr = tr["surface_fp"].astype(np.float32)
    surf_va = va["surface_fp"].astype(np.float32); surf_te = te["surface_fp"].astype(np.float32)
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

    phys_tr, phys_te = preprocess_physchem(
        tr["physchem_feat"], tr["has_physchem"],
        te["physchem_feat"], te["has_physchem"])
    _, phys_va = preprocess_physchem(
        tr["physchem_feat"], tr["has_physchem"],
        va["physchem_feat"], va["has_physchem"])
    hp_tr = tr["has_physchem"].astype(np.float32)
    hp_va = va["has_physchem"].astype(np.float32)
    hp_te = te["has_physchem"].astype(np.float32)

    v4_tr, v4_va, v4_te = v4_base(tr), v4_base(va), v4_base(te)
    y_tr, y_va, y_te = tr["targets"].astype(np.float32), va["targets"].astype(np.float32), te["targets"].astype(np.float32)
    th_tr, th_va, th_te = tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]

    def _feats(v4, morg, thermo, cp, surf, vit, cos, hs, hv, hc, phys, hp):
        return {"v4": v4.astype(np.float32), "morg": morg.astype(np.float32),
                 "thermo": thermo.astype(np.float32), "chemprop": cp.astype(np.float32),
                 "surface": surf.astype(np.float32), "vit": vit.astype(np.float32),
                 "cos": cos.astype(np.float32),
                 "has_surf": hs.astype(np.float32), "has_vit": hv.astype(np.float32),
                 "has_cos": hc.astype(np.float32),
                 "physchem": phys.astype(np.float32), "has_physchem": hp.astype(np.float32)}
    feats_tr = _feats(v4_tr, m_tr, th_tr, cp_tr, surf_tr, vit_tr, cos_tr, hs_tr, hv_tr, hc_tr, phys_tr, hp_tr)
    feats_va = _feats(v4_va, m_va, th_va, cp_va, surf_va, vit_va, cos_va, hs_va, hv_va, hc_va, phys_va, hp_va)
    feats_te = _feats(v4_te, m_te, th_te, cp_te, surf_te, vit_te, cos_te, hs_te, hv_te, hc_te, phys_te, hp_te)

    # ----- Load cached specialists + train router -----
    print("\nLoading cached specialists A, B, C...")
    spec = {}
    for kind in ("A", "B", "C"):
        ck = torch.load(BMA_DIR / f"specialist_{kind}.pt", map_location=device, weights_only=False)
        m = A5_BMA_Specialist(kind, m_tr.shape[1], 8, chemprop_dim=cp_tr.shape[1]).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        spec[kind] = m
        print(f"  [Sp {kind}] core7={ck.get('test_core7', float('nan')):.4f}")

    print("\nTraining scalar router...")
    router = train_router([spec["A"], spec["B"], spec["C"]],
                            feats_tr, y_tr, device,
                            epochs=args.epochs // 2, mode=args.router_mode)
    router.eval()

    # ----- 4-config x N-seed sweep -----
    print(f"\n{'='*70}\nTier 2 sweep: {{#5, #6}} × {{on, off}} × seeds={seed_iter}\n{'='*70}")
    all_results = {}
    nf = 40
    for use_mu in (False, True):
        for use_aug in (False, True):
            name = f"tier2_mu{int(use_mu)}_aug{int(use_aug)}"
            if args.configs is not None and name not in args.configs:
                print(f"\n--- Skipping {name} (not in --configs) ---")
                continue
            print(f"\n--- Config {name} (use_pred_mu={use_mu}, aug_noise={args.aug_noise if use_aug else 0.0}) ---")
            core7_list, lignin_list, g50_list = [], [], []
            rowpreds_list = []  # per-seed (n_test,) lignin predictions
            for seed in seed_iter:
                model = A5_BMA_Stage2_Tier2(spec, router, nf, use_pred_mu=use_mu).to(device)
                model = train_stage2_tier2(model, feats_tr, y_tr, device, seed,
                                              epochs=args.epochs,
                                              aug_noise=args.aug_noise if use_aug else 0.0)
                pred, lv = predict(model, feats_te, device)
                r = r2_per(pred, y_te)
                g50 = gated_r2(pred, y_te, lv, 0.5)
                core7_list.append(r["avg_core7"])
                lignin_list.append(r.get("lignin_wt", float("nan")))
                g50_list.append(g50["avg_core7"])
                rowpreds_list.append(pred[:, 7].astype(np.float32))  # lignin column
                print(f"  seed {seed}: core7={r['avg_core7']:.4f}  "
                       f"lignin={r.get('lignin_wt', float('nan')):.4f}  g50={g50['avg_core7']:.4f}")
            if args.save_rowpreds is not None and name == "tier2_mu1_aug1":
                import os
                os.makedirs(os.path.dirname(args.save_rowpreds) or ".", exist_ok=True)
                np.savez(args.save_rowpreds,
                          lignin_preds=np.stack(rowpreds_list),  # (n_seeds, n_test)
                          y_true=y_te[:, 7].astype(np.float32),
                          seeds=np.array(seed_iter, dtype=np.int32),
                          config=name)
                print(f"  [saved per-row lignin preds to {args.save_rowpreds}]")
            all_results[name] = {
                "use_pred_mu": use_mu,
                "aug_noise": args.aug_noise if use_aug else 0.0,
                "core7_mean": float(np.mean(core7_list)),
                "core7_std": float(np.std(core7_list)),
                "lignin_mean": float(np.nanmean(lignin_list)),
                "lignin_std": float(np.nanstd(lignin_list)),
                "g50_mean": float(np.mean(g50_list)),
                "g50_std": float(np.std(g50_list)),
                "lignin_per_seed": [float(x) for x in lignin_list],
            }
            print(f"  → mean: core7={all_results[name]['core7_mean']:.4f}  "
                   f"lignin={all_results[name]['lignin_mean']:.4f}±{all_results[name]['lignin_std']:.4f}  "
                   f"g50={all_results[name]['g50_mean']:.4f}")

    # ----- Summary ranked by lignin -----
    print(f"\n{'='*70}\nTier 2 summary (ranked by lignin R²)\n{'='*70}")
    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["lignin_mean"], reverse=True)
    print(f"  {'config':<22}{'core7':>10}{'lignin':>12}{'std':>8}{'g50':>10}")
    for name, r in ranked:
        print(f"  {name:<22}{r['core7_mean']:>10.4f}{r['lignin_mean']:>12.4f}"
               f"{r['lignin_std']:>8.4f}{r['g50_mean']:>10.4f}")

    suffix = "" if args.configs is None else "_" + "+".join(args.configs)
    out = V5 / "results" / f"a5_bma_tier2{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"configs": all_results,
                    "baseline": {"core7": 0.834, "lignin": 0.697, "g50": 0.9344}},
                   f, indent=2, default=float)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
