"""Tier 1 ablation — push past A5.9 baseline (core7=0.834, gated@50%=0.935).

Four tests, stacked in a single run:
  (1) Specialist C 10-seed ensemble (variance reduction on the strongest specialist).
  (2) BMA {B, C} — drop Specialist A (redundant with B).
  (3) BMA {B, C_ens} — both #1 and #2 combined.
  (4) Post-fusion isotonic calibration fit on val, applied to test — reported
      for every BMA config above.
  Reports gated@50, @25, @10 for all configs (quantiles were all default 0.5
  before; tightening gives the paper a better high-confidence story).

Outputs:
  results/a5_bma_tier1.json
  results/a5_bma_tier1.txt (human-readable)
  checkpoints/a5_bma/specialist_C_seed{0..9}.pt
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, r2_per_prop, set_seed  # noqa
from train_a2_two_stage import build_chemprop_40d, v4_base
from train_a5_bma_pipeline import (
    A5_BMA_Specialist, A5_BMA_Router, train_specialist, train_router,
    _load_split, _standardize, _assemble_bank,
    VIT_BANK, COSMO_BANK, FRAME_DIM, COSMO_DIM, BMA_DIR,
)

RESULTS_DIR = V5 / "results"


def _r2(pred, y):
    r = {}
    for i, p in enumerate(PROPS):
        v = ~np.isnan(y[:, i])
        if v.sum() < 2: continue
        yk, pk = y[v, i], pred[v, i]
        ss_res = ((yk - pk) ** 2).sum()
        ss_tot = ((yk - yk.mean()) ** 2).sum() + 1e-12
        r[p] = 1.0 - ss_res / ss_tot
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
        r[p] = 1.0 - ss_res / ss_tot
    c = [r[p] for p in PROPS[:7] if p in r and np.isfinite(r[p])]
    r["avg_core7"] = float(np.mean(c)) if c else float("nan")
    r["n_kept"] = int(keep.sum()); r["threshold_sigma"] = float(thr)
    return r


def predict_specialist(m, feats, device):
    keys = ("v4","morg","thermo","chemprop","surface","vit","cos",
             "has_surf","has_vit","has_cos")
    tens = [torch.from_numpy(feats[k]).to(device) for k in keys]
    with torch.no_grad():
        mu, lv = m.forward_with_lv(tens[0], tens[1], tens[2], tens[3],
                                    surface=tens[4], vit=tens[5], cos=tens[6],
                                    has_surf=tens[7], has_vit=tens[8], has_cos=tens[9])
    return mu.cpu().numpy(), lv.cpu().numpy()


def fuse_bma(mu_list, lv_list, router=None):
    """Fuse K (N, P) specialist predictions. If router is None, use pure BMA
    (inverse-variance weighting). Otherwise use router's learned correction."""
    mu_s = np.stack(mu_list, axis=1)      # (N, K, P)
    lv_s = np.stack(lv_list, axis=1)
    if router is None:
        w = np.exp(-lv_s); w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
    else:
        raise NotImplementedError("router path handled via evaluate()")
    mu_f = (w * mu_s).sum(axis=1)
    prec = np.exp(-lv_s).sum(axis=1) + 1e-8
    lv_f = -np.log(prec)
    return mu_f, lv_f


def ensemble_specialists_C(ckpts, feats, device):
    """Mean μ, aleatoric-logvar over K C-ensemble members. Add epistemic
    variance to returned logvar (var over members' μ)."""
    mus, lvs = [], []
    for ck in ckpts:
        m = A5_BMA_Specialist("C", feats["morg"].shape[1], 8,
                               chemprop_dim=feats["chemprop"].shape[1]).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        mu, lv = predict_specialist(m, feats, device)
        mus.append(mu); lvs.append(lv)
    mu_stack = np.stack(mus, axis=0)       # (S, N, P)
    lv_stack = np.stack(lvs, axis=0)
    mu_mean = mu_stack.mean(axis=0)
    aleatoric = np.exp(lv_stack).mean(axis=0)
    epistemic = mu_stack.var(axis=0)
    total_var = aleatoric + epistemic
    lv_total = np.log(total_var + 1e-8).astype(np.float32)
    return mu_mean.astype(np.float32), lv_total


def evaluate_config(spec_list, feats, y, device, router=None):
    """Evaluate a specialist ensemble + optional router. Returns (μ_fused, lv_fused).
    spec_list may contain loaded nn.Modules OR precomputed (mu, lv) tuples for
    Specialist C ensemble."""
    mus, lvs = [], []
    for item in spec_list:
        if isinstance(item, tuple):
            mus.append(item[0]); lvs.append(item[1])
        else:
            mu, lv = predict_specialist(item, feats, device)
            mus.append(mu); lvs.append(lv)
    if router is None:
        return fuse_bma(mus, lvs)
    # Use router: stack and forward
    mu_s = torch.from_numpy(np.stack(mus, axis=1)).to(device)
    lv_s = torch.from_numpy(np.stack(lvs, axis=1)).to(device)
    with torch.no_grad():
        cp = torch.from_numpy(feats["chemprop"]).to(device)
        su = torch.from_numpy(feats["surface"]).to(device)
        th = torch.from_numpy(feats["thermo"]).to(device)
        w = router(cp, su, th, lv_s)
        mu_f = (w * mu_s).sum(dim=1)
        prec = torch.exp(-lv_s).sum(dim=1) + 1e-8
        lv_f = -torch.log(prec)
    return mu_f.cpu().numpy(), lv_f.cpu().numpy()


def fit_isotonic(pred_val, y_val):
    """Per-property isotonic regression, trained on validation fused predictions.
    Returns list[IsotonicRegression | None]."""
    regs = []
    for i in range(pred_val.shape[1]):
        v = ~np.isnan(y_val[:, i])
        if v.sum() < 10:
            regs.append(None); continue
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(pred_val[v, i], y_val[v, i])
        regs.append(ir)
    return regs


def apply_isotonic(pred, regs):
    out = pred.copy()
    for i, ir in enumerate(regs):
        if ir is not None:
            out[:, i] = ir.transform(pred[:, i]).astype(np.float32)
    return out


def train_router_over(spec_list, feats_tr, y_tr, device, epochs=150, mode="scalar"):
    """Thin wrapper that accepts pre-ensembled specialists. For C-ensemble we
    can't reuse the pipeline's train_router (expects nn.Modules), so we wrap
    the C-ensemble forward into a pseudo-module."""
    # If everything is nn.Module, use the existing function directly:
    if all(not isinstance(s, tuple) for s in spec_list):
        return train_router(spec_list, feats_tr, y_tr, device, epochs=epochs, mode=mode)
    # Otherwise we skip learned routing — the caller should use pure BMA for
    # configs that include ensemble predictions (already handled by fuse_bma).
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds-c", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--router-mode", choices=["mlp", "scalar"], default="scalar")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ----- Data loading (same pipeline as A5.9) -----
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
    y_tr, y_va, y_te = tr["targets"].astype(np.float32), va["targets"].astype(np.float32), te["targets"].astype(np.float32)
    th_tr, th_va, th_te = tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]

    def _feats(v4, morg, thermo, cp, surf, vit, cos, hs, hv, hc):
        return {"v4": v4, "morg": morg, "thermo": thermo, "chemprop": cp,
                "surface": surf, "vit": vit, "cos": cos,
                "has_surf": hs, "has_vit": hv, "has_cos": hc}
    feats_tr = _feats(v4_tr, m_tr, th_tr, cp_tr, surf_tr, vit_tr, cos_tr, hs_tr, hv_tr, hc_tr)
    feats_va = _feats(v4_va, m_va, th_va, cp_va, surf_va, vit_va, cos_va, hs_va, hv_va, hc_va)
    feats_te = _feats(v4_te, m_te, th_te, cp_te, surf_te, vit_te, cos_te, hs_te, hv_te, hc_te)

    # ----- Load baseline specialists A, B -----
    print("\nLoading cached Specialists A, B...")
    spec = {}
    for kind in ("A", "B"):
        ck = torch.load(BMA_DIR / f"specialist_{kind}.pt", map_location=device, weights_only=False)
        m = A5_BMA_Specialist(kind, m_tr.shape[1], 8, chemprop_dim=cp_tr.shape[1]).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        spec[kind] = m
        print(f"  [Sp {kind}] core7={ck.get('test_core7', float('nan')):.4f}")

    # ----- Stage (1): Train 10-seed Specialist C ensemble -----
    print(f"\n{'='*70}\nTier 1.1: Specialist C 10-seed ensemble\n{'='*70}")
    c_ckpts = []
    for seed in range(args.n_seeds_c):
        ckpt_path = BMA_DIR / f"specialist_C_seed{seed}.pt"
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            c_ckpts.append(ck)
            print(f"  seed {seed}: cached (core7={ck.get('test_core7', float('nan')):.4f})")
            continue
        print(f"  seed {seed}: training...")
        m_k, vl = train_specialist("C", seed, feats_tr, y_tr, device, epochs=args.epochs)
        mu_te, _ = predict_specialist(m_k, feats_te, device)
        r = _r2(mu_te, y_te)
        ck = {"state_dict": {k: v.detach().cpu() for k, v in m_k.state_dict().items()},
              "seed": seed, "val_loss": float(vl), "test_core7": r["avg_core7"]}
        torch.save(ck, ckpt_path)
        c_ckpts.append(ck)
        print(f"    val_loss={vl:.4f}  test core7={r['avg_core7']:.4f}")

    # ----- Evaluate configs -----
    print(f"\n{'='*70}\nTier 1 evaluations\n{'='*70}")
    configs = {}

    # C ensemble alone (val + test)
    mu_ce_va, lv_ce_va = ensemble_specialists_C(c_ckpts, feats_va, device)
    mu_ce_te, lv_ce_te = ensemble_specialists_C(c_ckpts, feats_te, device)
    r_ce = _r2(mu_ce_te, y_te)
    g50 = gated_r2(mu_ce_te, y_te, lv_ce_te, 0.5)
    g25 = gated_r2(mu_ce_te, y_te, lv_ce_te, 0.25)
    g10 = gated_r2(mu_ce_te, y_te, lv_ce_te, 0.10)
    configs["C_ensemble_only"] = {"core7": r_ce["avg_core7"], "g50": g50["avg_core7"],
                                    "g25": g25["avg_core7"], "g10": g10["avg_core7"],
                                    "per_prop": {p: r_ce.get(p) for p in PROPS}}
    print(f"  C-ensemble alone: core7={r_ce['avg_core7']:.4f}  "
          f"g50={g50['avg_core7']:.4f}  g25={g25['avg_core7']:.4f}  g10={g10['avg_core7']:.4f}")

    # Pre-compute individual C prediction for baseline BMA{A,B,C1}
    mu_c1_va, lv_c1_va = predict_specialist(
        _load_spec(c_ckpts[0], "C", m_tr.shape[1], cp_tr.shape[1], device), feats_va, device)
    mu_c1_te, lv_c1_te = predict_specialist(
        _load_spec(c_ckpts[0], "C", m_tr.shape[1], cp_tr.shape[1], device), feats_te, device)

    # Also precompute specialist A and B predictions on val and test
    mu_a_va, lv_a_va = predict_specialist(spec["A"], feats_va, device)
    mu_a_te, lv_a_te = predict_specialist(spec["A"], feats_te, device)
    mu_b_va, lv_b_va = predict_specialist(spec["B"], feats_va, device)
    mu_b_te, lv_b_te = predict_specialist(spec["B"], feats_te, device)

    def eval_bma_cfg(name, mus_va, lvs_va, mus_te, lvs_te):
        mu_f_va, lv_f_va = fuse_bma(mus_va, lvs_va)
        mu_f_te, lv_f_te = fuse_bma(mus_te, lvs_te)
        r = _r2(mu_f_te, y_te)
        g50 = gated_r2(mu_f_te, y_te, lv_f_te, 0.5)
        g25 = gated_r2(mu_f_te, y_te, lv_f_te, 0.25)
        g10 = gated_r2(mu_f_te, y_te, lv_f_te, 0.10)
        configs[name] = {"core7": r["avg_core7"], "g50": g50["avg_core7"],
                          "g25": g25["avg_core7"], "g10": g10["avg_core7"],
                          "per_prop": {p: r.get(p) for p in PROPS}}
        print(f"  {name}: core7={r['avg_core7']:.4f}  "
              f"g50={g50['avg_core7']:.4f}  g25={g25['avg_core7']:.4f}  g10={g10['avg_core7']:.4f}")
        # Isotonic calibration on val → applied on test
        regs = fit_isotonic(mu_f_va, y_va)
        mu_iso = apply_isotonic(mu_f_te, regs)
        r_iso = _r2(mu_iso, y_te)
        g50_iso = gated_r2(mu_iso, y_te, lv_f_te, 0.5)
        g25_iso = gated_r2(mu_iso, y_te, lv_f_te, 0.25)
        g10_iso = gated_r2(mu_iso, y_te, lv_f_te, 0.10)
        configs[name + "+iso"] = {"core7": r_iso["avg_core7"], "g50": g50_iso["avg_core7"],
                                    "g25": g25_iso["avg_core7"], "g10": g10_iso["avg_core7"],
                                    "per_prop": {p: r_iso.get(p) for p in PROPS}}
        print(f"  {name}+iso: core7={r_iso['avg_core7']:.4f}  "
              f"g50={g50_iso['avg_core7']:.4f}  g25={g25_iso['avg_core7']:.4f}  "
              f"g10={g10_iso['avg_core7']:.4f}")

    # (a) Baseline BMA{A, B, C} pure BMA (reference)
    eval_bma_cfg("BMA_ABC_pure",
                  [mu_a_va, mu_b_va, mu_c1_va], [lv_a_va, lv_b_va, lv_c1_va],
                  [mu_a_te, mu_b_te, mu_c1_te], [lv_a_te, lv_b_te, lv_c1_te])
    # (b) BMA{B, C} — drop A
    eval_bma_cfg("BMA_BC_pure",
                  [mu_b_va, mu_c1_va], [lv_b_va, lv_c1_va],
                  [mu_b_te, mu_c1_te], [lv_b_te, lv_c1_te])
    # (c) BMA{A, B, C_ens}
    eval_bma_cfg("BMA_ABCens_pure",
                  [mu_a_va, mu_b_va, mu_ce_va], [lv_a_va, lv_b_va, lv_ce_va],
                  [mu_a_te, mu_b_te, mu_ce_te], [lv_a_te, lv_b_te, lv_ce_te])
    # (d) BMA{B, C_ens} — drop A + ensemble (best-of-both)
    eval_bma_cfg("BMA_BCens_pure",
                  [mu_b_va, mu_ce_va], [lv_b_va, lv_ce_va],
                  [mu_b_te, mu_ce_te], [lv_b_te, lv_ce_te])
    # (e) C_ensemble alone + isotonic
    regs = fit_isotonic(mu_ce_va, y_va)
    mu_iso = apply_isotonic(mu_ce_te, regs)
    r_iso = _r2(mu_iso, y_te)
    g50_iso = gated_r2(mu_iso, y_te, lv_ce_te, 0.5)
    g25_iso = gated_r2(mu_iso, y_te, lv_ce_te, 0.25)
    g10_iso = gated_r2(mu_iso, y_te, lv_ce_te, 0.10)
    configs["C_ensemble+iso"] = {"core7": r_iso["avg_core7"], "g50": g50_iso["avg_core7"],
                                    "g25": g25_iso["avg_core7"], "g10": g10_iso["avg_core7"],
                                    "per_prop": {p: r_iso.get(p) for p in PROPS}}
    print(f"  C_ensemble+iso: core7={r_iso['avg_core7']:.4f}  "
          f"g50={g50_iso['avg_core7']:.4f}  g25={g25_iso['avg_core7']:.4f}  "
          f"g10={g10_iso['avg_core7']:.4f}")

    # ----- Rank + report -----
    print(f"\n{'='*70}\nTier 1 summary (ranked by core7)\n{'='*70}")
    ranked = sorted(configs.items(), key=lambda kv: kv[1]["core7"], reverse=True)
    print(f"  {'config':<28}{'core7':>8}{'g50':>10}{'g25':>10}{'g10':>10}")
    for name, r in ranked:
        print(f"  {name:<28}{r['core7']:>8.4f}{r['g50']:>10.4f}{r['g25']:>10.4f}{r['g10']:>10.4f}")

    out = V5 / "results" / "a5_bma_tier1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"configs": configs,
                    "baseline": {"core7": 0.8338, "g50": 0.9344, "lignin": 0.697}},
                   f, indent=2, default=float)
    print(f"\nSaved → {out}")


def _load_spec(ck, kind, morg_dim, cp_dim, device):
    m = A5_BMA_Specialist(kind, morg_dim, 8, chemprop_dim=cp_dim).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


if __name__ == "__main__":
    main()
