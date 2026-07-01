"""Baran Task 2 leave-IL-out CV — A5.9 ensemble + Tier 2 Stage-2 #5 head.

Fork of `compare_a59_ens_vs_baran.py` that, for each fold, ALSO trains the
Tier 2 μ¹aug⁰ Stage-2 lignin head (predicted-μ feeding, no input jitter) on
top of the router-fused backbone, then evaluates BOTH:

  (a) raw ensemble mean (baseline A5.9, matches previous script)
  (b) Tier 2 Stage-2 lignin prediction — what the paper's same-split headline
      uses (lignin R² = 0.737 ± 0.025 on 10 seeds)

A priori: the Tier 2 head is an in-distribution feature-recycling win. On
OOD folds the backbone's core-7 predictions degrade, so pred-μ feeding may
be neutral or negative. This run TESTS that prediction.

Design:
  - Per fold: PCA(40) on train morgan; standardize surface/vit/cos on train
  - Train specialists A/B/C from scratch (n_specialist_seeds each)
  - Train router on frozen specialists (scalar mode)
  - Train Tier 2 Stage-2 head (n_stage2_seeds) — only deep_lignin + alpha
  - Evaluate ensemble mean lignin (baseline) AND Stage-2 lignin (Tier 2)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, set_seed  # noqa
from train_a2_two_stage import build_chemprop_40d, preprocess_physchem, v4_base
from train_a5_bma_pipeline import (
    A5_BMA_Specialist, train_specialist, train_router,
    _assemble_bank, _standardize,
    FRAME_DIM, COSMO_DIM, VIT_BANK, COSMO_BANK,
)
from train_a5_bma_tier2 import A5_BMA_Stage2_Tier2, train_stage2_tier2
from compare_a2_vs_baran import _load_baran_matched
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error

RESULTS = V5 / "results"
IDX_LIGNIN = 7


def _gated(pred, y, total_sigma, quantile):
    if len(pred) == 0: return None
    thr = np.quantile(total_sigma, quantile)
    keep = total_sigma <= thr
    if keep.sum() < 2: return None
    yk, pk = y[keep], pred[keep]
    return {"r2": float(r2_score(yk, pk)),
            "mae": float(mean_absolute_error(yk, pk)),
            "n_keep": int(keep.sum()), "n_total": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-specialist-seeds", type=int, default=2,
                    help="Seeds per specialist per fold (ensemble diversity).")
    ap.add_argument("--n-stage2-seeds", type=int, default=3,
                    help="Stage-2 head seeds per fold (averaged).")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--aug-noise", type=float, default=0.0,
                    help="Tier 2 #6 jitter (0.0 = μ¹aug⁰ / #5 only).")
    ap.add_argument("--fold", type=int, default=None,
                    help="Run only this fold index (for SLURM array). If set, "
                         "writes per-fold JSON instead of aggregate.")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  specialist_seeds={args.n_specialist_seeds}  "
          f"stage2_seeds={args.n_stage2_seeds}  aug_noise={args.aug_noise}")

    tr, va, te, lig_ils, baran_lig_mask = _load_baran_matched()

    np.random.seed(42)
    il_order = np.random.permutation(lig_ils)
    fold_size = max(1, len(il_order) // args.n_splits)
    folds = [il_order[i * fold_size : (i + 1) * fold_size if i < args.n_splits - 1 else None]
             for i in range(args.n_splits)]

    pool_smiles = np.concatenate([tr["smiles"], va["smiles"], te["smiles"]])
    pool_il = np.concatenate([tr["il_ids"], va["il_ids"], te["il_ids"]])
    pool_y = np.concatenate([tr["targets"], va["targets"], te["targets"]]).astype(np.float32)
    pool_th = np.concatenate([tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]]).astype(np.float32)
    pool_mg = np.concatenate([tr["morgan_fp"], va["morgan_fp"], te["morgan_fp"]]).astype(np.float32)
    pool_cp = np.concatenate([tr["chemprop_fp"], va["chemprop_fp"], te["chemprop_fp"]]).astype(np.float32)
    pool_pf = np.concatenate([tr["preds_fusion"], va["preds_fusion"], te["preds_fusion"]]).astype(np.float32)
    pool_pc = np.concatenate([tr["preds_chemprop"], va["preds_chemprop"], te["preds_chemprop"]]).astype(np.float32)
    pool_surf = np.concatenate([tr["surface_fp"], va["surface_fp"], te["surface_fp"]]).astype(np.float32)
    pool_ph = np.concatenate([tr["physchem_feat"], va["physchem_feat"], te["physchem_feat"]]).astype(np.float32)
    pool_hp = np.concatenate([tr["has_physchem"], va["has_physchem"], te["has_physchem"]]).astype(np.float32)
    pool_v4 = (0.4 * pool_pf + 0.6 * pool_pc).astype(np.float32)

    vit_bank = dict(zip(*[np.load(VIT_BANK, allow_pickle=True)[k]
                            for k in ("smiles", "vit_feat")]))
    cos_bank = dict(zip(*[np.load(COSMO_BANK, allow_pickle=True)[k]
                            for k in ("smiles", "cosmo_feat")]))
    pool_vit, pool_hv = _assemble_bank(pool_smiles, vit_bank, FRAME_DIM)
    pool_cos, pool_hc = _assemble_bank(pool_smiles, cos_bank, COSMO_DIM)
    pool_hs = (pool_surf != 0).any(axis=1).astype(np.float32)

    fold_iter = [(args.fold, folds[args.fold])] if args.fold is not None \
                else list(enumerate(folds))

    fold_results = []
    for k, held in fold_iter:
        te_mask = np.isin(pool_il, held) & baran_lig_mask
        tr_mask = ~np.isin(pool_il, held)
        if te_mask.sum() == 0:
            print(f"\n=== Fold {k}: 0 test rows — skip ===")
            continue

        held_names = sorted(set(pool_il[te_mask]))
        print(f"\n=== Fold {k} ({te_mask.sum()} test rows, held={held_names[:2]}...) ===")

        pca = PCA(40).fit(pool_mg[tr_mask])
        f_tr = pca.transform(pool_mg[tr_mask]).astype(np.float32)
        f_te = pca.transform(pool_mg[te_mask]).astype(np.float32)
        cp_tr_z, cp_te_z = build_chemprop_40d(pool_cp[tr_mask], pool_cp[te_mask])

        surf_tr_z, mu_p, sd_p = _standardize(pool_surf[tr_mask], pool_hs[tr_mask])
        surf_te_z = ((pool_surf[te_mask] - mu_p) / sd_p).astype(np.float32) * pool_hs[te_mask][:, None]
        vit_tr_z, mu_v, sd_v = _standardize(pool_vit[tr_mask], pool_hv[tr_mask])
        vit_te_z = ((pool_vit[te_mask] - mu_v) / sd_v).astype(np.float32) * pool_hv[te_mask][:, None]
        cos_tr_z, mu_c, sd_c = _standardize(pool_cos[tr_mask], pool_hc[tr_mask])
        cos_te_z = ((pool_cos[te_mask] - mu_c) / sd_c).astype(np.float32) * pool_hc[te_mask][:, None]

        phys_tr_z, phys_te_z = preprocess_physchem(
            pool_ph[tr_mask], pool_hp[tr_mask],
            pool_ph[te_mask], pool_hp[te_mask])

        feats_tr = {"v4": pool_v4[tr_mask], "morg": f_tr,
                    "thermo": pool_th[tr_mask], "chemprop": cp_tr_z,
                    "surface": surf_tr_z, "vit": vit_tr_z, "cos": cos_tr_z,
                    "has_surf": pool_hs[tr_mask], "has_vit": pool_hv[tr_mask],
                    "has_cos": pool_hc[tr_mask],
                    "physchem": phys_tr_z, "has_physchem": pool_hp[tr_mask]}
        feats_te = {"v4": pool_v4[te_mask], "morg": f_te,
                    "thermo": pool_th[te_mask], "chemprop": cp_te_z,
                    "surface": surf_te_z, "vit": vit_te_z, "cos": cos_te_z,
                    "has_surf": pool_hs[te_mask], "has_vit": pool_hv[te_mask],
                    "has_cos": pool_hc[te_mask],
                    "physchem": phys_te_z, "has_physchem": pool_hp[te_mask]}
        y_tr = pool_y[tr_mask]; y_te = pool_y[te_mask]

        # ---- Train S specialists A/B/C × n_specialist_seeds ----
        specialists = {"A": [], "B": [], "C": []}
        for kind in ("A", "B", "C"):
            for seed in range(args.n_specialist_seeds):
                m_k, _ = train_specialist(kind, seed, feats_tr, y_tr, device,
                                           epochs=args.epochs, patience=40)
                m_k.eval()
                specialists[kind].append(m_k)

        # ---- Ensemble prediction (baseline A5.9, for comparison) ----
        all_mus, all_lvs = [], []
        for kind in ("A", "B", "C"):
            seed_mus, seed_lvs = [], []
            for m_k in specialists[kind]:
                mu, lv = m_k.forward_with_lv(
                    torch.from_numpy(feats_te["v4"]).to(device),
                    torch.from_numpy(feats_te["morg"]).to(device),
                    torch.from_numpy(feats_te["thermo"]).to(device),
                    torch.from_numpy(feats_te["chemprop"]).to(device),
                    surface=torch.from_numpy(feats_te["surface"]).to(device),
                    vit=torch.from_numpy(feats_te["vit"]).to(device),
                    cos=torch.from_numpy(feats_te["cos"]).to(device),
                    has_surf=torch.from_numpy(feats_te["has_surf"]).to(device),
                    has_vit=torch.from_numpy(feats_te["has_vit"]).to(device),
                    has_cos=torch.from_numpy(feats_te["has_cos"]).to(device))
                seed_mus.append(mu.detach().cpu().numpy())
                seed_lvs.append(lv.detach().cpu().numpy())
            all_mus.append(np.stack(seed_mus))
            all_lvs.append(np.stack(seed_lvs))
        all_mus = np.stack(all_mus); all_lvs = np.stack(all_lvs)
        pred_ens = all_mus.mean(axis=(0, 1))
        aleatoric = np.exp(all_lvs).mean(axis=(0, 1))
        epistemic = all_mus.reshape(-1, *all_mus.shape[2:]).var(axis=0)
        total_var = aleatoric + epistemic
        total_sigma_row = np.sqrt(total_var).mean(axis=-1)

        y_true = y_te[:, IDX_LIGNIN]
        pred_ens_lig = pred_ens[:, IDX_LIGNIN]
        r2_ens_all = float(r2_score(y_true, pred_ens_lig))
        ens_gated = {q: _gated(pred_ens_lig, y_true, total_sigma_row, q)
                      for q in (0.25, 0.5, 0.75)}

        # ---- Train router on seed-0 specialists (to match pipeline pattern) ----
        spec_one = {kind: specialists[kind][0] for kind in ("A", "B", "C")}
        router = train_router([spec_one["A"], spec_one["B"], spec_one["C"]],
                                feats_tr, y_tr, device,
                                epochs=args.epochs // 2, mode="scalar")
        router.eval()

        # ---- Train Tier 2 Stage-2 head (n_stage2_seeds) ----
        nf = 40
        stage2_preds, stage2_lvs = [], []
        for s in range(args.n_stage2_seeds):
            model = A5_BMA_Stage2_Tier2(spec_one, router, nf,
                                         use_pred_mu=True).to(device)
            model = train_stage2_tier2(model, feats_tr, y_tr, device, s,
                                         epochs=args.epochs,
                                         aug_noise=args.aug_noise)
            model.eval()
            keys = ("v4","morg","thermo","chemprop","surface","vit","cos",
                     "has_surf","has_vit","has_cos","physchem","has_physchem")
            with torch.no_grad():
                mu, lv = model(*[torch.from_numpy(feats_te[kk]).to(device)
                                  for kk in keys])
            stage2_preds.append(mu.cpu().numpy())
            stage2_lvs.append(lv.cpu().numpy())
        pred_t2 = np.stack(stage2_preds).mean(axis=0)
        pred_t2_lig = pred_t2[:, IDX_LIGNIN]
        # Uncertainty: use ensemble total_sigma_row (Stage-2 head reuses same
        # backbone uncertainty; deep_lignin is deterministic given fused μ).
        r2_t2_all = float(r2_score(y_true, pred_t2_lig))
        t2_gated = {q: _gated(pred_t2_lig, y_true, total_sigma_row, q)
                     for q in (0.25, 0.5, 0.75)}

        def _fmt(d, q): return f"{d[q]['r2']:+.4f}" if d[q] else "n/a"
        print(f"  A5.9 ens  : R² all={r2_ens_all:+.4f}  g50={_fmt(ens_gated, 0.5)}  "
              f"g25={_fmt(ens_gated, 0.25)}")
        print(f"  +Tier 2 #5: R² all={r2_t2_all:+.4f}  g50={_fmt(t2_gated, 0.5)}  "
              f"g25={_fmt(t2_gated, 0.25)}")
        print(f"  epistemic/aleatoric (lignin) = "
              f"{epistemic[:, IDX_LIGNIN].mean():.3f}/{aleatoric[:, IDX_LIGNIN].mean():.3f}")

        fold_results.append({
            "fold": k,
            "n": int(te_mask.sum()),
            "held_out_ils": held_names,
            "ensemble": {
                "r2_all": r2_ens_all,
                "r2_gated": {str(q): ens_gated[q] for q in (0.25, 0.5, 0.75)},
            },
            "tier2": {
                "r2_all": r2_t2_all,
                "r2_gated": {str(q): t2_gated[q] for q in (0.25, 0.5, 0.75)},
            },
            "aleatoric_mean": float(aleatoric[:, IDX_LIGNIN].mean()),
            "epistemic_mean": float(epistemic[:, IDX_LIGNIN].mean()),
        })

    # Per-fold mode: write one JSON per fold, skip the aggregate block.
    if args.fold is not None:
        out = RESULTS / f"a59_tier2_baran_fold_{args.fold}.json"
        json.dump({"fold": args.fold, "result": fold_results[0] if fold_results else None,
                    "n_specialist_seeds": args.n_specialist_seeds,
                    "n_stage2_seeds": args.n_stage2_seeds,
                    "aug_noise": args.aug_noise}, open(out, "w"), indent=2)
        print(f"\nSaved fold {args.fold}: {out}")
        return

    def _agg(key):
        xs = [f[key]["r2_all"] for f in fold_results]
        return float(np.mean(xs)), float(np.std(xs))
    ens_m, ens_s = _agg("ensemble")
    t2_m, t2_s = _agg("tier2")

    print(f"\n{'='*70}\nBaran Task 2 — A5.9 vs +Tier 2 Stage-2 #5\n{'='*70}")
    print(f"A2 baseline (no gate)       : R² = -0.41 ± 1.04")
    print(f"A5.9 ensemble ALL (this run): R² = {ens_m:+.4f} ± {ens_s:.4f}")
    print(f"A5.9 +Tier 2 #5 ALL         : R² = {t2_m:+.4f} ± {t2_s:.4f}")
    for q_key, label in [("0.25", "25%"), ("0.5", "50%"), ("0.75", "75%")]:
        ens_vals = [f["ensemble"]["r2_gated"][q_key]["r2"]
                    for f in fold_results if f["ensemble"]["r2_gated"][q_key]]
        t2_vals = [f["tier2"]["r2_gated"][q_key]["r2"]
                   for f in fold_results if f["tier2"]["r2_gated"][q_key]]
        if ens_vals:
            print(f"  gated@{label:3s}  ensemble: {np.mean(ens_vals):+.4f}  "
                  f"+Tier2: {np.mean(t2_vals):+.4f}  "
                  f"(Δ={np.mean(t2_vals)-np.mean(ens_vals):+.4f})")
    print(f"Baran GB (their own CV)     : R² = +0.52 ± 0.20")

    out = RESULTS / "a59_tier2_baran_task2.json"
    json.dump({"folds": fold_results,
                "ensemble_r2_mean": ens_m, "ensemble_r2_std": ens_s,
                "tier2_r2_mean": t2_m, "tier2_r2_std": t2_s,
                "n_specialist_seeds": args.n_specialist_seeds,
                "n_stage2_seeds": args.n_stage2_seeds,
                "aug_noise": args.aug_noise,
                "n_splits": args.n_splits}, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
