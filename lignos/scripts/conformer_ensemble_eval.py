#!/usr/bin/env python3
"""Stage 2c — Conformer ensembling evaluation for the Combined(40D) baseline.

Expects cached_image_features_{split}_conf_{k}.npz files produced by
extract_vit_features_per_conformer.py. Two ensembling modes:

    feature_avg: average vit_feat across conformers, then PCA+PerPropHead
                 (simpler, cheaper)
    prediction_avg: train PerPropHead once per conformer, average final
                    predictions (more expressive, costs N× training)

The default is prediction_avg because feature-space averaging can under-use
conformer diversity when the downstream model is non-linear.

Outputs:
    results/conformer_ensemble/summary.json — per-conformer-count R² curves
    results/conformer_ensemble/per_conformer_preds.npz — all raw predictions

Usage:
    python conformer_ensemble_eval.py --n_conformers 5 --mode prediction_avg
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))

from audit_residuals import (  # noqa: E402
    PROPS,
    PerPropHead,
    load_split,
    predict,
    r2_per_prop,
    set_seed,
    train_one_seed,
)


def load_conformer_features(conf_id):
    """Load per-conformer (train, val, test) V-JEPA features."""
    def ld(split):
        p = V5 / f"data/cached_image_features_{split}_conf_{conf_id}.npz"
        return np.load(p)["vit_feat"]
    return ld("train"), ld("val"), ld("test")


def load_supervised_features():
    sup = np.load(V5 / "data/supervised_vit_features.npz")["features"]
    return sup[:152], sup[152:152 + 32], sup[152 + 32:]


def build_40d_for_conformer(vj_tr, vj_va, vj_te, sup_tr, sup_va, sup_te):
    pca_vj = PCA(20).fit(vj_tr)
    pca_sup = PCA(20).fit(sup_tr)

    def build(vj, sup):
        a = pca_vj.transform(vj).astype(np.float32)
        b = pca_sup.transform(sup).astype(np.float32)
        return np.concatenate([a, b], axis=1)

    return build(vj_tr, sup_tr), build(vj_va, sup_va), build(vj_te, sup_te)


def v4_base(cached):
    return (0.4 * cached["preds_fusion"] + 0.6 * cached["preds_chemprop"]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_conformers", type=int, default=5)
    ap.add_argument("--mode", choices=["feature_avg", "prediction_avg"],
                    default="prediction_avg")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_dir", type=str, default=str(V5 / "results/conformer_ensemble"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load splits & v4 base predictions ──
    tc = load_split("train")
    vc = load_split("val")
    tsc = load_split("test")
    v4_tr = v4_base(tc)
    v4_va = v4_base(vc)
    # For test, use v4 seed-ensemble if present
    v4_te = v4_base(tsc)
    seed_dir = PROJECT_ROOT / "cosmobridge_v4/results/seed_predictions"
    if seed_dir.exists():
        files = sorted(seed_dir.glob("seed_*.npz"))
        if files:
            key = "preds" if "preds" in np.load(files[0]) else "predictions"
            v4_te = np.mean([np.load(f)[key] for f in files], axis=0).astype(np.float32)

    tr_th = tc["thermo_feat"].astype(np.float32)
    va_th = vc["thermo_feat"].astype(np.float32)
    te_th = tsc["thermo_feat"].astype(np.float32)
    tr_y = tc["targets"].astype(np.float32)
    va_y = vc["targets"].astype(np.float32)
    te_y = tsc["targets"].astype(np.float32)

    sup_tr, sup_va, sup_te = load_supervised_features()

    # ── Load all conformers ──
    conf_vits_tr, conf_vits_va, conf_vits_te = [], [], []
    for k in range(args.n_conformers):
        try:
            vj_tr, vj_va, vj_te = load_conformer_features(k)
        except FileNotFoundError as e:
            print(f"Skipping conformer {k}: {e}")
            continue
        conf_vits_tr.append(vj_tr)
        conf_vits_va.append(vj_va)
        conf_vits_te.append(vj_te)
    n_confs = len(conf_vits_tr)
    if n_confs == 0:
        print("No conformer features found — nothing to do")
        return
    print(f"Loaded {n_confs} conformers")

    # ── Two modes ──
    results_curve = []

    for n_use in range(1, n_confs + 1):
        print(f"\n=== Ensembling {n_use} conformers ({args.mode}) ===")

        if args.mode == "feature_avg":
            vj_tr_avg = np.mean(conf_vits_tr[:n_use], axis=0)
            vj_va_avg = np.mean(conf_vits_va[:n_use], axis=0)
            vj_te_avg = np.mean(conf_vits_te[:n_use], axis=0)
            f_tr, f_va, f_te = build_40d_for_conformer(
                vj_tr_avg, vj_va_avg, vj_te_avg, sup_tr, sup_va, sup_te
            )

            seed_r2s = []
            for seed in range(args.n_seeds):
                model = train_one_seed(seed, v4_tr, f_tr, tr_th, tr_y, device=device)
                te_pred = predict(model, v4_te, f_te, te_th, device)
                seed_r2s.append(r2_per_prop(te_pred, te_y))
            avg_r2 = float(np.mean([m["avg"] for m in seed_r2s]))
            results_curve.append({
                "n_conformers": n_use, "mode": "feature_avg",
                "avg_r2_mean": avg_r2,
                "avg_r2_std": float(np.std([m["avg"] for m in seed_r2s])),
                "per_prop": {p: float(np.mean([m[p] for m in seed_r2s])) for p in PROPS},
            })
            print(f"  R² = {avg_r2:.4f}")

        else:  # prediction_avg
            per_conf_preds = []  # shape will be (n_use, n_seeds, n_test, 7)
            for k in range(n_use):
                f_tr, _, f_te = build_40d_for_conformer(
                    conf_vits_tr[k], conf_vits_va[k], conf_vits_te[k],
                    sup_tr, sup_va, sup_te,
                )
                this_conf_preds = []
                for seed in range(args.n_seeds):
                    model = train_one_seed(seed, v4_tr, f_tr, tr_th, tr_y, device=device)
                    te_pred = predict(model, v4_te, f_te, te_th, device)
                    this_conf_preds.append(te_pred)
                per_conf_preds.append(np.stack(this_conf_preds))  # (n_seeds, N, 7)
            per_conf_preds = np.stack(per_conf_preds)  # (n_use, n_seeds, N, 7)
            # Ensemble across conformers then average over seeds
            ensembled = per_conf_preds.mean(axis=0)  # (n_seeds, N, 7)
            per_seed_r2 = [r2_per_prop(ensembled[s], te_y) for s in range(args.n_seeds)]
            avg_r2 = float(np.mean([m["avg"] for m in per_seed_r2]))
            results_curve.append({
                "n_conformers": n_use, "mode": "prediction_avg",
                "avg_r2_mean": avg_r2,
                "avg_r2_std": float(np.std([m["avg"] for m in per_seed_r2])),
                "per_prop": {p: float(np.mean([m[p] for m in per_seed_r2])) for p in PROPS},
            })
            print(f"  R² = {avg_r2:.4f}")

    # ── Save ──
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"mode": args.mode, "curve": results_curve}, f, indent=2)
    print(f"\nWrote {summary_path}")
    print("\nR² curve vs conformer count:")
    for r in results_curve:
        print(f"  n={r['n_conformers']}: R² = {r['avg_r2_mean']:.4f} ± {r['avg_r2_std']:.4f}")


if __name__ == "__main__":
    main()
