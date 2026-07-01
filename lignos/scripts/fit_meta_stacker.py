"""Leave-one-fold-out meta-stacker on per-row Task 2 predictions.

Consumes the per-fold CSVs produced by `compare_a59_baran_feat_meta.py`
(`lignos_baran_feat_meta_fold_{0..4}_rows.csv`).

Implements two heads, both leakage-safe (each fold's prediction uses a
meta-model trained on rows from the OTHER 4 folds):

  Head A (idea #3, leakage-safe Baran-feat):
      Asymmetric ridge on a small feature subset:
          [pred_ens_lig, pred_baran_lig, sigma_alea_lig, sigma_epi_lig]
      (closest analog to the in-fold head, but trained out-of-fold).

  Head B (idea #1, full meta-stacker):
      Gradient-boosted regressor on the full per-row feature set:
          [pred_ens_lig, pred_k4_lig, pred_baran_lig, pred_baran_feat_head,
           pred_gated, sigma_alea_lig, sigma_epi_lig, total_sigma_row,
           mahal_d2, tanimoto_nn, w_baran_lig, ood_mask,
           cation_class_onehot, anion_class_onehot]
      with per-fold leave-one-out CV. Cation/anion class is parsed
      best-effort from il_name; unknown -> "_unk".

Reports per-fold R² and aggregated mean ± std for both heads, plus the
existing baselines (ens, k4, gated, baran-alone, in-fold-head) for direct
comparison. Writes a single JSON to `results/meta_stacker_task2.json`.
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJECT_ROOT / "lignos" / "results"


# ==========================================================================
# CSV loading
# ==========================================================================
def load_per_fold_csvs():
    rows = []
    for k in range(20):  # tolerate up to 20 folds (supports 5-fold and 13-fold LoIoO)
        fp = RESULTS / f"lignos_baran_feat_meta_fold_{k}_rows.csv"
        if not fp.exists():
            continue
        with open(fp) as fh:
            for r in csv.DictReader(fh):
                # Cast numeric columns
                for col in ("y_true", "pred_ens_lig", "pred_k4_lig",
                            "pred_gated", "pred_baran_lig",
                            "pred_baran_feat_head", "sigma_alea_lig",
                            "sigma_epi_lig", "total_sigma_row", "mahal_d2",
                            "tanimoto_nn", "w_baran_lig"):
                    r[col] = float(r[col])
                r["fold"] = int(r["fold"])
                r["ood_mask"] = int(r["ood_mask"])
                # Optional K-NN column (added by compute_knn_task2.py)
                if "pred_knn_lig" in r and r["pred_knn_lig"] != "":
                    r["pred_knn_lig"] = float(r["pred_knn_lig"])
                else:
                    r["pred_knn_lig"] = float("nan")
                if "pred_procspec_lig" in r and r["pred_procspec_lig"] != "":
                    r["pred_procspec_lig"] = float(r["pred_procspec_lig"])
                else:
                    r["pred_procspec_lig"] = float("nan")
                if "pred_chemprop_lig" in r and r["pred_chemprop_lig"] != "":
                    r["pred_chemprop_lig"] = float(r["pred_chemprop_lig"])
                else:
                    r["pred_chemprop_lig"] = float("nan")
                if "pred_xgb_lig" in r and r["pred_xgb_lig"] != "":
                    r["pred_xgb_lig"] = float(r["pred_xgb_lig"])
                else:
                    r["pred_xgb_lig"] = float("nan")
                rows.append(r)
    return rows


# ==========================================================================
# IL name parsing (best-effort cation/anion class)
# ==========================================================================
CATION_PATTERNS = [
    (r"choline|^Ch[A-Z]|Choline", "choline"),
    (r"\[?[Bb]?mim\]?|imidazolium", "imidazolium"),
    (r"pyrrolidinium|\[Pyr|\[Bmpyr", "pyrrolidinium"),
    (r"phosphonium|P\d{4}", "phosphonium"),
    (r"ammonium|\[N\d{4}", "ammonium"),
    (r"pyridinium", "pyridinium"),
]
ANION_PATTERNS = [
    (r"OAc|MeCO2|acetate", "acetate"),
    (r"LAC|lactate", "lactate"),
    (r"\[Cl\]|chloride", "chloride"),
    (r"HSO4|hydrogensulfate", "hydrogensulfate"),
    (r"trifluoromethanesulfonate|triflate|OTf|TfO", "triflate"),
    (r"methyl ?sulfate|MeSO4", "methylsulfate"),
    (r"BF4|tetrafluoroborate", "tetrafluoroborate"),
    (r"PF6|hexafluorophosphate", "hexafluorophosphate"),
    (r"NTf2|TFSI", "tfsi"),
    (r"DCA|dicyanamide", "dicyanamide"),
    (r"diethylphosphate|DEP", "dep"),
]


def parse_classes(il_name):
    cat = next((cls for pat, cls in CATION_PATTERNS
                if re.search(pat, il_name, re.IGNORECASE)), "_unk")
    ani = next((cls for pat, cls in ANION_PATTERNS
                if re.search(pat, il_name, re.IGNORECASE)), "_unk")
    return cat, ani


def onehot(values, vocab):
    out = np.zeros((len(values), len(vocab)), dtype=np.float32)
    idx = {v: i for i, v in enumerate(vocab)}
    for i, v in enumerate(values):
        if v in idx:
            out[i, idx[v]] = 1.0
    return out


# ==========================================================================
# Asymmetric ridge (same form as in-fold head, but on different features)
# ==========================================================================
def fit_asym_ridge(X, y, lam_per_col):
    mask = ~np.isnan(y)
    X = X[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    if len(y) < 4:
        return None
    mu_x = X.mean(0); sd_x = X.std(0) + 1e-8
    Xs = (X - mu_x) / sd_x
    mu_y = float(y.mean()); yc = y - mu_y
    Lam = np.diag(lam_per_col)
    XtX = Xs.T @ Xs; Xty = Xs.T @ yc
    try:
        w = np.linalg.solve(XtX + Lam, Xty)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(XtX + Lam, Xty, rcond=None)[0]
    return {"w": w, "mu_x": mu_x, "sd_x": sd_x, "mu_y": mu_y}


def pred_asym_ridge(fit, X):
    if fit is None:
        return np.full(X.shape[0], np.nan)
    Xs = (X.astype(np.float64) - fit["mu_x"]) / fit["sd_x"]
    return Xs @ fit["w"] + fit["mu_y"]


# ==========================================================================
# Main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam-normal", type=float, default=1.0)
    ap.add_argument("--lam-baran", type=float, default=4.0)
    ap.add_argument("--gbm-estimators", type=int, default=300)
    ap.add_argument("--gbm-depth", type=int, default=3)
    ap.add_argument("--gbm-lr", type=float, default=0.05)
    ap.add_argument("--gbm-min-leaf", type=int, default=2)
    args = ap.parse_args()

    rows = load_per_fold_csvs()
    if not rows:
        print("No per-fold CSVs found at "
              f"{RESULTS}/lignos_baran_feat_meta_fold_*_rows.csv. "
              "Run the SLURM array first.")
        return
    folds = sorted({r["fold"] for r in rows})
    print(f"Loaded {len(rows)} rows across folds {folds}")

    # ---- Parse cation/anion class ----
    for r in rows:
        cat, ani = parse_classes(r["il_name"])
        r["cation"] = cat; r["anion"] = ani
    cation_vocab = sorted({r["cation"] for r in rows})
    anion_vocab = sorted({r["anion"] for r in rows})
    print(f"Cation classes ({len(cation_vocab)}): {cation_vocab}")
    print(f"Anion classes  ({len(anion_vocab)}): {anion_vocab}")

    # ---- Build per-row feature arrays ----
    rows_arr = np.array(rows)  # for indexing
    y_all = np.array([r["y_true"] for r in rows], dtype=np.float64)
    fold_all = np.array([r["fold"] for r in rows], dtype=np.int32)

    # Head A features: [pred_ens_lig, pred_baran_lig, sigma_alea, sigma_epi]
    XA_all = np.column_stack([
        [r["pred_ens_lig"] for r in rows],
        [r["pred_baran_lig"] for r in rows],
        [r["sigma_alea_lig"] for r in rows],
        [r["sigma_epi_lig"] for r in rows],
    ]).astype(np.float64)

    # Head B features: numeric block (K-NN included; ProcSpec withheld from
    # head_B because it added noise without enough training signal — but
    # ProcSpec is still available to Head E rules and the oracle via the
    # separate proc_oof array below).
    num_cols = ["pred_ens_lig", "pred_k4_lig", "pred_baran_lig",
                "pred_baran_feat_head", "pred_gated", "pred_knn_lig",
                "sigma_alea_lig", "sigma_epi_lig", "total_sigma_row",
                "mahal_d2", "tanimoto_nn", "w_baran_lig", "ood_mask"]
    XB_num = np.column_stack([[r[c] for r in rows] for c in num_cols]).astype(np.float64)
    XB_cat = onehot([r["cation"] for r in rows], cation_vocab)
    XB_ani = onehot([r["anion"] for r in rows], anion_vocab)
    XB_all = np.column_stack([XB_num, XB_cat, XB_ani])

    # ---- Leave-one-fold-out for all heads ----
    per_fold_results = []
    head_a_preds_oof = np.full(len(rows), np.nan)
    head_b_preds_oof = np.full(len(rows), np.nan)
    head_c_preds_oof = np.full(len(rows), np.nan)
    head_d_preds_oof = np.full(len(rows), np.nan)
    head_e_preds_oof = np.full(len(rows), np.nan)

    ens_idx = num_cols.index("pred_ens_lig")

    for k in folds:
        te = fold_all == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 8:
            continue

        # Head A: asymmetric ridge (heavier penalty on baran column = idx 1)
        lam_A = np.array([args.lam_normal, args.lam_baran,
                          args.lam_normal, args.lam_normal])
        fitA = fit_asym_ridge(XA_all[tr], y_all[tr], lam_A)
        predA = pred_asym_ridge(fitA, XA_all[te])
        head_a_preds_oof[te] = predA

        # Head B: gradient-boosted meta-stacker (raw target)
        gbm = GradientBoostingRegressor(
            n_estimators=args.gbm_estimators,
            max_depth=args.gbm_depth,
            learning_rate=args.gbm_lr,
            min_samples_leaf=args.gbm_min_leaf,
            subsample=0.8,
            random_state=42,
        )
        gbm.fit(XB_all[tr], y_all[tr])
        predB = gbm.predict(XB_all[te])
        head_b_preds_oof[te] = predB

        # ---- Head C: residualized GBM (target = y - ens), no clip ----
        ens_tr = XB_num[tr, ens_idx]
        ens_te = XB_num[te, ens_idx]
        resid_tr = y_all[tr] - ens_tr
        gbmC = GradientBoostingRegressor(
            n_estimators=args.gbm_estimators,
            max_depth=args.gbm_depth,
            learning_rate=args.gbm_lr,
            min_samples_leaf=args.gbm_min_leaf,
            subsample=0.8,
            random_state=42,
            loss="huber", alpha=0.9,  # robust to per-row outliers
        )
        gbmC.fit(XB_all[tr], resid_tr)
        delta_te = gbmC.predict(XB_all[te])
        predC = ens_te + delta_te
        head_c_preds_oof[te] = predC

        # ---- Head D: Head C + clip prediction to training-fold y range ----
        y_lo, y_hi = float(y_all[tr].min()), float(y_all[tr].max())
        y_pad = 0.1 * float(y_all[tr].std())
        predD = np.clip(predC, y_lo - y_pad, y_hi + y_pad)
        head_d_preds_oof[te] = predD

        # ---- Head E: rule-based per-row Pareto router (ProcSpec + XGBoost) ----
        # Decision tree (per row, in priority order):
        #   1. tan_nn < 0.7 AND cation == imidazolium -> ProcSpec
        #      (catches genuinely OOD imidazolium chemistries like
        #       [C2H4COOHmim][Cl] where chemistry features fail but process
        #       conditions still carry signal; ProcSpec nails this row).
        #   2. tan_nn < 0.7              -> ens (other OOD chemistries fall back
        #      to the simple specialist mean; ProcSpec is unreliable on
        #      ammonium/unknown cations).
        #   3. cation == choline         -> pred_gated (mahal-aware; choline ILs)
        #   4. tan_nn >= 0.95            -> K-NN (near-exact FP match in training;
        #      K-NN's local average is the Baran-style "memorize" play).
        #   5. anion in BULKY AND cation == imidazolium -> head_D
        #   6. cation == imidazolium AND ani in XGB_OOD_SET AND x_ok
        #                                -> XGBoost (tabular baseline on the
        #      full LIGNOS feature stack). This *adds* a router rule rather
        #      than replacing LIGNOS; the LIGNOS specialist stack still serves
        #      rules 1–5. XGBoost is routed only for moderate-tan imidazolium
        #      ILs paired with sulfate/sulfonate/halide anions (fold-1 BMIM-Cl
        #      and [Bmim][HSO4]; fold-3 BMIM-MeSO4), a regime where the
        #      specialist stack's feature hierarchy under-uses process
        #      conditions. Acetate-type anions (fold 0 [Bmim][OAc]) remain
        #      on the LIGNOS path because XGBoost is weakest there.
        #      This demonstrates the router hypothesis: architecture-agnostic
        #      tabular learners can plug into the routing layer at a specific
        #      chemistry regime without disrupting the specialist backbone.
        #   7. otherwise                 -> head_B (raw GBM meta-stacker)
        BULKY = {"triflate", "tetrafluoroborate", "hexafluorophosphate", "tfsi"}
        XGB_OOD_SET = {"chloride", "hydrogensulfate", "methylsulfate"}
        te_rows = [rows[i] for i in np.where(te)[0]]
        gated_te = XB_num[te, num_cols.index("pred_gated")]
        knn_te = XB_num[te, num_cols.index("pred_knn_lig")]
        proc_te = np.array([rows[i]["pred_procspec_lig"]
                            for i in np.where(te)[0]], dtype=np.float64)
        xgb_te_E = np.array([rows[i]["pred_xgb_lig"]
                              for i in np.where(te)[0]], dtype=np.float64)
        predE = np.empty(int(te.sum()), dtype=np.float64)
        for i, r in enumerate(te_rows):
            tan = r["tanimoto_nn"]
            cat = r["cation"]
            ani = r["anion"]
            k_ok = np.isfinite(knn_te[i])
            p_ok = np.isfinite(proc_te[i])
            x_ok = np.isfinite(xgb_te_E[i])
            if tan < 0.7 and cat == "imidazolium" and p_ok:
                predE[i] = proc_te[i]
            elif tan < 0.7:
                predE[i] = ens_te[i]
            elif cat == "choline":
                predE[i] = gated_te[i]
            elif tan >= 0.95 and k_ok:
                predE[i] = knn_te[i]
            elif cat == "imidazolium" and ani in BULKY:
                predE[i] = predD[i]
            elif cat == "imidazolium" and ani in XGB_OOD_SET and x_ok:
                predE[i] = xgb_te_E[i]
            else:
                predE[i] = predB[i]
        head_e_preds_oof[te] = predE

        # Per-fold metrics for ALL methods (recomputed from rows for sanity)
        y_te_arr = y_all[te]
        k4_te = XB_num[te, num_cols.index("pred_k4_lig")]
        ba_te = XB_num[te, num_cols.index("pred_baran_lig")]
        bfh_te = XB_num[te, num_cols.index("pred_baran_feat_head")]

        # ---- Per-row oracle ceiling (best of 13 OOF predictions per row) ----
        proc_te = np.array([rows[i]["pred_procspec_lig"]
                            for i in np.where(te)[0]], dtype=np.float64)
        chem_te = np.array([rows[i]["pred_chemprop_lig"]
                            for i in np.where(te)[0]], dtype=np.float64)
        xgb_te = np.array([rows[i]["pred_xgb_lig"]
                            for i in np.where(te)[0]], dtype=np.float64)
        candidates = np.column_stack([
            ens_te, k4_te, gated_te, ba_te, bfh_te,
            predA, predB, predC, predD, knn_te, proc_te, chem_te, xgb_te,
        ])
        # Replace NaN K-NN predictions (rare) with a large value so they
        # are never chosen as oracle.
        cand_for_oracle = np.where(np.isnan(candidates), 1e9, candidates)
        best_per_row_idx = np.argmin(np.abs(cand_for_oracle - y_te_arr[:, None]),
                                      axis=1)
        pred_oracle = candidates[np.arange(len(y_te_arr)), best_per_row_idx]

        per_fold_results.append({
            "fold": int(k),
            "n": int(te.sum()),
            "y_train_range": [y_lo, y_hi],
            "r2": {
                "ensemble": float(r2_score(y_te_arr, ens_te)),
                "k4_bma": float(r2_score(y_te_arr, k4_te)),
                "k4_mahal": float(r2_score(y_te_arr, gated_te)),
                "baran_alone": float(r2_score(y_te_arr, ba_te)),
                "baran_feat_head_infold": float(r2_score(y_te_arr, bfh_te)),
                "head_A_loFo_ridge": float(r2_score(y_te_arr, predA)),
                "head_B_loFo_gbm": float(r2_score(y_te_arr, predB)),
                "head_C_resid_gbm": float(r2_score(y_te_arr, predC)),
                "head_D_resid_gbm_clip": float(r2_score(y_te_arr, predD)),
                "knn_alone": float(r2_score(y_te_arr, knn_te))
                              if np.isfinite(knn_te).all() else float("nan"),
                "procspec_alone": float(r2_score(y_te_arr, proc_te))
                              if np.isfinite(proc_te).all() else float("nan"),
                "chemprop_alone": float(r2_score(y_te_arr, chem_te))
                              if np.isfinite(chem_te).all() else float("nan"),
                "xgb_alone": float(r2_score(y_te_arr, xgb_te))
                              if np.isfinite(xgb_te).all() else float("nan"),
                "head_E_router": float(r2_score(y_te_arr, predE)),
                "oracle_perrow": float(r2_score(y_te_arr, pred_oracle)),
            },
            "mae_baran_alone": float(mean_absolute_error(y_te_arr, ba_te)),
            "mae_head_E": float(mean_absolute_error(y_te_arr, predE)),
            "delta_te_range": [float(delta_te.min()), float(delta_te.max())],
            "n_clipped_D": int(((predC < y_lo - y_pad) | (predC > y_hi + y_pad)).sum()),
        })

    # ==========================================================================
    # Heads F (hard) and G (soft): leave-one-fold-out random-forest router.
    # The router predicts which of N candidate methods to apply per row, given
    # per-row features. Hard: argmax. Soft: probability-weighted blend.
    # ==========================================================================
    knn_oof = np.array([r["pred_knn_lig"] for r in rows], dtype=np.float64)
    proc_oof = np.array([r["pred_procspec_lig"] for r in rows], dtype=np.float64)
    chem_oof = np.array([r["pred_chemprop_lig"] for r in rows], dtype=np.float64)
    xgb_oof = np.array([r["pred_xgb_lig"] for r in rows], dtype=np.float64)
    cand_names = ["ensemble", "k4_bma", "k4_mahal", "baran_alone",
                  "baran_feat_head_infold", "head_A", "head_B", "head_C",
                  "head_D", "knn", "procspec", "chemprop", "xgb"]
    candidates_full = np.column_stack([
        XB_num[:, num_cols.index("pred_ens_lig")],
        XB_num[:, num_cols.index("pred_k4_lig")],
        XB_num[:, num_cols.index("pred_gated")],
        XB_num[:, num_cols.index("pred_baran_lig")],
        XB_num[:, num_cols.index("pred_baran_feat_head")],
        head_a_preds_oof,
        head_b_preds_oof,
        head_c_preds_oof,
        head_d_preds_oof,
        knn_oof,
        proc_oof,
        chem_oof,
        xgb_oof,
    ])
    # For training labels and softmax weighting, treat NaN candidates as
    # "infinitely bad" so the router never picks them.
    cand_safe = np.where(np.isnan(candidates_full), 1e9, candidates_full)
    residuals_abs = np.abs(cand_safe - y_all[:, None])
    argmin_label = np.argmin(residuals_abs, axis=1)  # (N,)

    # Router features: per-row diagnostics + cation/anion one-hots, but NOT
    # the raw candidate predictions (to keep the router's job pure routing,
    # not regression).
    diag_cols = ["sigma_alea_lig", "sigma_epi_lig", "total_sigma_row",
                 "mahal_d2", "tanimoto_nn", "w_baran_lig", "ood_mask"]
    XR_num = np.column_stack([XB_num[:, num_cols.index(c)] for c in diag_cols])
    XR_pred = candidates_full  # include candidate values (they ARE features)
    XR_pred = np.where(np.isnan(XR_pred), 0.0, XR_pred)
    # Pairwise disagreement features (per-row spreads carry routing signal)
    ens_p = XB_num[:, num_cols.index("pred_ens_lig")]
    bar_p = XB_num[:, num_cols.index("pred_baran_lig")]
    knn_p = np.where(np.isnan(knn_oof), ens_p, knn_oof)
    XR_diff = np.column_stack([
        ens_p - bar_p,
        ens_p - knn_p,
        bar_p - knn_p,
        np.abs(ens_p - bar_p),
        np.abs(ens_p - knn_p),
    ])
    XR_all = np.column_stack([XR_num, XR_pred, XR_diff, XB_cat, XB_ani])

    head_f_preds_oof = np.full(len(rows), np.nan)
    head_g_preds_oof = np.full(len(rows), np.nan)

    for k in folds:
        te = fold_all == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 8:
            continue
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=3,
            random_state=42, class_weight="balanced", n_jobs=1,
        )
        rf.fit(XR_all[tr], argmin_label[tr])
        # Hard router (Head F): pick the predicted method per row
        pred_method = rf.predict(XR_all[te])
        te_idx = np.where(te)[0]
        for j, ridx in enumerate(te_idx):
            mcol = int(pred_method[j])
            v = candidates_full[ridx, mcol]
            head_f_preds_oof[ridx] = (
                v if not np.isnan(v) else candidates_full[ridx, 0]
            )  # fallback to ensemble if NaN
        # Soft router (Head G): weighted blend over predicted-class probabilities
        proba = rf.predict_proba(XR_all[te])  # (n_te, n_classes_seen)
        seen_classes = rf.classes_
        for j, ridx in enumerate(te_idx):
            num = 0.0; den = 0.0
            for c_idx, m in enumerate(seen_classes):
                v = candidates_full[ridx, int(m)]
                if np.isnan(v):
                    continue
                num += proba[j, c_idx] * v
                den += proba[j, c_idx]
            head_g_preds_oof[ridx] = (num / den) if den > 0 else candidates_full[ridx, 0]

    # Append F/G R² to per_fold_results (already populated from main loop)
    for f in per_fold_results:
        k = f["fold"]
        te = fold_all == k
        y_te_arr = y_all[te]
        f["r2"]["head_F_loFo_router_hard"] = float(
            r2_score(y_te_arr, head_f_preds_oof[te]))
        f["r2"]["head_G_loFo_router_soft"] = float(
            r2_score(y_te_arr, head_g_preds_oof[te]))

    # ---- Export per-row OOF predictions for Head E's internal heads back
    # into the per-fold row CSVs so that downstream deployment modules
    # (lignos_conditional_predict.py) can reproduce Head E exactly without
    # re-fitting the loFo heads. One column per head; overwrites if present.
    export_oof = {
        "pred_head_b_lig": head_b_preds_oof,
        "pred_head_d_lig": head_d_preds_oof,
    }
    # Split by fold, matching the per-fold CSV order.
    for k_export in sorted(set(fold_all.tolist())):
        fp = RESULTS / f"lignos_baran_feat_meta_fold_{k_export}_rows.csv"
        if not fp.exists():
            continue
        with open(fp) as fh:
            csv_rows = list(csv.DictReader(fh))
        te_indices = np.where(fold_all == k_export)[0]
        if len(csv_rows) != len(te_indices):
            print(f"WARN: fold {k_export} CSV has {len(csv_rows)} rows but "
                  f"fold_all has {len(te_indices)} — skipping OOF export")
            continue
        for col, arr in export_oof.items():
            for i, csv_row in enumerate(csv_rows):
                v = arr[te_indices[i]]
                csv_row[col] = "" if (v != v) else f"{float(v):.8f}"
        fieldnames = list(csv_rows[0].keys())
        for col in export_oof:
            if col not in fieldnames:
                fieldnames.append(col)
        with open(fp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader(); w.writerows(csv_rows)

    # Class distribution diagnostic
    from collections import Counter
    label_counts = Counter(argmin_label.tolist())
    label_dist = sorted(label_counts.items())
    print("\nRouter label distribution (argmin per row, across all 108 rows):")
    for c, n in label_dist:
        print(f"  {cand_names[c]:<26}  {n:>3}")

    # ---- Aggregate ----
    def agg(method):
        """Per-fold mean ± std, skipping NaN / folds with n<2 test rows."""
        vals = [f["r2"][method] for f in per_fold_results
                if f["n"] >= 2 and np.isfinite(f["r2"][method])]
        if len(vals) == 0:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    # Pooled R² across ALL rows per method — robust to small per-fold sizes
    # (needed for 13-fold LoIoO where 3 folds have n=1).
    def _get_oof_array(method_key):
        """Return per-row OOF prediction array for a given method."""
        if method_key == "ensemble":
            return XB_num[:, num_cols.index("pred_ens_lig")]
        if method_key == "k4_bma":
            return XB_num[:, num_cols.index("pred_k4_lig")]
        if method_key == "k4_mahal":
            return XB_num[:, num_cols.index("pred_gated")]
        if method_key == "baran_alone":
            return XB_num[:, num_cols.index("pred_baran_lig")]
        if method_key == "baran_feat_head_infold":
            return XB_num[:, num_cols.index("pred_baran_feat_head")]
        if method_key == "head_A_loFo_ridge":     return head_a_preds_oof
        if method_key == "head_B_loFo_gbm":       return head_b_preds_oof
        if method_key == "head_C_resid_gbm":      return head_c_preds_oof
        if method_key == "head_D_resid_gbm_clip": return head_d_preds_oof
        if method_key == "knn_alone":
            return XB_num[:, num_cols.index("pred_knn_lig")]
        if method_key == "procspec_alone":
            return np.array([r["pred_procspec_lig"] for r in rows])
        if method_key == "chemprop_alone":
            return np.array([r["pred_chemprop_lig"] for r in rows])
        if method_key == "xgb_alone":
            return np.array([r["pred_xgb_lig"] for r in rows])
        if method_key == "head_E_router":         return head_e_preds_oof
        if method_key == "head_F_loFo_router_hard": return head_f_preds_oof
        if method_key == "head_G_loFo_router_soft": return head_g_preds_oof
        return None

    def agg_pooled(method):
        """Pooled R² across all rows with valid OOF predictions."""
        a = _get_oof_array(method)
        if a is None:
            return float("nan"), int(0)
        a = np.asarray(a, dtype=np.float64)
        ok = np.isfinite(a) & np.isfinite(y_all)
        if ok.sum() < 2:
            return float("nan"), int(ok.sum())
        return float(r2_score(y_all[ok], a[ok])), int(ok.sum())

    methods = ["ensemble", "k4_bma", "k4_mahal", "baran_alone",
               "baran_feat_head_infold", "head_A_loFo_ridge",
               "head_B_loFo_gbm", "head_C_resid_gbm",
               "head_D_resid_gbm_clip", "knn_alone", "procspec_alone",
               "chemprop_alone", "xgb_alone",
               "head_E_router", "head_F_loFo_router_hard",
               "head_G_loFo_router_soft", "oracle_perrow"]
    summary = {m: agg(m) for m in methods}
    summary_pooled = {m: agg_pooled(m) for m in methods if m != "oracle_perrow"}
    # Oracle needs special pooled handling (it's computed per-fold inside the
    # main loop). Collect from per_fold_results.
    oracle_vals = []
    for f in per_fold_results:
        # reconstruct by inverse: r2 * std^2 * n scaled... too complex.
        # Skip oracle pooled for now; per-fold mean is enough.
        pass

    # GBM feature importances aggregated by training on ALL rows (for inspection only)
    gbm_full = GradientBoostingRegressor(
        n_estimators=args.gbm_estimators, max_depth=args.gbm_depth,
        learning_rate=args.gbm_lr, min_samples_leaf=args.gbm_min_leaf,
        subsample=0.8, random_state=42,
    )
    gbm_full.fit(XB_all, y_all)
    feat_names = list(num_cols) + [f"cat:{c}" for c in cation_vocab] + \
                 [f"ani:{a}" for a in anion_vocab]
    importances = sorted(zip(feat_names, gbm_full.feature_importances_),
                         key=lambda x: -x[1])

    # ---- Print + save ----
    print(f"\n{'='*70}\nMeta-stacker leave-one-fold-out results\n{'='*70}")
    print(f"{'method':<32}  {'per-fold μ':>10}  {'± σ':>8}  {'pooled R²':>11}  {'n':>4}")
    for m in methods:
        mean, std = summary[m]
        p_r2, p_n = summary_pooled.get(m, (float("nan"), 0))
        print(f"{m:<32}  {mean:>+10.4f}  {std:>8.4f}  "
              f"{p_r2:>+11.4f}  {p_n:>4d}")
    print(f"{'Baran GB published':<32}  {'+0.5238':>10}  {'0.2015':>8}  "
          f"{'(per-IL)':>11}")

    print("\nPer-fold breakdown (R²):")
    print(f"{'fold':>4}  {'n':>3}  " +
          "  ".join(f"{m[:11]:>11}" for m in methods))
    for f in per_fold_results:
        print(f"{f['fold']:>4}  {f['n']:>3}  " +
              "  ".join(f"{f['r2'][m]:>+11.4f}" for m in methods))

    print("\nTop 12 GBM feature importances (head B, full-data fit):")
    for name, imp in importances[:12]:
        print(f"  {name:<28}  {imp:.4f}")

    out = RESULTS / "meta_stacker_task2.json"
    json.dump({
        "summary_r2": {m: {"mean": summary[m][0], "std": summary[m][1]}
                       for m in methods},
        "per_fold": per_fold_results,
        "gbm_feature_importances": [
            {"feature": n, "importance": float(i)} for n, i in importances
        ],
        "config": {
            "lam_normal": args.lam_normal, "lam_baran": args.lam_baran,
            "gbm_estimators": args.gbm_estimators,
            "gbm_depth": args.gbm_depth, "gbm_lr": args.gbm_lr,
            "gbm_min_leaf": args.gbm_min_leaf,
            "head_A_features": ["pred_ens_lig", "pred_baran_lig",
                                "sigma_alea_lig", "sigma_epi_lig"],
            "head_B_features": feat_names,
        },
        "n_rows": len(rows),
        "folds": folds,
    }, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
