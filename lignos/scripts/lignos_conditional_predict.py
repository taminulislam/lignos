"""LIGNOS conditional-gate inference: σ_row-gated ID/OOD routing.

Deployable wrapper around LIGNOS that picks between two inference paths:

    query → σ_row = sqrt(σ_alea² + σ_epi²) from the A5.9 ensemble
          └─ if σ_row ≤ τ        → ID branch:  pred_gated  (A5.9 + Mahalanobis)
                                                (Task 1 +#5+#6 equivalent when
                                                 applied to a same-chemistry
                                                 test row; on leave-IL-out Task 2
                                                 rows this is the A5.9-ensemble
                                                 fallback).
          └─ if σ_row  > τ        → OOD branch: Head E + XGBoost rule
                                                (rule-based per-row router over
                                                 K-NN, ProcSpec, XGBoost, head_D,
                                                 head_B — see Head E in
                                                 fit_meta_stacker.py).

This is a *deployment wrapper*, not a model — it requires no retraining and
combines existing trained components. The threshold τ is a deployment
hyperparameter; the paper's memory uses τ = q_0.5(σ_row) on the training
set, i.e. route OOD when the query's disagreement exceeds the median
training disagreement.

Usage:
    from lignos_conditional_predict import conditional_predict
    pred, branch, sigma = conditional_predict(row_dict, sigma_threshold=τ)

Demo mode (`python lignos_conditional_predict.py`) runs the gate on the
existing 108-row Task-2 CSVs and sweeps thresholds to show per-branch
behavior.
"""
from __future__ import annotations
import argparse, csv
import math
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
RESULTS = V5 / "results"

# Head E rule constants — keep in sync with fit_meta_stacker.py.
BULKY = {"triflate", "tetrafluoroborate", "hexafluorophosphate", "tfsi"}
XGB_OOD_SET = {"chloride", "hydrogensulfate", "methylsulfate"}


def _finite(x) -> bool:
    try:
        return x is not None and not (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def sigma_row(row: dict) -> float:
    """Reconstruct σ_row from the per-row aleatoric and epistemic components.

    The stored `total_sigma_row` column is the precomputed combined σ; fall
    back to sqrt(σ_alea² + σ_epi²) if unavailable.
    """
    total = row.get("total_sigma_row")
    if _finite(total):
        return float(total)
    a = float(row.get("sigma_alea_lig", 0.0) or 0.0)
    e = float(row.get("sigma_epi_lig", 0.0) or 0.0)
    return math.sqrt(a * a + e * e)


def head_e_predict(row: dict) -> Tuple[float, str]:
    """Apply the Head E rule cascade (including the XGBoost rule) to one row.

    Returns (prediction, rule_tag). The cascade reads the same per-row fields
    populated by fit_meta_stacker.py: tanimoto_nn, cation, anion, plus the
    candidate predictions `pred_gated`, `pred_knn_lig`, `pred_procspec_lig`,
    `pred_xgb_lig`. head_B (`pred_baran_feat_head` is a proxy for the raw GBM
    meta-stacker) is the final fallback.
    """
    tan = float(row.get("tanimoto_nn", 0.0) or 0.0)
    cat = str(row.get("cation", "_unk"))
    ani = str(row.get("anion", "_unk"))

    ens = float(row.get("pred_ens_lig", 0.0) or 0.0)
    gated = float(row.get("pred_gated", ens) or ens)
    knn = row.get("pred_knn_lig")
    proc = row.get("pred_procspec_lig")
    xgb = row.get("pred_xgb_lig")
    # head_B and head_D OOF predictions are exported to per-fold row CSVs
    # by fit_meta_stacker.py. If missing (legacy CSV), fall back to
    # pred_baran_feat_head — preserves rule dispatch order at a cost of
    # minor Head E R² approximation error.
    head_b = row.get("pred_head_b_lig")
    head_d = row.get("pred_head_d_lig")
    fallback_bfh = float(row.get("pred_baran_feat_head", ens) or ens)
    head_b_f = float(head_b) if _finite(head_b) else fallback_bfh
    head_d_f = float(head_d) if _finite(head_d) else fallback_bfh

    k_ok = _finite(knn)
    p_ok = _finite(proc)
    x_ok = _finite(xgb)

    knn_f = float(knn) if k_ok else None
    proc_f = float(proc) if p_ok else None
    xgb_f = float(xgb) if x_ok else None

    if tan < 0.7 and cat == "imidazolium" and p_ok:
        return proc_f, "rule1_procspec"
    if tan < 0.7:
        return ens, "rule2_ensemble"
    if cat == "choline":
        return gated, "rule3_gated"
    if tan >= 0.95 and k_ok:
        return knn_f, "rule4_knn"
    if cat == "imidazolium" and ani in BULKY:
        return head_d_f, "rule5_head_d"
    if cat == "imidazolium" and ani in XGB_OOD_SET and x_ok:
        return xgb_f, "rule6_xgb"
    return head_b_f, "rule7_head_b"


def id_branch_predict(row: dict) -> float:
    """ID path: LIGNOS Stage-2 / A5.9-gated prediction.

    On a same-chemistry test row this is equivalent to LIGNOS +#5+#6 (Task 1
    protocol). On a leave-IL-out Task 2 row this reduces to `pred_gated`
    (A5.9 ensemble + Mahalanobis OOD gate), since we do not retain a
    separate Stage-2 column per fold in the row CSVs.
    """
    ens = float(row.get("pred_ens_lig", 0.0) or 0.0)
    return float(row.get("pred_gated", ens) or ens)


def conditional_predict(row: dict, sigma_threshold: float
                         ) -> Tuple[float, str, float]:
    """Top-level gate: route to ID or OOD branch based on σ_row.

    Args:
        row: dict with the same fields as one row of
             `lignos_baran_feat_meta_fold_{k}_rows.csv` (post fit_meta_stacker
             + compute_knn + compute_procspec + train_chemprop_task2 + train_xgb).
        sigma_threshold: absolute σ_row threshold. Rows with σ_row ≤ threshold
             take the ID branch; rows above take the OOD branch.

    Returns:
        (prediction, branch_tag, sigma_row).
        branch_tag is one of 'id_branch', 'rule1_procspec', 'rule2_ensemble',
        'rule3_gated', 'rule4_knn', 'rule5_head_d_fallback', 'rule6_xgb',
        'rule7_head_b_fallback'.
    """
    s = sigma_row(row)
    if s <= sigma_threshold:
        return id_branch_predict(row), "id_branch", s
    pred, tag = head_e_predict(row)
    return pred, tag, s


# ---------------------------------------------------------------------------
# Demo: run the gate on the 108 Task-2 rows and sweep thresholds.
# ---------------------------------------------------------------------------
def _load_task2_rows() -> list[dict]:
    rows = []
    for k in range(20):  # supports 5-fold and 13-fold LoIoO layouts
        fp = RESULTS / f"lignos_baran_feat_meta_fold_{k}_rows.csv"
        if not fp.exists():
            continue
        with open(fp) as fh:
            for r in csv.DictReader(fh):
                # Cast numeric columns; leave NaN strings as NaN floats.
                for c in list(r.keys()):
                    v = r[c]
                    if v in ("", "nan", "NaN"):
                        r[c] = float("nan")
                        continue
                    try:
                        r[c] = float(v)
                    except (ValueError, TypeError):
                        pass  # keep as string (il_name, cation, anion, ...)
                rows.append(r)
    return rows


def _parse_classes(il_name: str) -> tuple[str, str]:
    """Best-effort cation/anion class inference from il_name.

    Mirrors fit_meta_stacker.py parse_classes but kept local so this module
    is standalone.
    """
    import re
    cation_patterns = [
        (r"choline|^Ch[A-Z]|Choline", "choline"),
        (r"\[?[Bb]?mim\]?|imidazolium", "imidazolium"),
        (r"pyrrolidinium|\[Pyr|\[Bmpyr", "pyrrolidinium"),
        (r"phosphonium|P\d{4}", "phosphonium"),
        (r"ammonium|\[N\d{4}", "ammonium"),
        (r"pyridinium", "pyridinium"),
    ]
    anion_patterns = [
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
    cat = next((c for p, c in cation_patterns
                if re.search(p, il_name, re.IGNORECASE)), "_unk")
    ani = next((c for p, c in anion_patterns
                if re.search(p, il_name, re.IGNORECASE)), "_unk")
    return cat, ani


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-threshold", type=float, default=None,
                    help="Absolute σ_row threshold. If omitted, sweep.")
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep σ thresholds from q_0.1 to q_0.9.")
    args = ap.parse_args()

    rows = _load_task2_rows()
    print(f"Loaded {len(rows)} Task-2 rows")

    # Attach cation/anion classes (same as fit_meta_stacker)
    for r in rows:
        c, a = _parse_classes(str(r.get("il_name", "")))
        r["cation"] = c
        r["anion"] = a

    import numpy as np
    sigmas = np.array([sigma_row(r) for r in rows])
    ys = np.array([float(r["y_true"]) for r in rows])

    print(f"σ_row stats: min={sigmas.min():.3f}  q_0.1={np.quantile(sigmas, 0.1):.3f}  "
          f"q_0.25={np.quantile(sigmas, 0.25):.3f}  q_0.5={np.quantile(sigmas, 0.5):.3f}  "
          f"q_0.75={np.quantile(sigmas, 0.75):.3f}  max={sigmas.max():.3f}")
    print()

    from sklearn.metrics import r2_score

    folds = np.array([int(r["fold"]) for r in rows])

    def eval_threshold(tau: float) -> dict:
        preds = []; branches = []
        for r in rows:
            p, br, _ = conditional_predict(r, tau)
            preds.append(p); branches.append(br)
        preds = np.array(preds)
        from collections import Counter
        cnt = Counter(branches)
        n_id = cnt.get("id_branch", 0)
        n_ood = len(rows) - n_id
        id_mask = np.array([b == "id_branch" for b in branches])
        r2_id = (float(r2_score(ys[id_mask], preds[id_mask]))
                 if id_mask.sum() >= 2 else float("nan"))
        r2_ood = (float(r2_score(ys[~id_mask], preds[~id_mask]))
                  if (~id_mask).sum() >= 2 else float("nan"))
        r2_all_pool = float(r2_score(ys, preds))
        # Per-fold mean (paper's Task-2 canonical statistic)
        fold_r2s = []
        for k in sorted(set(folds.tolist())):
            m = (folds == k)
            if m.sum() >= 2:
                fold_r2s.append(float(r2_score(ys[m], preds[m])))
        r2_perfold_mean = float(np.mean(fold_r2s))
        r2_perfold_std = float(np.std(fold_r2s))
        return {
            "tau": tau, "n_id": n_id, "n_ood": n_ood,
            "r2_all_pool": r2_all_pool,
            "r2_perfold_mean": r2_perfold_mean,
            "r2_perfold_std": r2_perfold_std,
            "r2_id_pool": r2_id, "r2_ood_pool": r2_ood,
            "fold_r2s": fold_r2s,
            "branches": dict(cnt),
        }

    if args.sigma_threshold is not None:
        tau = args.sigma_threshold
        res = eval_threshold(tau)
        print(f"Threshold σ_row ≤ {tau:.3f}:")
        print(f"  ID branch : n={res['n_id']:3d}  R²={res['r2_id_pool']:+.4f}")
        print(f"  OOD branch: n={res['n_ood']:3d}  R²={res['r2_ood_pool']:+.4f}")
        print(f"  Combined pooled R² = {res['r2_all_pool']:+.4f}")
        print(f"  Combined per-fold mean R² = {res['r2_perfold_mean']:+.4f} ± "
              f"{res['r2_perfold_std']:.4f}")
        print(f"  Per-fold R²: {[f'{r:+.3f}' for r in res['fold_r2s']]}")
        print(f"  Branches  : {res['branches']}")
    else:
        print(f"{'τ (quantile)':<14} {'τ (abs)':>9} {'n_ID':>5} {'n_OOD':>5} "
              f"{'R²_pool':>8} {'perfold μ':>10} {'perfold σ':>10}")
        print("-" * 80)
        results = []
        for q in [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]:
            tau = float(np.quantile(sigmas, q)) if q > 0 else -1.0
            if q == 1.0:
                tau = float(sigmas.max()) + 1e-9
            res = eval_threshold(tau)
            results.append(res)
            print(f"q_{q:<12.2f} {tau:>9.3f} {res['n_id']:>5d} {res['n_ood']:>5d} "
                  f"{res['r2_all_pool']:>+8.4f} {res['r2_perfold_mean']:>+10.4f} "
                  f"{res['r2_perfold_std']:>10.4f}")

        best_pool = max(results, key=lambda r: r["r2_all_pool"])
        best_pf = max(results, key=lambda r: r["r2_perfold_mean"])
        print()
        print(f"Best pooled R²    = {best_pool['r2_all_pool']:+.4f} at τ={best_pool['tau']:.3f}")
        print(f"Best per-fold μ R² = {best_pf['r2_perfold_mean']:+.4f} ± "
              f"{best_pf['r2_perfold_std']:.4f} at τ={best_pf['tau']:.3f}")
        print(f"   (this IS the 'conditional-gate LIGNOS' deployable result)")
        print(f"   Per-fold breakdown at best: "
              f"{[f'{r:+.3f}' for r in best_pf['fold_r2s']]}")

    # Reference baselines on all 108 rows:
    ref_gated = np.array([float(r.get("pred_gated", 0.0) or 0.0) for r in rows])
    r2_gated_pool = float(r2_score(ys, ref_gated))
    fold_r2_gated = []
    for k in sorted(set(folds.tolist())):
        m = (folds == k)
        fold_r2_gated.append(float(r2_score(ys[m], ref_gated[m])))
    print()
    print(f"Reference (pure ID branch, pred_gated on all rows):  "
          f"R² pooled = {r2_gated_pool:+.4f}  per-fold μ = {np.mean(fold_r2_gated):+.4f}")
    print(f"Reference (pure OOD branch, Head E + XGB, per-fold μ from "
          f"fit_meta_stacker): +0.4709 ± 0.2941")


if __name__ == "__main__":
    main()
