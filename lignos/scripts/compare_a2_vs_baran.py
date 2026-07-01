"""Head-to-head comparison: our A2 two-stage vs Baran 2024 (GB/RF) on lignin.

Two protocols:
  Task 1 — Train on LignoIL_A1 train+val lignin-labeled rows, evaluate
           on LignoIL_A1 test (39 rows / 5 ILs). Scores: our a2_2stg
           (read from results/a2_two_stage.json) vs fresh Baran-style
           GB/RF fit on the same train features. Apples-to-apples on
           the same test set.

  Task 2 — 5-fold leave-IL-out CV on the 38-IL Baran 2024 dataset, using
           our A2 Stage-1 + Stage-2 recipe. Compare against published
           Baran GB result (R² = 0.524 ± 0.201 from
           results/baran_baseline_comparison.json).

Expected runtime: Task 1 ≈ 2 min (sklearn), Task 2 ≈ 50 min (GPU, 5 folds × 1 seed).
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, r2_per_prop, set_seed  # noqa
from train_a2_two_stage import (
    A2Head, A2StageTwoLigninWrapper,
    train_stage1_a2, train_stage2_lignin,
    build_chemprop_40d, preprocess_physchem,
    predict_stage1, predict_stage2, v4_base,
)

CACHE = V5 / "data" / "LignoIL_A1"
RESULTS = V5 / "results"

# ---------------------------------------------------------------------------
# TASK 1 — same test set as a2_2stg
# ---------------------------------------------------------------------------

def task1_baran_on_lignoil_test():
    print("\n" + "=" * 70)
    print("TASK 1 — Baran GB/RF on LignoIL_A1 test split (n=39)")
    print("=" * 70)
    tr = {k: v for k, v in np.load(CACHE / "cached_train.npz", allow_pickle=True).items()}
    va = {k: v for k, v in np.load(CACHE / "cached_val.npz", allow_pickle=True).items()}
    te = {k: v for k, v in np.load(CACHE / "cached_test.npz", allow_pickle=True).items()}

    # Features: 40-D Morgan PCA + 25-D thermo + 12-D physchem (Baran-style flat vector)
    tr_morgan = np.concatenate([tr["morgan_fp"], va["morgan_fp"]])
    te_morgan = te["morgan_fp"]
    pca = PCA(40).fit(tr_morgan)
    tr_fp = pca.transform(tr_morgan).astype(np.float32)
    te_fp = pca.transform(te_morgan).astype(np.float32)

    tr_thermo = np.concatenate([tr["thermo_feat"], va["thermo_feat"]]).astype(np.float32)
    te_thermo = te["thermo_feat"].astype(np.float32)
    tr_phys = np.concatenate([tr["physchem_feat"], va["physchem_feat"]]).astype(np.float32)
    te_phys = te["physchem_feat"].astype(np.float32)

    tr_y = np.concatenate([tr["targets"], va["targets"]])[:, 7].astype(np.float32)
    te_y = te["targets"][:, 7].astype(np.float32)

    # Filter to rows with non-NaN lignin labels
    tr_mask = ~np.isnan(tr_y)
    te_mask = ~np.isnan(te_y)
    print(f"Train lignin rows: {tr_mask.sum()}/{len(tr_y)}")
    print(f"Test lignin rows : {te_mask.sum()}/{len(te_y)}")

    X_tr = np.column_stack([tr_fp[tr_mask], tr_thermo[tr_mask], tr_phys[tr_mask]])
    X_te = np.column_stack([te_fp[te_mask], te_thermo[te_mask], te_phys[te_mask]])
    y_tr = tr_y[tr_mask]
    y_te = te_y[te_mask]

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_te_s = sc.transform(X_te)

    results = {}
    for name, model in [
        ("Baran_GB",  GradientBoostingRegressor(n_estimators=500, max_depth=4,
                                                  learning_rate=0.05, subsample=0.8,
                                                  random_state=42)),
        ("RF",        RandomForestRegressor(n_estimators=500, max_depth=8,
                                              random_state=42, n_jobs=-1)),
    ]:
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_te_s)
        r2 = r2_score(y_te, pred)
        mae = mean_absolute_error(y_te, pred)
        print(f"  {name:12s}: R² = {r2:.4f}   MAE = {mae:.3f}")
        results[name] = {"r2": float(r2), "mae": float(mae), "n_train": int(tr_mask.sum()),
                         "n_test": int(te_mask.sum())}

    # Our a2_2stg headline from the already-completed run
    try:
        a2 = json.load(open(RESULTS / "a2_two_stage.json"))
        lig_s2 = a2[1]["per_prop"]["lignin_wt"]
        core7_s2 = a2[1]["avg_r2_core7"]
        results["A2_2stg_Stage2"] = {"r2": lig_s2, "mae": None,
                                       "core7": core7_s2, "n_test": int(te_mask.sum()),
                                       "note": "10-seed mean from a2_two_stage.json; "
                                               "Stage-2 evaluated on same 39-row test split."}
        print(f"  {'A2_2stg (ours)':12s}: R² = {lig_s2:.4f}   core7 = {core7_s2:.4f}  (10-seed mean)")
    except FileNotFoundError:
        print("  (a2_two_stage.json missing — skipping our headline)")

    return results


# ---------------------------------------------------------------------------
# TASK 2 — A2 two-stage under Baran 5-fold IL-stratified CV
# ---------------------------------------------------------------------------

def _baran_il_smiles_set():
    """Return canonical SMILES set for the 38 Baran-2024 ILs.

    Uses the same abbreviation → systematic-name → SMILES resolution as
    eval_lignin_cv.py::load_baran_matched, but restricted to cation+anion
    lookup against ilthermo_expanded_v2.csv (the dictionary we already
    vetted).
    """
    from rdkit import Chem
    baran = pd.read_csv(V5 / "data/LignoIL/baran2024_lignin_data.csv")
    baran = baran.dropna(subset=["yield_pct"]).reset_index(drop=True)
    ilthermo = pd.read_csv(V5 / "data/ilthermo_expanded_v2.csv")
    name_to_smi = {}
    for _, row in ilthermo.iterrows():
        if isinstance(row.get("il_name"), str) and isinstance(row.get("il_smiles"), str):
            name_to_smi[row["il_name"].lower()] = row["il_smiles"]
    # Complete cation dict — all 8 Baran cations
    cation_smi = {
        "[Ch]":          "C[N+](C)(C)CCO",                     # cholinium
        "[Bmim]":        "CCCC[n+]1ccn(C)c1",                   # 1-butyl-3-methylimidazolium
        "[Emim]":        "CC[n+]1ccn(C)c1",                     # 1-ethyl-3-methylimidazolium
        "[Mmim]":        "Cn1cc[n+](C)c1",                      # 1,3-dimethylimidazolium
        "[Mim]":         "c1cn(c[nH+]1)",                       # 1-methylimidazolium (protonated)
        "[DMEA]":        "C[NH+](C)CCO",                        # dimethylethanolammonium
        "[C4H8SO3Hmim]": "Cn1cc[n+](CCCCS(=O)(=O)O)c1",         # 1-(4-sulfobutyl)-3-methylimidazolium
        "[C2H4COOHmim]": "Cn1cc[n+](CCC(=O)O)c1",               # 1-(2-carboxyethyl)-3-methylimidazolium
    }
    # Complete anion dict — all 27 Baran anions (mostly amino acids + small organics)
    anion_smi = {
        "[Cl]":     "[Cl-]", "[OAc]": "CC(=O)[O-]", "[MeCO2]": "CC(=O)[O-]",
        "[HSO4]":   "OS(=O)(=O)[O-]", "[MeSO4]": "COS(=O)(=O)[O-]",
        "[CF3SO3]": "O=S(=O)([O-])C(F)(F)F", "[DMP]": "COP(=O)([O-])OC",
        "[TFA]":    "O=C([O-])C(F)(F)F",
        "[For]":    "O=C[O-]",                         # formate
        "[But]":    "CCCC(=O)[O-]",                    # butyrate
        "[Hex]":    "CCCCCC(=O)[O-]",                  # hexanoate
        "[Oct]":    "CCCCCCCC(=O)[O-]",                # octanoate
        "[i-Oct]":  "CCCCCCC(C)C(=O)[O-]",             # 2-methylheptanoate (approx)
        "[Piv]":    "CC(C)(C)C(=O)[O-]",               # pivalate
        "[Bz]":     "O=C([O-])c1ccccc1",               # benzoate
        "[Nic]":    "O=C([O-])c1ccncc1",               # nicotinate
        "[Lac]":    "CC(O)C(=O)[O-]",                  # lactate
        "[Glc]":    "OC[C@@H](O)[C@H](O)[C@@H](O)C(=O)[O-]",  # gluconate
        "[C4H5O4]": "O=C([O-])CCC(=O)O",               # succinate monoanion
        # amino-acid anions (deprotonated carboxyl)
        "[Gly]":    "[O-]C(=O)CN",
        "[Ala]":    "C[C@@H](N)C(=O)[O-]",
        "[Ser]":    "OC[C@@H](N)C(=O)[O-]",
        "[Thr]":    "C[C@@H](O)[C@H](N)C(=O)[O-]",
        "[Met]":    "CSCC[C@H](N)C(=O)[O-]",
        "[Pro]":    "O=C([O-])[C@@H]1CCCN1",
        "[Phe]":    "N[C@@H](Cc1ccccc1)C(=O)[O-]",
        "[Lys]":    "NCCCC[C@H](N)C(=O)[O-]",
    }

    def _canon(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return Chem.MolToSmiles(m) if m else None

    smi_set = set()
    unmatched = []
    for _, r in baran.iterrows():
        cat = cation_smi.get(r["cation"].strip()) if isinstance(r.get("cation"), str) else None
        an = anion_smi.get(r["anion"].strip()) if isinstance(r.get("anion"), str) else None
        if cat and an:
            cs = _canon(f"{cat}.{an}")
            if cs:
                smi_set.add(cs)
        else:
            unmatched.append((r.get("cation"), r.get("anion")))
    if unmatched:
        print(f"Unmatched Baran cation/anion pairs: {len(set(unmatched))}")
    return smi_set


def _load_baran_matched():
    """Filter cache rows to Baran-only ILs with yield-type lignin labels.

    Returns (tr, va, te, lig_ils, mask_lig) where lig_ils is the list of
    BARAN-matching IL identifiers (not all lignin-labeled ILs). This fixes
    the 2026-04-19 CV bug where the pool contained 49 ILs (Baran +
    literature + COSMO-SAC proxy) and folds 1/4 catastrophically crashed
    because the held-out ILs came from a different distribution than the
    training ILs. Baran's own 5-fold CV uses ONLY their 38 ILs.
    """
    from rdkit import Chem
    tr = {k: v for k, v in np.load(CACHE / "cached_train.npz", allow_pickle=True).items()}
    va = {k: v for k, v in np.load(CACHE / "cached_val.npz", allow_pickle=True).items()}
    te = {k: v for k, v in np.load(CACHE / "cached_test.npz", allow_pickle=True).items()}

    baran_smi = _baran_il_smiles_set()
    print(f"Baran 2024 resolved SMILES: {len(baran_smi)}")

    # Canonicalize our cache SMILES and check which il_ids map to Baran.
    def _canon(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return Chem.MolToSmiles(m) if m else None
    all_smi = np.concatenate([tr["smiles"], va["smiles"], te["smiles"]])
    all_il = np.concatenate([tr["il_ids"], va["il_ids"], te["il_ids"]])
    all_y = np.concatenate([tr["targets"], va["targets"], te["targets"]])[:, 7]

    canon_smi = np.array([_canon(s) for s in all_smi])
    is_baran = np.array([s in baran_smi for s in canon_smi])
    mask_lig = (~np.isnan(all_y)) & is_baran

    lig_ils = sorted(set(all_il[mask_lig]))
    print(f"Baran-matched ILs with lignin labels in cache: {len(lig_ils)}")
    print(f"Baran-matched lignin rows                    : {int(mask_lig.sum())}")
    return tr, va, te, lig_ils, mask_lig


def task2_a2_cv_on_baran(n_seeds=3, n_splits=5):
    print("\n" + "=" * 70)
    print(f"TASK 2 — A2 two-stage, {n_splits}-fold leave-IL-out CV on BARAN-ONLY ILs")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tr, va, te, lig_ils, baran_lig_mask = _load_baran_matched()

    # Stratified IL split
    np.random.seed(42)
    il_order = np.random.permutation(lig_ils)
    fold_size = max(1, len(il_order) // n_splits)
    folds = [il_order[i * fold_size : (i + 1) * fold_size if i < n_splits - 1 else None]
             for i in range(n_splits)]

    # Concatenate all splits → one working pool.
    pool_smiles = np.concatenate([tr["smiles"], va["smiles"], te["smiles"]])
    pool_il = np.concatenate([tr["il_ids"], va["il_ids"], te["il_ids"]])
    pool_y = np.concatenate([tr["targets"], va["targets"], te["targets"]]).astype(np.float32)
    pool_th = np.concatenate([tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]]).astype(np.float32)
    pool_mg = np.concatenate([tr["morgan_fp"], va["morgan_fp"], te["morgan_fp"]]).astype(np.float32)
    pool_cp = np.concatenate([tr["chemprop_fp"], va["chemprop_fp"], te["chemprop_fp"]]).astype(np.float32)
    pool_ph = np.concatenate([tr["physchem_feat"], va["physchem_feat"], te["physchem_feat"]]).astype(np.float32)
    pool_hp = np.concatenate([tr["has_physchem"], va["has_physchem"], te["has_physchem"]])
    pool_pf = np.concatenate([tr["preds_fusion"], va["preds_fusion"], te["preds_fusion"]]).astype(np.float32)
    pool_pc = np.concatenate([tr["preds_chemprop"], va["preds_chemprop"], te["preds_chemprop"]]).astype(np.float32)
    pool_v4 = (0.4 * pool_pf + 0.6 * pool_pc).astype(np.float32)

    # Pre-compute which rows are "non-Baran lignin" — we want these to stay in
    # the training pool (they're extra data) but NOT be used for evaluation.
    non_baran_lig_mask = (~np.isnan(pool_y[:, 7])) & (~baran_lig_mask)
    print(f"Non-Baran lignin rows kept in training pool: {int(non_baran_lig_mask.sum())}")

    fold_r2s, fold_maes, fold_ns = [], [], []
    fold_details = []
    for k, held in enumerate(folds):
        # Test = held-out Baran ILs only
        te_mask = np.isin(pool_il, held) & baran_lig_mask
        # Train = everything NOT in held-out ILs (includes non-Baran lignin + all multi-task data)
        tr_mask = ~np.isin(pool_il, held)
        if te_mask.sum() == 0:
            print(f"  Fold {k}: 0 test rows (held-out Baran ILs empty) — skip")
            continue

        # PCA on the train-fold's Morgan fingerprints only
        pca = PCA(40).fit(pool_mg[tr_mask])
        f_tr = pca.transform(pool_mg[tr_mask]).astype(np.float32)
        f_te = pca.transform(pool_mg[te_mask]).astype(np.float32)

        cp_tr_raw = pool_cp[tr_mask]
        cp_te_raw = pool_cp[te_mask]
        cp_tr, cp_te = build_chemprop_40d(cp_tr_raw, cp_te_raw)

        # Physchem normalized on train fold
        phys_tr, phys_te = preprocess_physchem(
            pool_ph[tr_mask], pool_hp[tr_mask],
            pool_ph[te_mask], pool_hp[te_mask])
        hp_tr = pool_hp[tr_mask].astype(np.float32)
        hp_te = pool_hp[te_mask].astype(np.float32)

        v4_tr = pool_v4[tr_mask]; v4_te = pool_v4[te_mask]
        th_tr = pool_th[tr_mask]; th_te = pool_th[te_mask]
        y_tr = pool_y[tr_mask]; y_te = pool_y[te_mask]

        seed_preds = []
        for seed in range(n_seeds):
            s1 = train_stage1_a2(seed, v4_tr, f_tr, th_tr, cp_tr, y_tr, device)
            s2 = train_stage2_lignin(s1, v4_tr, f_tr, th_tr, cp_tr, phys_tr, hp_tr, y_tr,
                                      device, seed=seed + 100)
            p = predict_stage2(s2, v4_te, f_te, th_te, cp_te, phys_te, hp_te, device)
            seed_preds.append(p[:, 7])
        pred = np.mean(seed_preds, axis=0)

        y_true = y_te[:, 7]
        r2 = r2_score(y_true, pred)
        mae = mean_absolute_error(y_true, pred)
        fold_r2s.append(r2)
        fold_maes.append(mae)
        fold_ns.append(int(te_mask.sum()))
        held_names = sorted(set(pool_il[te_mask]))
        print(f"  Fold {k}: R² = {r2:.4f}  MAE = {mae:.3f}  n={te_mask.sum()}  ILs={held_names[:3]}...")
        fold_details.append({
            "fold": k, "r2": float(r2), "mae": float(mae), "n": int(te_mask.sum()),
            "held_out_ils": list(held_names),
        })

    r2_mean = float(np.mean(fold_r2s)) if fold_r2s else float("nan")
    r2_std = float(np.std(fold_r2s)) if fold_r2s else float("nan")
    mae_mean = float(np.mean(fold_maes)) if fold_maes else float("nan")
    print(f"\n  A2_2stg  CV mean R² = {r2_mean:.4f} ± {r2_std:.4f}   MAE = {mae_mean:.3f}")

    # Baran's own published numbers for context
    try:
        b = json.load(open(RESULTS / "baran_baseline_comparison.json"))
        gb_key = "GB (Baran's method)"
        gb_m, gb_s = b[gb_key]["r2_mean"], b[gb_key]["r2_std"]
        rf_m, rf_s = b["RF"]["r2_mean"], b["RF"]["r2_std"]
        print(f"\n  BARAN GB (their method, 5-fold CV): R² = {gb_m:.4f} ± {gb_s:.4f}")
        print(f"  BARAN RF baseline (5-fold CV)    : R² = {rf_m:.4f} ± {rf_s:.4f}")
    except Exception:
        pass

    return {"r2_mean": r2_mean, "r2_std": r2_std, "mae_mean": mae_mean,
            "folds": fold_details, "n_seeds": n_seeds, "n_splits": n_splits}


def main():
    out_json = RESULTS / "compare_a2_vs_baran.json"
    out = {}
    out["task1"] = task1_baran_on_lignoil_test()
    out["task2"] = task2_a2_cv_on_baran(n_seeds=3, n_splits=5)
    json.dump(out, open(out_json, "w"), indent=2)
    print(f"\nSaved → {out_json}")


if __name__ == "__main__":
    main()
