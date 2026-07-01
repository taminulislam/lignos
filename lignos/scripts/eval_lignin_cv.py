"""Lignin-specific leave-one-IL-out and 5-fold CV evaluation.

Compares our PerPropHead (shallow/deep, with/without wide thermo)
against Baran's GB/RF on the same Baran experimental data.

Uses the full LignoIL multi-task training set for PerPropHead
(transfer learning from gamma, H_E, etc.) but evaluates ONLY on
lignin yield prediction for Baran ILs.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, train_one_seed, predict, r2_per_prop


def morgan_fp(smi, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    return np.array(fp, dtype=np.float32)


def load_baran_matched():
    """Load Baran data matched to ILThermo SMILES."""
    baran = pd.read_csv(V5 / "data/LignoIL/baran2024_lignin_data.csv")
    baran = baran.dropna(subset=["yield_pct"]).reset_index(drop=True)

    ilthermo = pd.read_csv(V5 / "data/ilthermo_expanded_v2.csv")
    name_to_smi = {}
    for _, row in ilthermo.iterrows():
        if isinstance(row.get("il_name"), str) and isinstance(row.get("il_smiles"), str):
            name_to_smi[row["il_name"].lower()] = row["il_smiles"]

    abbrev = {
        "[Ch]": "cholinium", "[Bmim]": "1-butyl-3-methylimidazolium",
        "[Emim]": "1-ethyl-3-methylimidazolium", "[Mmim]": "1,3-dimethylimidazolium",
        "[Mim]": "1-methylimidazolium", "[DMEA]": "dimethylethanolammonium",
        "[C4H8SO3Hmim]": "1-(4-sulfobutyl)-3-methylimidazolium",
        "[C2H4COOHmim]": "1-(2-carboxyethyl)-3-methylimidazolium",
        "[OAc]": "acetate", "[Cl]": "chloride", "[HSO4]": "hydrogen sulfate",
        "[MeSO4]": "methyl sulfate", "[MeCO2]": "acetate",
        "[CF3SO3]": "trifluoromethanesulfonate", "[DMP]": "dimethyl phosphate",
    }

    def match(cat, an):
        c = abbrev.get(cat.strip(), cat.strip()).lower()
        a = abbrev.get(an.strip(), an.strip()).lower()
        for name, smi in name_to_smi.items():
            if c in name and a in name:
                return smi
        return None

    baran["smiles"] = baran.apply(
        lambda r: match(r["cation"], r["anion"]) if pd.notna(r["cation"]) else None, axis=1)
    return baran[baran["smiles"].notna()].reset_index(drop=True)


def build_baran_features(baran_df):
    """Build feature matrices for Baran data."""
    # Morgan FP
    fp_cache = {s: morgan_fp(s) for s in baran_df["smiles"].unique()}
    X_fp = np.array([fp_cache[s] for s in baran_df["smiles"]], dtype=np.float32)

    # Process conditions
    T = (baran_df["temp_C"].fillna(100).values + 273.15).astype(np.float32)
    time_h = baran_df["time_h"].fillna(1).values.astype(np.float32)
    il_conc = baran_df["il_conc"].fillna(1).values.astype(np.float32)
    perc_lig = baran_df["perc_lignins"].fillna(20).values.astype(np.float32)

    # Thermo feat (25D): [T, time, il_conc, perc_lignin, 0, ..., 0]
    thermo = np.zeros((len(baran_df), 25), dtype=np.float32)
    T_sc = StandardScaler().fit(T.reshape(-1, 1))
    time_sc = StandardScaler().fit(time_h.reshape(-1, 1))
    conc_sc = StandardScaler().fit(il_conc.reshape(-1, 1))
    lig_sc = StandardScaler().fit(perc_lig.reshape(-1, 1))
    thermo[:, 0] = T_sc.transform(T.reshape(-1, 1)).ravel()
    thermo[:, 1] = time_sc.transform(time_h.reshape(-1, 1)).ravel()
    thermo[:, 2] = conc_sc.transform(il_conc.reshape(-1, 1)).ravel()
    thermo[:, 3] = lig_sc.transform(perc_lig.reshape(-1, 1)).ravel()

    # Process features for GB/RF (flat vector)
    X_proc = np.column_stack([T, time_h, il_conc, perc_lig,
                               baran_df["perc_cellulose"].fillna(40).values,
                               baran_df["perc_hemicellulose"].fillna(20).values])

    y = baran_df["yield_pct"].values.astype(np.float32)
    smiles = baran_df["smiles"].values

    return X_fp, X_proc, thermo, y, smiles


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    baran = load_baran_matched()
    print(f"Baran matched data: {len(baran)} rows, {baran['smiles'].nunique()} ILs")

    X_fp, X_proc, thermo, y, smiles = build_baran_features(baran)
    unique_ils = list(set(smiles))
    print(f"Unique ILs: {len(unique_ils)}")
    for il in unique_ils:
        n = (smiles == il).sum()
        ym = y[smiles == il].mean()
        print(f"  {il[:50]:50s} n={n:>3} yield={ym:.1f}%")

    # Also load multi-task training data (for transfer learning in PerPropHead)
    lignoil_tr = np.load(V5 / "data/LignoIL/cached_train.npz", allow_pickle=True)
    mt_fp = lignoil_tr["morgan_fp"]
    mt_thermo = lignoil_tr["thermo_feat"]
    mt_targets = lignoil_tr["targets"].astype(np.float32)
    mt_v4f = lignoil_tr["preds_fusion"].astype(np.float32)
    mt_v4c = lignoil_tr["preds_chemprop"].astype(np.float32)
    mt_v4 = (0.4 * mt_v4f + 0.6 * mt_v4c).astype(np.float32)

    # ============================================================
    # Leave-one-IL-out CV
    # ============================================================
    print(f"\n{'='*60}")
    print("LEAVE-ONE-IL-OUT CROSS-VALIDATION")
    print(f"{'='*60}")

    results = {}

    for model_name in ["GB", "RF", "PerPropHead_deep_wide", "PerPropHead_shallow"]:
        all_preds = []
        all_true = []

        for held_out_il in unique_ils:
            te_mask = smiles == held_out_il
            tr_mask = ~te_mask

            if te_mask.sum() < 1 or tr_mask.sum() < 5:
                continue

            if model_name in ["GB", "RF"]:
                # PCA Morgan FP + process conditions
                pca = PCA(min(40, tr_mask.sum()-1)).fit(X_fp[tr_mask])
                X_tr = np.concatenate([pca.transform(X_fp[tr_mask]), X_proc[tr_mask]], axis=1)
                X_te = np.concatenate([pca.transform(X_fp[te_mask]), X_proc[te_mask]], axis=1)
                sc = StandardScaler().fit(X_tr)

                if model_name == "GB":
                    m = GradientBoostingRegressor(n_estimators=500, max_depth=4,
                                                   learning_rate=0.05, subsample=0.8, random_state=42)
                else:
                    m = RandomForestRegressor(n_estimators=500, max_depth=8, random_state=42)
                m.fit(sc.transform(X_tr), y[tr_mask])
                pred = m.predict(sc.transform(X_te))

            else:
                # PerPropHead: combine Baran train fold with multi-task LignoIL data
                # Baran train fold
                baran_fp_tr = X_fp[tr_mask]
                baran_thermo_tr = thermo[tr_mask]
                baran_y_tr = np.full((tr_mask.sum(), 8), np.nan, dtype=np.float32)
                y_mean, y_std = y[tr_mask].mean(), y[tr_mask].std()
                if y_std < 1e-8: y_std = 1.0
                baran_y_tr[:, 7] = (y[tr_mask] - y_mean) / y_std
                baran_v4_tr = np.zeros((tr_mask.sum(), 8), dtype=np.float32)

                # Combine with multi-task data
                combined_fp = np.concatenate([mt_fp, baran_fp_tr])
                combined_thermo = np.concatenate([mt_thermo, baran_thermo_tr])
                # Normalize multi-task lignin col with same scaler
                mt_tgt = mt_targets.copy()
                mt_lig = mt_tgt[:, 7]
                mt_lig_valid = mt_lig[~np.isnan(mt_lig)]
                if len(mt_lig_valid) > 0:
                    mt_tgt[:, 7] = (mt_lig - y_mean) / y_std  # re-normalize to Baran scale
                combined_y = np.concatenate([mt_tgt, baran_y_tr])
                combined_v4 = np.concatenate([mt_v4, baran_v4_tr])

                pca = PCA(40).fit(combined_fp)
                f_tr = pca.transform(combined_fp).astype(np.float32)
                f_te = pca.transform(X_fp[te_mask]).astype(np.float32)

                v4_te = np.zeros((te_mask.sum(), 8), dtype=np.float32)

                is_deep = "deep" in model_name
                is_wide = "wide" in model_name

                # Average over 3 seeds for stability
                preds_seeds = []
                for seed in range(3):
                    m = train_one_seed(seed, combined_v4, f_tr, combined_thermo,
                                       combined_y, device=device, balance_props=True,
                                       depth="deep" if is_deep else "shallow",
                                       wide_thermo=is_wide)
                    p = predict(m, v4_te, f_te, thermo[te_mask], device)
                    preds_seeds.append(p[:, 7])
                pred_norm = np.mean(preds_seeds, axis=0)
                pred = pred_norm * y_std + y_mean

            all_preds.extend(pred.tolist())
            all_true.extend(y[te_mask].tolist())

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        r2 = r2_score(all_true, all_preds)
        mae = mean_absolute_error(all_true, all_preds)

        print(f"\n  {model_name:35s}: R² = {r2:.4f}  MAE = {mae:.2f}%  (n={len(all_true)})")
        results[model_name] = {"r2": float(r2), "mae": float(mae), "n": len(all_true)}

    # ============================================================
    # 5-fold CV (IL-stratified)
    # ============================================================
    print(f"\n{'='*60}")
    print("5-FOLD IL-STRATIFIED CV")
    print(f"{'='*60}")

    np.random.seed(42)
    il_order = np.random.permutation(unique_ils)
    kf = KFold(n_splits=min(5, len(unique_ils)), shuffle=True, random_state=42)

    for model_name in ["GB", "RF", "PerPropHead_deep_wide"]:
        fold_r2 = []
        for fold_tr_ils, fold_te_ils in kf.split(il_order):
            te_il_set = set(il_order[fold_te_ils])
            te_mask = np.array([s in te_il_set for s in smiles])
            tr_mask = ~te_mask

            if model_name in ["GB", "RF"]:
                pca = PCA(min(40, X_fp[tr_mask].shape[0]-1)).fit(X_fp[tr_mask])
                X_tr = np.concatenate([pca.transform(X_fp[tr_mask]), X_proc[tr_mask]], axis=1)
                X_te = np.concatenate([pca.transform(X_fp[te_mask]), X_proc[te_mask]], axis=1)
                sc = StandardScaler().fit(X_tr)
                if model_name == "GB":
                    m = GradientBoostingRegressor(n_estimators=500, max_depth=4,
                                                   learning_rate=0.05, subsample=0.8, random_state=42)
                else:
                    m = RandomForestRegressor(n_estimators=500, max_depth=8, random_state=42)
                m.fit(sc.transform(X_tr), y[tr_mask])
                pred = m.predict(sc.transform(X_te))
            else:
                baran_fp_tr = X_fp[tr_mask]
                baran_thermo_tr = thermo[tr_mask]
                baran_y_tr = np.full((tr_mask.sum(), 8), np.nan, dtype=np.float32)
                y_mean, y_std = y[tr_mask].mean(), max(y[tr_mask].std(), 1e-8)
                baran_y_tr[:, 7] = (y[tr_mask] - y_mean) / y_std
                baran_v4_tr = np.zeros((tr_mask.sum(), 8), dtype=np.float32)

                combined_fp = np.concatenate([mt_fp, baran_fp_tr])
                combined_thermo = np.concatenate([mt_thermo, baran_thermo_tr])
                mt_tgt = mt_targets.copy()
                mt_tgt[:, 7] = np.where(np.isnan(mt_tgt[:, 7]), np.nan,
                                         (mt_tgt[:, 7] - y_mean) / y_std)
                combined_y = np.concatenate([mt_tgt, baran_y_tr])
                combined_v4 = np.concatenate([mt_v4, baran_v4_tr])

                pca = PCA(40).fit(combined_fp)
                f_tr = pca.transform(combined_fp).astype(np.float32)
                f_te = pca.transform(X_fp[te_mask]).astype(np.float32)
                v4_te = np.zeros((te_mask.sum(), 8), dtype=np.float32)

                preds_seeds = []
                for seed in range(3):
                    m = train_one_seed(seed, combined_v4, f_tr, combined_thermo,
                                       combined_y, device=device, balance_props=True,
                                       depth="deep", wide_thermo=True)
                    p = predict(m, v4_te, f_te, thermo[te_mask], device)
                    preds_seeds.append(p[:, 7])
                pred = np.mean(preds_seeds, axis=0) * y_std + y_mean

            fold_r2.append(r2_score(y[te_mask], pred))

        mean_r2 = np.mean(fold_r2)
        std_r2 = np.std(fold_r2)
        print(f"  {model_name:35s}: R² = {mean_r2:.4f} ± {std_r2:.4f}  folds={fold_r2}")
        results[f"{model_name}_5fold"] = {"mean": float(mean_r2), "std": float(std_r2),
                                           "folds": [float(f) for f in fold_r2]}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — Lignin Prediction Comparison")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'LOIO R²':>8} {'LOIO MAE':>10} {'5fold R²':>10}")
    print("-"*65)
    for name in ["GB", "RF", "PerPropHead_deep_wide", "PerPropHead_shallow"]:
        loio = results.get(name, {})
        fivef = results.get(f"{name}_5fold", {})
        print(f"{name:<35} {loio.get('r2','?'):>8} {loio.get('mae','?'):>10} "
              f"{fivef.get('mean','?'):>10}")

    out = V5 / "results" / "lignin_cv_comparison.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
