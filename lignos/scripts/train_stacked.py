"""Stacked model: ensemble fingerprints + Kamlet-Taft β + GB residual stacking.

Stage 1: Train per-property GB models → get OOF predictions
Stage 2: Train PerPropHead with [ensemble_FP + GB_preds + KT_descriptors] as features
         Two-stage lignin transfer on top.
"""
import json, sys, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import (PROPS, CORE_PROPS, PerPropHead, predict,
                              r2_per_prop, set_seed, train_one_seed)


def load_split(split):
    d = np.load(V5 / "data/LignoIL" / f"cached_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    f, p = c.get("preds_fusion"), c.get("preds_chemprop")
    if f is not None and p is not None:
        return (0.4 * f + 0.6 * p).astype(np.float32)
    return np.zeros_like(c["targets"], dtype=np.float32)


def compute_ensemble_fp(smiles, nbits_morgan=2048):
    """Morgan + MACCS + RDKit FP + Kamlet-Taft proxy descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits_morgan + 166 + 10, dtype=np.float32)

    # Morgan FP (2048)
    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits_morgan)
    morgan_arr = np.array(morgan, dtype=np.float32)

    # MACCS keys (166)
    maccs = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    maccs_arr = np.array(maccs, dtype=np.float32)

    # Kamlet-Taft β proxy + molecular descriptors (10D)
    kt_descs = np.array([
        Descriptors.NumHAcceptors(mol),      # H-bond acceptors (β proxy)
        Descriptors.NumHDonors(mol),         # H-bond donors (α proxy)
        Descriptors.TPSA(mol),               # topological polar surface area
        Descriptors.MolLogP(mol),            # lipophilicity
        Descriptors.MolWt(mol),              # molecular weight
        Descriptors.NumRotatableBonds(mol),  # flexibility
        Descriptors.NumAromaticRings(mol),   # aromaticity
        Descriptors.FractionCSP3(mol),       # sp3 fraction
        Descriptors.HeavyAtomCount(mol),     # size
        rdMolDescriptors.CalcLabuteASA(mol), # accessible surface area
    ], dtype=np.float32)

    return np.concatenate([morgan_arr, maccs_arr, kt_descs])


def compute_gb_oof_predictions(X_tr, y_tr, X_te, n_props):
    """Out-of-fold GB predictions for stacking."""
    oof_tr = np.full_like(y_tr, np.nan)
    preds_te = np.zeros((len(X_te), n_props), dtype=np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for prop_i in range(n_props):
        valid_mask = ~np.isnan(y_tr[:, prop_i])
        if valid_mask.sum() < 20:
            continue

        X_valid = X_tr[valid_mask]
        y_valid = y_tr[valid_mask, prop_i]

        # OOF predictions
        for fold_tr, fold_te in kf.split(X_valid):
            gb = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                            learning_rate=0.05, random_state=42)
            gb.fit(X_valid[fold_tr], y_valid[fold_tr])
            oof_tr[np.where(valid_mask)[0][fold_te], prop_i] = gb.predict(X_valid[fold_te])

        # Full model for test predictions
        gb_full = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                             learning_rate=0.05, random_state=42)
        gb_full.fit(X_valid, y_valid)
        preds_te[:, prop_i] = gb_full.predict(X_te)

    # Fill NaN OOF with 0 (neutral prediction)
    oof_tr = np.nan_to_num(oof_tr, nan=0.0).astype(np.float32)
    return oof_tr, preds_te.astype(np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tr = load_split("train")
    te = load_split("test")

    y_tr = tr["targets"].astype(np.float32)
    y_te = te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]
    v4_tr, v4_te = v4_base(tr), v4_base(te)

    smiles_tr = [s.decode() if isinstance(s, bytes) else s for s in tr["smiles"]]
    smiles_te = [s.decode() if isinstance(s, bytes) else s for s in te["smiles"]]

    # ================================================================
    # Build ensemble fingerprints + KT descriptors
    # ================================================================
    print("Computing ensemble fingerprints (Morgan+MACCS+KT)...")
    fp_cache = {}
    for s in set(smiles_tr + smiles_te):
        fp_cache[s] = compute_ensemble_fp(s)

    X_fp_tr = np.array([fp_cache[s] for s in smiles_tr], dtype=np.float32)
    X_fp_te = np.array([fp_cache[s] for s in smiles_te], dtype=np.float32)
    print(f"  Ensemble FP dim: {X_fp_tr.shape[1]} (2048 Morgan + 166 MACCS + 10 KT)")

    # PCA to 60D (wider than 40D to capture more from the richer FP)
    pca = PCA(60).fit(X_fp_tr)
    f_fp_tr = pca.transform(X_fp_tr).astype(np.float32)
    f_fp_te = pca.transform(X_fp_te).astype(np.float32)
    print(f"  After PCA: {f_fp_tr.shape[1]}D")

    # ================================================================
    # GB out-of-fold predictions for stacking
    # ================================================================
    print("\nComputing GB out-of-fold predictions for stacking...")
    # Use PCA'd FP + thermo[:5] as GB features
    X_gb_tr = np.concatenate([f_fp_tr, th_tr[:, :5]], axis=1)
    X_gb_te = np.concatenate([f_fp_te, th_te[:, :5]], axis=1)
    sc = StandardScaler().fit(X_gb_tr)
    X_gb_tr_s = sc.transform(X_gb_tr).astype(np.float32)
    X_gb_te_s = sc.transform(X_gb_te).astype(np.float32)

    gb_oof_tr, gb_preds_te = compute_gb_oof_predictions(X_gb_tr_s, y_tr, X_gb_te_s, len(PROPS))
    print(f"  GB OOF: {gb_oof_tr.shape}, GB test: {gb_preds_te.shape}")

    # ================================================================
    # Stacked features: PCA'd ensemble FP (60D) + GB predictions (8D) = 68D
    # ================================================================
    f_stacked_tr = np.concatenate([f_fp_tr, gb_oof_tr], axis=1).astype(np.float32)
    f_stacked_te = np.concatenate([f_fp_te, gb_preds_te], axis=1).astype(np.float32)
    print(f"\nStacked features: {f_stacked_tr.shape[1]}D (60 FP + {gb_oof_tr.shape[1]} GB)")

    results = []
    n_seeds = 10

    # ================================================================
    # Ablation 1: Stacked features + shallow (new)
    # ================================================================
    print(f"\n{'='*60}")
    print("Stacked (ensemble FP + GB) + Shallow + Unbalanced")
    print(f"{'='*60}")
    seed_r2s = []
    stage1_models = []
    for seed in range(n_seeds):
        model = train_one_seed(seed, v4_tr, f_stacked_tr, th_tr, y_tr,
                                device=device, balance_props=False)
        stage1_models.append(model)
        te_pred = predict(model, v4_te, f_stacked_te, th_te, device)
        r2 = r2_per_prop(te_pred, y_te)
        seed_r2s.append(r2)
    avg_c7 = float(np.mean([m["avg_core7"] for m in seed_r2s]))
    std_c7 = float(np.std([m["avg_core7"] for m in seed_r2s]))
    print(f"  R2_core7 = {avg_c7:.4f} +/- {std_c7:.4f}")
    pp = {}
    for p in PROPS:
        vals = [m.get(p) for m in seed_r2s if m.get(p) is not None and not np.isnan(m.get(p, float("nan")))]
        pp[p] = float(np.mean(vals)) if vals else float("nan")
        print(f"    {p}: {pp[p]:.4f}")
    results.append({"name": "Stacked+Shallow+Unbal", "avg_r2_core7": avg_c7,
                     "std_r2_core7": std_c7, "per_prop": pp})

    # ================================================================
    # Ablation 2: Stacked + two-stage lignin transfer
    # ================================================================
    print(f"\n{'='*60}")
    print("Stacked + Two-Stage Lignin Transfer")
    print(f"{'='*60}")
    stage2_r2s = []
    for seed in range(n_seeds):
        s1 = stage1_models[seed]
        s2 = copy.deepcopy(s1).to(device)
        for param in s2.parameters():
            param.requires_grad = False
        nf = f_stacked_tr.shape[1]
        head_in = nf + 5
        s2.heads[7] = nn.Sequential(
            nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        ).to(device)
        with torch.no_grad():
            s2.heads[7][-1].weight.mul_(0.01)
            s2.heads[7][-1].bias.zero_()
        for param in s2.heads[7].parameters():
            param.requires_grad = True
        s2.alphas.requires_grad = True

        set_seed(seed + 200)
        opt = AdamW(list(s2.heads[7].parameters()) + [s2.alphas], lr=1e-3, weight_decay=1e-2)
        sched = CosineAnnealingLR(opt, T_max=300)
        ds = TensorDataset(
            torch.from_numpy(v4_tr), torch.from_numpy(f_stacked_tr),
            torch.from_numpy(th_tr), torch.from_numpy(y_tr))
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        best_loss, best_state, bad = float("inf"), {k: v.clone() for k, v in s2.state_dict().items()}, 0
        for _ in range(300):
            s2.train()
            for v, i, t, y in loader:
                v, i, t, y = v.to(device), i.to(device), t.to(device), y.to(device)
                pred = s2(v, i, t)
                m = ~torch.isnan(y[:, 7])
                if m.sum() == 0: continue
                loss = ((pred[m, 7] - y[m, 7].nan_to_num(0)) ** 2).mean()
                if not torch.isfinite(loss): continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(s2.heads[7].parameters()) + [s2.alphas], 1.0)
                opt.step()
            sched.step()
            s2.eval()
            with torch.no_grad():
                vp = s2(torch.from_numpy(v4_tr).to(device), torch.from_numpy(f_stacked_tr).to(device),
                        torch.from_numpy(th_tr).to(device))
                lm = ~torch.isnan(torch.from_numpy(y_tr[:, 7]).to(device))
                if lm.sum() > 0:
                    tl = ((vp[lm, 7] - torch.from_numpy(y_tr[lm.cpu().numpy(), 7]).to(device).nan_to_num(0)) ** 2).mean().item()
                else: continue
            if np.isfinite(tl) and tl < best_loss:
                best_loss = tl; best_state = {k: v.clone() for k, v in s2.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= 50: break
        s2.load_state_dict(best_state); s2.eval()
        te_pred = predict(s2, v4_te, f_stacked_te, th_te, device)
        r2 = r2_per_prop(te_pred, y_te)
        stage2_r2s.append(r2)

    avg_c7_s2 = float(np.mean([m["avg_core7"] for m in stage2_r2s]))
    std_c7_s2 = float(np.std([m["avg_core7"] for m in stage2_r2s]))
    print(f"  R2_core7 = {avg_c7_s2:.4f} +/- {std_c7_s2:.4f}")
    pp2 = {}
    for p in PROPS:
        vals = [m.get(p) for m in stage2_r2s if m.get(p) is not None and not np.isnan(m.get(p, float("nan")))]
        pp2[p] = float(np.mean(vals)) if vals else float("nan")
        print(f"    {p}: {pp2[p]:.4f}")
    results.append({"name": "Stacked+TwoStage_lignin", "avg_r2_core7": avg_c7_s2,
                     "std_r2_core7": std_c7_s2, "per_prop": pp2})

    # ================================================================
    # Ablation 3: Morgan FP only (baseline comparison)
    # ================================================================
    print(f"\n{'='*60}")
    print("Morgan FP only (40D) + Shallow [baseline]")
    print(f"{'='*60}")
    pca_morgan = PCA(40).fit(tr["morgan_fp"])
    f_m_tr = pca_morgan.transform(tr["morgan_fp"]).astype(np.float32)
    f_m_te = pca_morgan.transform(te["morgan_fp"]).astype(np.float32)
    seed_r2s_b = []
    for seed in range(n_seeds):
        model = train_one_seed(seed, v4_tr, f_m_tr, th_tr, y_tr,
                                device=device, balance_props=False)
        te_pred = predict(model, v4_te, f_m_te, th_te, device)
        seed_r2s_b.append(r2_per_prop(te_pred, y_te))
    avg_b = float(np.mean([m["avg_core7"] for m in seed_r2s_b]))
    std_b = float(np.std([m["avg_core7"] for m in seed_r2s_b]))
    print(f"  R2_core7 = {avg_b:.4f} +/- {std_b:.4f}")
    ppb = {}
    for p in PROPS:
        vals = [m.get(p) for m in seed_r2s_b if m.get(p) is not None and not np.isnan(m.get(p, float("nan")))]
        ppb[p] = float(np.mean(vals)) if vals else float("nan")
        print(f"    {p}: {ppb[p]:.4f}")
    results.append({"name": "Morgan40D+Shallow [baseline]", "avg_r2_core7": avg_b,
                     "std_r2_core7": std_b, "per_prop": ppb})

    # Summary
    print(f"\n{'='*60}\n  FINAL COMPARISON\n{'='*60}")
    print(f"{'Name':<40} {'core7':>7} {'std':>7} {'lignin':>8}")
    print("-" * 65)
    for r in results:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<40} {r['avg_r2_core7']:>7.4f} {r['std_r2_core7']:>7.4f} {lig:>8.4f}")

    out = V5 / "results" / "stacked_comparison.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
