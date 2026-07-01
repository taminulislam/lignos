#!/usr/bin/env python3
"""v4 + Physics Consistency + Stacking: Two NEW strategies to beat v4.

Strategy 1: THERMODYNAMIC CONSISTENCY (reverse direction from paper)
    Paper did: derive γ1 from G_E + γ2 → improved γ1 from 0.887 to 0.899
    NEW: derive G_E from γ1 + γ2 → should improve G_E from 0.787 to ~0.90+
    NEW: derive G_mix from G_E → should improve G_mix from 0.769 to ~0.85+
    (The paper's γ1,γ2 predictions at R²>0.90 are MUCH stronger than G_E at 0.787)

Strategy 2: NON-LINEAR STACKING
    Paper uses: linear blending α*path_a + (1-α)*path_b
    NEW: gradient-boosted stacking on (pred_a, pred_b, pred_atom_surface, thermo)
    LOO analysis showed this could reach R²=0.97 on G_E (vs 0.787 linear)

Usage:
    python train_v4_physics_stacking.py --seeds 0-9
"""

import argparse, json, sys, pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5_ROOT = Path(__file__).resolve().parent.parent

PROPS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]

# v4 paper gate values
V4_GATES = [0.36, 0.39, 0.36, 0.42, 0.45, 0.37, 0.69]


def load_data():
    """Load v4 cached predictions and targets."""
    splits = {}
    for split in ["train", "val", "test"]:
        d = np.load(PROJECT_ROOT / f"cosmobridge_v4/data/cached_{split}.npz",
                     allow_pickle=True)
        splits[split] = {k: d[k] for k in d.keys()}
    return splits


def load_scaler():
    with open(PROJECT_ROOT / "data/processed/target_scaler.pkl", "rb") as f:
        return pickle.load(f)


def denormalize(values, scaler, prop_idx):
    return values * scaler.scale_[prop_idx] + scaler.mean_[prop_idx]


def normalize(values, scaler, prop_idx):
    return (values - scaler.mean_[prop_idx]) / scaler.scale_[prop_idx]


def compute_metrics(preds, targets):
    m = {}
    for i, p in enumerate(PROPS):
        ss_r = ((targets[:,i]-preds[:,i])**2).sum()
        ss_t = ((targets[:,i]-targets[:,i].mean())**2).sum()
        m[f"{p}_r2"] = 1 - ss_r/(ss_t+1e-8)
    m["avg_r2"] = np.mean(list(m.values()))
    return m


def physics_correction(preds_norm, thermo_feat, scaler):
    """Apply thermodynamic consistency corrections.

    Strategy 1a: Derive G_E from γ1 + γ2 using G_E = RT(x1·ln(γ1) + x2·ln(γ2))
    Strategy 1b: Derive G_mix from G_E using G_mix = G_E + RT(x1·ln(x1) + x2·ln(x2))
    Strategy 1c: Derive γ1 from G_E + γ2 (paper's approach)
    """
    corrected = preds_norm.copy()
    R = 8.314e-3  # kJ/(mol·K)

    # Denormalize to physical units
    gamma1_raw = denormalize(preds_norm[:, 0], scaler, 0)
    gamma2_raw = denormalize(preds_norm[:, 1], scaler, 1)

    # Get raw temperature from thermo features
    # thermo_feat is normalized; we need raw T
    feat_scaler_path = PROJECT_ROOT / "data/processed/feature_scaler.pkl"
    with open(feat_scaler_path, "rb") as f:
        fscaler = pickle.load(f)
    T_raw = thermo_feat[:, 0] * fscaler.scale_[0] + fscaler.mean_[0]
    x1 = 0.5  # equimolar

    # ── Strategy 1a: Derive G_E from γ1, γ2 ──
    # G_E = RT * (x1·ln(γ1) + x2·ln(γ2))
    # But we need the exact scaling factor (paper says r=0.9999 but ratio ~0.24)
    # First find the calibration constant from training data
    ge_raw_pred = denormalize(preds_norm[:, 2], scaler, 2)  # current G_E prediction

    # The identity: G_E ∝ RT*(x1·ln(γ1) + x2·ln(γ2))
    # From our analysis: ratio is ~0.2385
    ln_term = x1 * np.log(np.clip(gamma1_raw, 1e-6, None)) + \
              (1-x1) * np.log(np.clip(gamma2_raw, 1e-6, None))
    ge_physics = R * T_raw * ln_term

    # The ratio is constant, find it from the predictions themselves
    # Use a blend: α * physics + (1-α) * ml
    # Since gamma predictions are R²>0.90, the physics-derived G_E should be better
    # than the direct G_E prediction (R²=0.787)
    ge_derived_norm = normalize(ge_physics, scaler, 2)

    # ── Strategy 1b: Derive G_mix from G_E ──
    # G_mix = G_E + RT*(x1·ln(x1) + x2·ln(x2))
    ideal_term = R * T_raw * (x1 * np.log(x1) + (1-x1) * np.log(1-x1))
    gm_physics = ge_physics + ideal_term
    gm_derived_norm = normalize(gm_physics, scaler, 4)

    # ── Strategy 1c: Derive γ1 from G_E, γ2 (paper's approach) ──
    # γ1 = exp((G_E/(RT) - x2·ln(γ2)) / x1)
    ge_for_g1 = ge_raw_pred  # use ML G_E for this direction
    exponent = (ge_for_g1 / (R * T_raw) - (1-x1) * np.log(np.clip(gamma2_raw, 1e-6, None))) / x1
    gamma1_physics = np.exp(np.clip(exponent, -10, 10))
    g1_derived_norm = normalize(gamma1_physics, scaler, 0)

    return {
        "ge_derived": ge_derived_norm,
        "gm_derived": gm_derived_norm,
        "g1_derived": g1_derived_norm,
        "ge_physics_raw": ge_physics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0-9")
    args = parser.parse_args()

    s, e = map(int, args.seeds.split("-"))
    seeds = list(range(s, e + 1))

    print("v4 + Physics Consistency + Non-Linear Stacking")
    print("="*60)

    splits = load_data()
    scaler = load_scaler()

    # v4 predictions (using paper gate values)
    for split in splits:
        d = splits[split]
        d["v4_preds"] = np.zeros_like(d["targets"])
        for i in range(7):
            d["v4_preds"][:, i] = V4_GATES[i] * d["preds_fusion"][:, i] + \
                                   (1-V4_GATES[i]) * d["preds_chemprop"][:, i]

    # v4 baseline
    m_v4 = compute_metrics(splits["test"]["v4_preds"], splits["test"]["targets"])
    print(f"\nv4 baseline (paper gates): avg R²={m_v4['avg_r2']:.4f}")
    for p in PROPS:
        print(f"  {p:8s}: {m_v4[f'{p}_r2']:.4f}")

    # ══════════════════════════════════════════
    # Strategy 1: Physics Correction
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STRATEGY 1: Thermodynamic Consistency Corrections")
    print(f"{'='*60}")

    # Find optimal blending ratio on TRAINING data
    train_physics = physics_correction(
        splits["train"]["v4_preds"], splits["train"]["thermo_feat"], scaler)

    # Calibrate: find the linear scaling factor for ge_physics -> ge_actual
    ge_target_train = splits["train"]["targets"][:, 2]
    ge_ml_train = splits["train"]["v4_preds"][:, 2]
    ge_phy_train = train_physics["ge_derived"]

    # Find best blend alpha: G_E_final = α * G_E_physics + (1-α) * G_E_ml
    from scipy.optimize import minimize_scalar
    def neg_r2_blend(alpha):
        blended = alpha * ge_phy_train + (1-alpha) * ge_ml_train
        return -r2_score(ge_target_train, blended)
    res = minimize_scalar(neg_r2_blend, bounds=(0, 1), method='bounded')
    best_alpha_ge = res.x
    print(f"\n  G_E blend: α={best_alpha_ge:.3f} (physics) + {1-best_alpha_ge:.3f} (ML)")

    # Apply to test
    test_physics = physics_correction(
        splits["test"]["v4_preds"], splits["test"]["thermo_feat"], scaler)

    physics_corrected = splits["test"]["v4_preds"].copy()

    # G_E correction
    physics_corrected[:, 2] = best_alpha_ge * test_physics["ge_derived"] + \
                               (1-best_alpha_ge) * splits["test"]["v4_preds"][:, 2]

    # G_mix correction (similar blend)
    gm_target_train = splits["train"]["targets"][:, 4]
    gm_ml_train = splits["train"]["v4_preds"][:, 4]
    gm_phy_train = train_physics["gm_derived"]
    def neg_r2_gm(alpha):
        return -r2_score(gm_target_train, alpha*gm_phy_train + (1-alpha)*gm_ml_train)
    best_alpha_gm = minimize_scalar(neg_r2_gm, bounds=(0, 1), method='bounded').x
    print(f"  G_mix blend: α={best_alpha_gm:.3f} (physics) + {1-best_alpha_gm:.3f} (ML)")
    physics_corrected[:, 4] = best_alpha_gm * test_physics["gm_derived"] + \
                               (1-best_alpha_gm) * splits["test"]["v4_preds"][:, 4]

    # γ1 correction (paper's direction)
    g1_phy_train = train_physics["g1_derived"]
    g1_ml_train = splits["train"]["v4_preds"][:, 0]
    g1_target_train = splits["train"]["targets"][:, 0]
    def neg_r2_g1(alpha):
        return -r2_score(g1_target_train, alpha*g1_phy_train + (1-alpha)*g1_ml_train)
    best_alpha_g1 = minimize_scalar(neg_r2_g1, bounds=(0, 1), method='bounded').x
    print(f"  γ1 blend: α={best_alpha_g1:.3f} (physics) + {1-best_alpha_g1:.3f} (ML)")
    physics_corrected[:, 0] = best_alpha_g1 * test_physics["g1_derived"] + \
                               (1-best_alpha_g1) * splits["test"]["v4_preds"][:, 0]

    m_phys = compute_metrics(physics_corrected, splits["test"]["targets"])
    print(f"\n  After physics correction: avg R²={m_phys['avg_r2']:.4f} (Δ={m_phys['avg_r2']-m_v4['avg_r2']:+.4f})")
    for p in PROPS:
        delta = m_phys[f"{p}_r2"] - m_v4[f"{p}_r2"]
        marker = "**" if delta > 0.005 else ""
        print(f"    {p:8s}: {m_phys[f'{p}_r2']:.4f} (Δ={delta:+.4f}) {marker}")

    # ══════════════════════════════════════════
    # Strategy 2: Non-Linear Stacking
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STRATEGY 2: Non-Linear Stacking (Gradient Boosting)")
    print(f"{'='*60}")

    # Build stacking features: [pred_fusion, pred_chemprop, thermo_feat]
    X_train = np.column_stack([
        splits["train"]["preds_fusion"],
        splits["train"]["preds_chemprop"],
        splits["train"]["thermo_feat"],
    ])
    X_test = np.column_stack([
        splits["test"]["preds_fusion"],
        splits["test"]["preds_chemprop"],
        splits["test"]["thermo_feat"],
    ])

    all_seed_metrics = []
    for seed in seeds:
        np.random.seed(seed)
        stacked_preds = np.zeros_like(splits["test"]["targets"])

        for i, p in enumerate(PROPS):
            gb = GradientBoostingRegressor(
                n_estimators=100, max_depth=2, learning_rate=0.05,
                min_samples_leaf=5, subsample=0.8,
                random_state=seed,
            )
            gb.fit(X_train, splits["train"]["targets"][:, i])
            stacked_preds[:, i] = gb.predict(X_test)

        m_stack = compute_metrics(stacked_preds, splits["test"]["targets"])
        all_seed_metrics.append(m_stack)

        if seed == 0:
            print(f"\n  Seed {seed}: avg R²={m_stack['avg_r2']:.4f}")
            for p in PROPS:
                delta = m_stack[f"{p}_r2"] - m_v4[f"{p}_r2"]
                print(f"    {p:8s}: {m_stack[f'{p}_r2']:.4f} (Δ={delta:+.4f})")

    avgs = [m["avg_r2"] for m in all_seed_metrics]
    print(f"\n  Stacking {len(seeds)} seeds: avg R²={np.mean(avgs):.4f} ± {np.std(avgs):.4f}")

    # ══════════════════════════════════════════
    # Strategy 1+2 Combined: Physics + Stacking
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("COMBINED: Physics Correction + Stacking")
    print(f"{'='*60}")

    # Add physics-derived features to stacking
    train_physics_feats = physics_correction(
        splits["train"]["v4_preds"], splits["train"]["thermo_feat"], scaler)
    test_physics_feats = physics_correction(
        splits["test"]["v4_preds"], splits["test"]["thermo_feat"], scaler)

    X_train_full = np.column_stack([
        X_train,
        train_physics_feats["ge_derived"].reshape(-1, 1),
        train_physics_feats["gm_derived"].reshape(-1, 1),
        train_physics_feats["g1_derived"].reshape(-1, 1),
    ])
    X_test_full = np.column_stack([
        X_test,
        test_physics_feats["ge_derived"].reshape(-1, 1),
        test_physics_feats["gm_derived"].reshape(-1, 1),
        test_physics_feats["g1_derived"].reshape(-1, 1),
    ])

    all_combined_metrics = []
    for seed in seeds:
        np.random.seed(seed)
        combined_preds = np.zeros_like(splits["test"]["targets"])

        for i, p in enumerate(PROPS):
            gb = GradientBoostingRegressor(
                n_estimators=100, max_depth=2, learning_rate=0.05,
                min_samples_leaf=5, subsample=0.8, random_state=seed,
            )
            gb.fit(X_train_full, splits["train"]["targets"][:, i])
            combined_preds[:, i] = gb.predict(X_test_full)

        m_comb = compute_metrics(combined_preds, splits["test"]["targets"])
        all_combined_metrics.append(m_comb)

    avgs_c = [m["avg_r2"] for m in all_combined_metrics]
    print(f"\n  Combined {len(seeds)} seeds: avg R²={np.mean(avgs_c):.4f} ± {np.std(avgs_c):.4f}")

    # ══════════════════════════════════════════
    # FINAL COMPARISON
    # ══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  v4 paper (2-path gates):       {m_v4['avg_r2']:.4f}")
    print(f"  v4 paper (3-path router):      0.8078")
    print(f"  + Physics correction:          {m_phys['avg_r2']:.4f} (Δ={m_phys['avg_r2']-0.8078:+.4f})")
    print(f"  + Stacking:                    {np.mean(avgs):.4f} ± {np.std(avgs):.4f} (Δ={np.mean(avgs)-0.8078:+.4f})")
    print(f"  + Physics + Stacking:          {np.mean(avgs_c):.4f} ± {np.std(avgs_c):.4f} (Δ={np.mean(avgs_c)-0.8078:+.4f})")

    print(f"\nPer-property (Physics + Stacking, seed 0):")
    m = all_combined_metrics[0]
    for p in PROPS:
        print(f"  {p:8s}: {m[f'{p}_r2']:.4f} (v4={m_v4[f'{p}_r2']:.4f}, Δ={m[f'{p}_r2']-m_v4[f'{p}_r2']:+.4f})")

    # Save
    out = V5_ROOT / "results/v4_physics_stacking"
    out.mkdir(exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({
            "v4_baseline": {k: float(v) for k,v in m_v4.items()},
            "physics_correction": {k: float(v) for k,v in m_phys.items()},
            "stacking": {"per_seed": all_seed_metrics, "avg": float(np.mean(avgs))},
            "combined": {"per_seed": all_combined_metrics, "avg": float(np.mean(avgs_c))},
        }, f, indent=2, default=float)
    print(f"\nSaved: {out}/summary.json")


if __name__ == "__main__":
    main()
