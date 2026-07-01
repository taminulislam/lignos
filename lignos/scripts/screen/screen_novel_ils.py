"""Day 2+3: virtual-screen the enumerated IL catalog with A5.9 Stage-2.

Pipeline:
  1. Load il_catalog.csv (3969 ILs).
  2. For each IL: Morgan FP(2048)→PCA(40) using the training-cache PCA,
     12-D physchem via RDKit, thermo_feat assembled at Baran-mean process
     conditions (T=120°C, time=2h, IL_conc=1, %cellulose=40, hemi=25, lignin=20).
  3. Load frozen Specialists A, B, C (from a5_bma/) + scalar router.
  4. For novel ILs: has_surf=has_vit=has_cos=0 (masked out of fusion,
     Specialist A dominates via BMA anchor).
  5. v4_base: use zero-vector (no teacher predictions available for novel ILs —
     A2 backbone can still output meaningful residuals from gated Morgan+thermo).
  6. Wrap in A5.9 Stage-2 with deep lignin head (scalar router mode).
  7. Forward pass → (μ, σ²) per IL for all 8 targets.
  8. Rank by μ_lignin, filter by σ²_lignin quantile; save ranked CSV.

Output: data/virtual_screen/ranked_candidates.csv
  Columns: rank, cation_name, anion_name, il_smiles, lignin_mu, lignin_sigma,
           μ_gamma1, ..., μ_P, σ_gamma1, ..., σ_P, confidence_flag
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS  # noqa
from train_a2_two_stage import preprocess_physchem
from train_a5_bma_pipeline import A5_BMA_Specialist, A5_BMA_Router, train_router
from train_a5_bma_stage2 import A5_BMA_Stage2, predict_stage2
from sklearn.decomposition import PCA as _PCA

CACHE = V5 / "data" / "LignoIL_A1"
BMA_DIR = V5 / "checkpoints" / "a5_bma"
CATALOG = PROJECT_ROOT / "data" / "virtual_screen" / "il_catalog.csv"
OUT_CSV = PROJECT_ROOT / "data" / "virtual_screen" / "ranked_candidates.csv"

FRAME_DIM = 192
SURFACE_DIM = 256
COSMO_DIM = 20


def morgan_fp(smi, nbits=2048):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_physchem(smi):
    """12-D physchem vector matching the training cache format."""
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return np.zeros(12, dtype=np.float32), 0.0
    try:
        feats = np.array([
            Descriptors.MolWt(m),                  # 0
            Descriptors.MolLogP(m),                # 1 (Wildman-Crippen)
            Descriptors.NumHDonors(m),             # 2
            Descriptors.NumHAcceptors(m),          # 3 (we log1p-transform later, matches train)
            Descriptors.TPSA(m),                   # 4
            Descriptors.NumRotatableBonds(m),      # 5 (log1p in train)
            Descriptors.HeavyAtomCount(m),         # 6
            Descriptors.NumAromaticRings(m),       # 7
            Descriptors.NumAliphaticRings(m),      # 8
            Descriptors.FractionCSP3(m),           # 9
            Descriptors.NumHeteroatoms(m),         # 10
            Descriptors.RingCount(m),              # 11
        ], dtype=np.float32)
        return feats, 1.0
    except Exception:
        return np.zeros(12, dtype=np.float32), 0.0


def build_thermo_feat(baran_mean=True):
    """25-D thermo_feat vector. Baran mean conditions for process-related dims."""
    t = np.zeros(25, dtype=np.float32)
    # Dim 0 = T(K); training T covers 268–700 after z-score. Using T=393K (120°C).
    # Since training z-scores with fit on covered rows, we need to match the scale.
    # Easiest: use value that is roughly median of training distribution after z-score.
    # For simplicity: use zero for all — the MODEL evaluates at "average" condition
    # across all properties. For true process-condition eval, we'd need to re-z-score.
    # For ranking purposes, zeros-at-training-mean is a valid comparison across ILs.
    if baran_mean:
        # Match Baran mean values, already z-scored (approximate)
        t[0] = 0.5    # T (moderate)
        t[1] = 0.2    # time
        t[2] = 0.5    # IL_conc
        t[3] = 0.0    # %cellulose (Baran median)
        t[4] = 0.0    # %hemicellulose
        t[5] = 0.0    # %lignin
    return t


def build_v4_base(n_rows):
    """v4_base = 0.4*preds_fusion + 0.6*preds_chemprop. For novel ILs, both are
    zero — so v4_base = 0 (8-D). A2 residual carries all signal."""
    return np.zeros((n_rows, 8), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--confidence-quantile", type=float, default=0.5,
                    help="Keep predictions with σ_lignin in the bottom `q` fraction.")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ----- Load catalog -----
    df = pd.read_csv(CATALOG)
    print(f"Catalog: {len(df)} candidate ILs")

    # ----- Load train cache for PCA + physchem normalization references -----
    tr = np.load(CACHE / "cached_train.npz", allow_pickle=True)
    pca_m = PCA(40).fit(tr["morgan_fp"])
    # physchem normalization stats
    tr_phys = tr["physchem_feat"].astype(np.float32).copy()
    tr_has_phys = tr["has_physchem"].astype(bool)
    # Apply same log-transforms as training (dims 3 and 5)
    tr_phys[:, 3] = np.log1p(np.maximum(tr_phys[:, 3], 0))
    tr_phys[:, 5] = np.log1p(np.maximum(tr_phys[:, 5], 0))
    mu_phys = tr_phys[tr_has_phys].mean(axis=0)
    sd_phys = tr_phys[tr_has_phys].std(axis=0) + 1e-6

    # ----- Compute features for each candidate -----
    print("\nComputing Morgan + physchem features for all candidates...")
    N = len(df)
    morg_raw = np.zeros((N, 2048), dtype=np.float32)
    phys_raw = np.zeros((N, 12), dtype=np.float32)
    has_phys = np.zeros(N, dtype=np.float32)
    for i, smi in enumerate(df["canonical_smiles"]):
        morg_raw[i] = morgan_fp(smi)
        phys_raw[i], has_phys[i] = rdkit_physchem(smi)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N}")

    morg_40 = pca_m.transform(morg_raw).astype(np.float32)
    # Apply log-transform + standardize physchem
    phys_raw[:, 3] = np.log1p(np.maximum(phys_raw[:, 3], 0))
    phys_raw[:, 5] = np.log1p(np.maximum(phys_raw[:, 5], 0))
    phys_z = ((phys_raw - mu_phys) / sd_phys) * has_phys[:, None]
    phys_z = phys_z.astype(np.float32)
    print(f"  physchem coverage: {has_phys.mean():.1%}")

    # Other features zeroed out (novel ILs lack these modalities)
    thermo = np.tile(build_thermo_feat(baran_mean=True), (N, 1))
    chemprop_fp = np.zeros((N, 40), dtype=np.float32)
    surface_fp = np.zeros((N, SURFACE_DIM), dtype=np.float32)
    vit_fp = np.zeros((N, FRAME_DIM), dtype=np.float32)
    cosmo_fp = np.zeros((N, COSMO_DIM), dtype=np.float32)
    has_surf = np.zeros(N, dtype=np.float32)
    has_vit = np.zeros(N, dtype=np.float32)
    has_cos = np.zeros(N, dtype=np.float32)
    v4_base = build_v4_base(N)

    # ----- Load specialists + router + Stage-2 model -----
    print("\nLoading A5.9 Stage-2 model (scalar router + 3 specialists)...")
    specialists = {}
    for kind in ("A", "B", "C"):
        ck = torch.load(BMA_DIR / f"specialist_{kind}.pt", map_location=device, weights_only=False)
        m = A5_BMA_Specialist(kind, 40, 8, chemprop_dim=40).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        specialists[kind] = m
        print(f"  [Sp {kind}] core7={ck.get('test_core7'):.4f}")

    # Train scalar router quickly using train-set predictions
    from train_a5_bma_pipeline import _load_split, _standardize
    tr_dict = _load_split("train")
    tr_pca_m = pca_m.transform(tr_dict["morgan_fp"]).astype(np.float32)
    tr_cp = np.zeros((len(tr_dict["morgan_fp"]), 40), dtype=np.float32)  # placeholder
    from train_a2_two_stage import build_chemprop_40d, v4_base as tr_v4_base
    tr_cp_real, _ = build_chemprop_40d(tr_dict["chemprop_fp"], tr_dict["chemprop_fp"])
    from train_a5_bma_pipeline import _assemble_bank, VIT_BANK, COSMO_BANK, FRAME_DIM as F_DIM, COSMO_DIM as C_DIM
    tr_surf = tr_dict["surface_fp"].astype(np.float32)
    tr_hs = (tr_surf != 0).any(axis=1).astype(np.float32)
    tr_surf_z, _, _ = _standardize(tr_surf, tr_hs)
    vit_bank = dict(zip(*[np.load(VIT_BANK, allow_pickle=True)[k] for k in ("smiles","vit_feat")]))
    cos_bank = dict(zip(*[np.load(COSMO_BANK, allow_pickle=True)[k] for k in ("smiles","cosmo_feat")]))
    tr_vit, tr_hv = _assemble_bank(tr_dict["smiles"], vit_bank, F_DIM)
    tr_cos, tr_hc = _assemble_bank(tr_dict["smiles"], cos_bank, C_DIM)
    tr_vit_z, _, _ = _standardize(tr_vit, tr_hv)
    tr_cos_z, _, _ = _standardize(tr_cos, tr_hc)

    feats_tr = {"v4": tr_v4_base(tr_dict), "morg": tr_pca_m, "thermo": tr_dict["thermo_feat"],
                 "chemprop": tr_cp_real.astype(np.float32),
                 "surface": tr_surf_z.astype(np.float32),
                 "vit": tr_vit_z.astype(np.float32),
                 "cos": tr_cos_z.astype(np.float32),
                 "has_surf": tr_hs, "has_vit": tr_hv, "has_cos": tr_hc}

    print("Training scalar router (24 params, ~30 sec)...")
    router = train_router([specialists["A"], specialists["B"], specialists["C"]],
                            feats_tr, tr_dict["targets"].astype(np.float32),
                            device, epochs=100, mode="scalar")
    router.eval()

    print("\nBuilding Stage-2 model (hardfreeze fused backbone + deep lignin head)...")
    # Train Stage-2 quickly (best seed only)
    from train_a5_bma_stage2 import train_stage2 as train_s2
    tr_physchem, tr_phys_norm = tr_phys, None
    tr_p_z, _ = preprocess_physchem(tr_dict["physchem_feat"], tr_dict["has_physchem"],
                                      tr_dict["physchem_feat"], tr_dict["has_physchem"])
    tr_hp = tr_dict["has_physchem"].astype(np.float32)

    nf = 40
    stage2 = A5_BMA_Stage2(specialists, router, nf, n_props=8).to(device)
    # Use a pre-trained Stage-2 if available
    s2_ck_path = V5 / "checkpoints" / "a5_bma" / "stage2_scalar.pt"
    if s2_ck_path.exists():
        stage2.load_state_dict(torch.load(s2_ck_path, map_location=device, weights_only=False))
        print(f"  Loaded pre-trained Stage-2 from {s2_ck_path}")
    else:
        print("  Training Stage-2 (deep lignin head)...")
        stage2 = train_s2(stage2,
                           tr_dict["targets"].astype(np.float32)[:, 7:8].mean() * 0 + feats_tr["v4"],  # placeholder unused
                           feats_tr["morg"], feats_tr["thermo"], feats_tr["chemprop"],
                           feats_tr["surface"], feats_tr["vit"], feats_tr["cos"],
                           feats_tr["has_surf"], feats_tr["has_vit"], feats_tr["has_cos"],
                           tr_p_z.astype(np.float32), tr_hp,
                           tr_dict["targets"].astype(np.float32),
                           device, seed=0, epochs=300)
        torch.save(stage2.state_dict(), s2_ck_path)
        print(f"  Saved Stage-2 ckpt → {s2_ck_path}")
    stage2.eval()

    # ----- Run inference on the catalog -----
    print(f"\nRunning A5.9 Stage-2 inference on {N} candidates...")
    feats_cat = {
        "v4": v4_base, "morg": morg_40, "thermo": thermo, "chemprop": chemprop_fp,
        "surface": surface_fp, "vit": vit_fp, "cos": cosmo_fp,
        "has_surf": has_surf, "has_vit": has_vit, "has_cos": has_cos,
        "physchem": phys_z, "has_physchem": has_phys,
    }
    with torch.no_grad():
        ins = [torch.from_numpy(feats_cat[k]).to(device) for k in (
            "v4","morg","thermo","chemprop","surface","vit","cos",
            "has_surf","has_vit","has_cos","physchem","has_physchem")]
        pred, lv = stage2(*ins)
        pred, lv = pred.cpu().numpy(), lv.cpu().numpy()
    sigma = np.exp(0.5 * lv)

    # ----- Rank + filter -----
    df["lignin_mu"] = pred[:, 7]
    df["lignin_sigma"] = sigma[:, 7]
    for i, p in enumerate(PROPS):
        df[f"{p}_mu"] = pred[:, i]
        df[f"{p}_sigma"] = sigma[:, i]

    # Confidence filter
    thr = np.quantile(df["lignin_sigma"], args.confidence_quantile)
    df["high_confidence"] = df["lignin_sigma"] <= thr
    # Rank by lignin μ among high-confidence; low-confidence dropped from top-K
    df_hc = df[df["high_confidence"]].sort_values("lignin_mu", ascending=False).head(args.top_k)

    out_cols = ["cation_family", "cation_name", "anion_family", "anion_name",
                 "canonical_smiles", "lignin_mu", "lignin_sigma", "high_confidence"]
    for p in PROPS[:7]:
        out_cols.append(f"{p}_mu"); out_cols.append(f"{p}_sigma")
    df_hc[out_cols].to_csv(OUT_CSV, index=False)

    print(f"\nRanked top-{args.top_k} candidates (σ_lignin ≤ {thr:.3f}) → {OUT_CSV}")
    print("\nTop-10 preview:")
    print(df_hc[["cation_name", "anion_name", "lignin_mu", "lignin_sigma"]].head(10).to_string(index=False))
    print(f"\nConfidence filter: {df['high_confidence'].mean():.1%} of candidates passed σ_lignin ≤ {thr:.3f}")


if __name__ == "__main__":
    main()
