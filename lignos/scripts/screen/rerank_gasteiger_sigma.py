"""LIGNOS revision C3 — re-rank top lignin candidates with a Gasteiger-charge
proxy sigma-profile so Specialist C contributes real signal.

>>> STAGED COPY <<<
The intended final location is lignos/scripts/screen/rerank_gasteiger_sigma.py,
but that directory is owned by user kahmed2 and is NOT writable by tislam6, so this
script is staged at the project root. Move it into place once write access is granted:
    mv rerank_gasteiger_sigma.PENDING.py lignos/scripts/screen/rerank_gasteiger_sigma.py
It also CANNOT RUN until read access is granted to:
    lignos/data/LignoIL_A1/            (training cache; dir mode 700, kahmed2)
    lignos/data/cosmo_sac_feat_bank.npz (DFT cos descriptor bank; mode 600)
    data/virtual_screen/il_catalog.csv          (catalog fallback; dir mode 700)
See REPORT for the exact chmod/setfacl the file owner needs to run.

Problem this fixes
------------------
screen_novel_ils.py sets cos=0 and has_cos=0 for novel candidates. That masks out
Specialist C (the COSMO-SAC branch is gated by has_cos), collapsing the A5.9 ensemble
onto the A2/SMILES backbone and producing non-physical lignin yields (~0.006-0.007).

This script populates `cos` (20-D) for each candidate with a CHEAP proxy sigma-profile
descriptor derived from RDKit ETKDG geometry + Gasteiger partial charges (same machinery
as generate_proxy_point_clouds.py), sets has_cos=1, and calibrates+standardizes it into
the DFT cos feature space Specialist C was trained on.

Scientific guardrail
--------------------
Gasteiger charges (~+/-0.5) are on a DIFFERENT scale than DFT screening charge densities
(~+/-0.025 e/A^2). So we (1) compute proxy descriptors for every training IL that also has
a real DFT cos feature, (2) report per-dim + overall Pearson r (the r>0.92 Gasteiger-vs-DFT
check), (3) fit + apply a per-dim linear calibration proxy->DFT, then standardize with the
TRAINING DFT mu_c/sd_c, and (4) sanity-check the new lignin_mu range vs the old degenerate one.

Outputs
-------
  results/virtual_screening/top_lignin_gasteiger_reranked.csv
  results/virtual_screening/gasteiger_proxy_validation.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.decomposition import PCA

# When staged at root, the existing modules live under lignos/scripts.
# When moved into scripts/screen/, parent[2] == lignos. Handle both.
_here = Path(__file__).resolve()
if _here.parent.name == "screen":
    PROJECT_ROOT = _here.parent.parent.parent.parent
else:  # staged at project root
    PROJECT_ROOT = _here.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
sys.path.insert(0, str(V5 / "scripts" / "screen"))

from audit_residuals import PROPS  # noqa: E402
from train_a2_two_stage import preprocess_physchem, build_chemprop_40d, v4_base as tr_v4_base  # noqa: E402
from train_a5_bma_pipeline import (  # noqa: E402
    A5_BMA_Specialist, train_router, _load_split, _standardize, _assemble_bank,
    VIT_BANK, COSMO_BANK, FRAME_DIM as F_DIM, COSMO_DIM as C_DIM,
)
from train_a5_bma_stage2 import A5_BMA_Stage2, train_stage2 as train_s2  # noqa: E402
from generate_proxy_point_clouds import embed_il, sample_surface  # noqa: E402
from build_cosmo_sac_features import compute_descriptors, canon  # noqa: E402

CACHE = V5 / "data" / "LignoIL_A1"
BMA_DIR = V5 / "checkpoints" / "a5_bma"
S2_CKPT = BMA_DIR / "stage2_scalar.pt"

TOP_LIGNIN_CSV = PROJECT_ROOT / "results" / "virtual_screening" / "top_lignin.csv"
CATALOG_CSV = PROJECT_ROOT / "data" / "virtual_screen" / "il_catalog.csv"  # fallback
OUT_CSV = PROJECT_ROOT / "results" / "virtual_screening" / "top_lignin_gasteiger_reranked.csv"
OUT_JSON = PROJECT_ROOT / "results" / "virtual_screening" / "gasteiger_proxy_validation.json"

SURFACE_DIM = 256
N_PROXY_CONF = 3


def morgan_fp(smi, nbits=2048):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_physchem(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return np.zeros(12, dtype=np.float32), 0.0
    try:
        feats = np.array([
            Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.NumHDonors(m),
            Descriptors.NumHAcceptors(m), Descriptors.TPSA(m),
            Descriptors.NumRotatableBonds(m), Descriptors.HeavyAtomCount(m),
            Descriptors.NumAromaticRings(m), Descriptors.NumAliphaticRings(m),
            Descriptors.FractionCSP3(m), Descriptors.NumHeteroatoms(m),
            Descriptors.RingCount(m),
        ], dtype=np.float32)
        return feats, 1.0
    except Exception:
        return np.zeros(12, dtype=np.float32), 0.0


def build_thermo_feat():
    t = np.zeros(25, dtype=np.float32)
    t[0] = 0.5; t[1] = 0.2; t[2] = 0.5
    return t


def proxy_descriptor(smi: str, n_conf: int = N_PROXY_CONF):
    descs = []
    for seed in range(n_conf):
        try:
            mol, charges = embed_il(smi, seed)
            pc = sample_surface(mol, charges, seed)
            d = compute_descriptors(pc[:, 6])
            if np.all(np.isfinite(d)):
                descs.append(d)
        except Exception:
            continue
    if not descs:
        return None
    return np.mean(descs, axis=0).astype(np.float32)


def fit_per_dim_calibration(proxy: np.ndarray, dft: np.ndarray):
    D = proxy.shape[1]
    a = np.ones(D, dtype=np.float64)
    b = np.zeros(D, dtype=np.float64)
    identity = np.zeros(D, dtype=bool)
    for d in range(D):
        px = proxy[:, d].astype(np.float64)
        dy = dft[:, d].astype(np.float64)
        if px.std() < 1e-8 or not np.all(np.isfinite(px)) or not np.all(np.isfinite(dy)):
            identity[d] = True
            continue
        A = np.vstack([px, np.ones_like(px)]).T
        sol, *_ = np.linalg.lstsq(A, dy, rcond=None)
        if not np.all(np.isfinite(sol)):
            identity[d] = True
            continue
        a[d], b[d] = sol[0], sol[1]
    return a, b, identity


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ===== 1. Candidate list =====
    if TOP_LIGNIN_CSV.exists():
        df = pd.read_csv(TOP_LIGNIN_CSV)
        smi_col = "smiles" if "smiles" in df.columns else (
            "canonical_smiles" if "canonical_smiles" in df.columns else None)
        if smi_col is None:
            raise RuntimeError(f"No SMILES column in {TOP_LIGNIN_CSV}: {list(df.columns)}")
        src = str(TOP_LIGNIN_CSV)
    else:
        df = pd.read_csv(CATALOG_CSV)
        smi_col = "canonical_smiles"
        src = str(CATALOG_CSV)
    print(f"Candidates: {len(df)} from {src} (SMILES col='{smi_col}')")

    old_lignin, old_col = None, None
    for c in ("lignin_score", "lignin_mu"):
        if c in df.columns:
            old_lignin = df[c].to_numpy(dtype=float); old_col = c; break

    # ===== 2. Train-cache refs =====
    tr = _load_split("train")
    pca_m = PCA(40).fit(tr["morgan_fp"])
    tr_phys = tr["physchem_feat"].astype(np.float32).copy()
    tr_has_phys = tr["has_physchem"].astype(bool)
    tr_phys[:, 3] = np.log1p(np.maximum(tr_phys[:, 3], 0))
    tr_phys[:, 5] = np.log1p(np.maximum(tr_phys[:, 5], 0))
    mu_phys = tr_phys[tr_has_phys].mean(axis=0)
    sd_phys = tr_phys[tr_has_phys].std(axis=0) + 1e-6

    cos_bank = dict(zip(*[np.load(COSMO_BANK, allow_pickle=True)[k]
                          for k in ("smiles", "cosmo_feat")]))
    tr_cos, tr_hc = _assemble_bank(tr["smiles"], cos_bank, C_DIM)
    tr_cos_z, mu_c, sd_c = _standardize(tr_cos, tr_hc)
    print(f"Training DFT cos coverage: {tr_hc.mean():.1%} ({int(tr_hc.sum())} rows)")

    # ===== 3. PROXY-vs-DFT VALIDATION =====
    seen, overlap_smiles = set(), []
    for s in tr["smiles"]:
        cs = canon(s)
        if cs and cs in cos_bank and cs not in seen:
            seen.add(cs); overlap_smiles.append(cs)
    print(f"\nValidation: {len(overlap_smiles)} unique training ILs with DFT cos.")

    proxy_rows, dft_rows = [], []
    for i, cs in enumerate(overlap_smiles):
        pdz = proxy_descriptor(cs)
        if pdz is None:
            continue
        proxy_rows.append(pdz); dft_rows.append(np.asarray(cos_bank[cs], dtype=np.float32))
        if (i + 1) % 20 == 0:
            print(f"  proxy descriptors {i+1}/{len(overlap_smiles)}")
    proxy_arr = np.stack(proxy_rows); dft_arr = np.stack(dft_rows)
    M = proxy_arr.shape[0]
    print(f"  built proxy+DFT descriptors for {M} overlapping ILs")

    per_dim_r = [pearson(proxy_arr[:, d], dft_arr[:, d]) for d in range(C_DIM)]
    def zstd(a):
        return (a - a.mean(0)) / (a.std(0) + 1e-12)
    overall_r = pearson(zstd(proxy_arr).ravel(), zstd(dft_arr).ravel())
    finite_r = [r for r in per_dim_r if np.isfinite(r)]
    print(f"\n=== PROXY-vs-DFT (n={M}) ===")
    print(f"  overall Pearson r: {overall_r:.4f}")
    print(f"  per-dim r: median {np.median(finite_r):.4f} mean {np.mean(finite_r):.4f} "
          f"min {np.min(finite_r):.4f} max {np.max(finite_r):.4f}")
    for d, r in enumerate(per_dim_r):
        print(f"    dim {d:2d}: r={r:.4f}" if np.isfinite(r) else f"    dim {d:2d}: nan")

    a_cal, b_cal, used_identity = fit_per_dim_calibration(proxy_arr, dft_arr)
    print(f"Calibration: {int(used_identity.sum())}/{C_DIM} dims -> identity fallback.")

    # ===== 4. Candidate features =====
    N = len(df); smis = df[smi_col].tolist()
    print(f"\nCandidate features for {N} ILs...")
    morg_raw = np.zeros((N, 2048), dtype=np.float32)
    phys_raw = np.zeros((N, 12), dtype=np.float32)
    has_phys = np.zeros(N, dtype=np.float32)
    proxy_cand = np.zeros((N, C_DIM), dtype=np.float32)
    has_cos = np.zeros(N, dtype=np.float32)
    for i, smi in enumerate(smis):
        morg_raw[i] = morgan_fp(smi)
        phys_raw[i], has_phys[i] = rdkit_physchem(smi)
        pdz = proxy_descriptor(smi)
        if pdz is not None:
            proxy_cand[i] = pdz; has_cos[i] = 1.0

    morg_40 = pca_m.transform(morg_raw).astype(np.float32)
    phys_raw[:, 3] = np.log1p(np.maximum(phys_raw[:, 3], 0))
    phys_raw[:, 5] = np.log1p(np.maximum(phys_raw[:, 5], 0))
    phys_z = (((phys_raw - mu_phys) / sd_phys) * has_phys[:, None]).astype(np.float32)

    cos_cal = proxy_cand * a_cal[None, :] + b_cal[None, :]
    cos_z = (((cos_cal - mu_c) / sd_c) * has_cos[:, None]).astype(np.float32)
    print(f"  has_cos coverage on candidates: {has_cos.mean():.1%}")

    thermo = np.tile(build_thermo_feat(), (N, 1)).astype(np.float32)
    chemprop_fp = np.zeros((N, 40), dtype=np.float32)
    surface_fp = np.zeros((N, SURFACE_DIM), dtype=np.float32)
    vit_fp = np.zeros((N, F_DIM), dtype=np.float32)
    has_surf = np.zeros(N, dtype=np.float32)
    has_vit = np.zeros(N, dtype=np.float32)
    v4_base_cand = np.zeros((N, 8), dtype=np.float32)

    # ===== 5. Specialists / router / Stage-2 =====
    print("\nLoading specialists A/B/C...")
    specialists = {}
    for kind in ("A", "B", "C"):
        ck = torch.load(BMA_DIR / f"specialist_{kind}.pt", map_location=device, weights_only=False)
        m = A5_BMA_Specialist(kind, 40, 8, chemprop_dim=40).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        specialists[kind] = m

    tr_pca_m = pca_m.transform(tr["morgan_fp"]).astype(np.float32)
    tr_cp_real, _ = build_chemprop_40d(tr["chemprop_fp"], tr["chemprop_fp"])
    tr_surf = tr["surface_fp"].astype(np.float32)
    tr_hs = (tr_surf != 0).any(axis=1).astype(np.float32)
    tr_surf_z, _, _ = _standardize(tr_surf, tr_hs)
    vit_bank = dict(zip(*[np.load(VIT_BANK, allow_pickle=True)[k] for k in ("smiles", "vit_feat")]))
    tr_vit, tr_hv = _assemble_bank(tr["smiles"], vit_bank, F_DIM)
    tr_vit_z, _, _ = _standardize(tr_vit, tr_hv)

    feats_tr = {"v4": tr_v4_base(tr), "morg": tr_pca_m, "thermo": tr["thermo_feat"],
                "chemprop": tr_cp_real.astype(np.float32),
                "surface": tr_surf_z.astype(np.float32),
                "vit": tr_vit_z.astype(np.float32),
                "cos": tr_cos_z.astype(np.float32),
                "has_surf": tr_hs, "has_vit": tr_hv, "has_cos": tr_hc}

    print("Training scalar router...")
    router = train_router([specialists["A"], specialists["B"], specialists["C"]],
                          feats_tr, tr["targets"].astype(np.float32),
                          device, epochs=100, mode="scalar")
    router.eval()

    stage2 = A5_BMA_Stage2(specialists, router, 40, n_props=8).to(device)
    if S2_CKPT.exists():
        stage2.load_state_dict(torch.load(S2_CKPT, map_location=device, weights_only=False))
        print(f"Loaded Stage-2 ckpt {S2_CKPT.name}")
    else:
        tr_p_z, _ = preprocess_physchem(tr["physchem_feat"], tr["has_physchem"],
                                        tr["physchem_feat"], tr["has_physchem"])
        tr_hp = tr["has_physchem"].astype(np.float32)
        stage2 = train_s2(stage2, feats_tr["v4"], feats_tr["morg"], feats_tr["thermo"],
                          feats_tr["chemprop"], feats_tr["surface"], feats_tr["vit"],
                          feats_tr["cos"], feats_tr["has_surf"], feats_tr["has_vit"],
                          feats_tr["has_cos"], tr_p_z.astype(np.float32), tr_hp,
                          tr["targets"].astype(np.float32), device, seed=0, epochs=300)
        torch.save(stage2.state_dict(), S2_CKPT)
    stage2.eval()

    # ===== 6. Inference with Specialist C active =====
    print(f"\nInference on {N} candidates (Specialist C active)...")
    feats_cat = {"v4": v4_base_cand, "morg": morg_40, "thermo": thermo,
                 "chemprop": chemprop_fp, "surface": surface_fp, "vit": vit_fp,
                 "cos": cos_z, "has_surf": has_surf, "has_vit": has_vit,
                 "has_cos": has_cos, "physchem": phys_z, "has_physchem": has_phys}
    with torch.no_grad():
        ins = [torch.from_numpy(feats_cat[k]).to(device) for k in (
            "v4", "morg", "thermo", "chemprop", "surface", "vit", "cos",
            "has_surf", "has_vit", "has_cos", "physchem", "has_physchem")]
        pred, lv = stage2(*ins)
        pred, lv = pred.cpu().numpy(), lv.cpu().numpy()
    new_lignin = pred[:, 7]; new_sigma = np.exp(0.5 * lv[:, 7])

    # ===== 7. Re-rank + diagnostics =====
    df_out = df.copy()
    df_out["new_lignin_mu"] = new_lignin
    df_out["new_lignin_sigma"] = new_sigma
    new_rank = (-new_lignin).argsort().argsort()
    df_out["new_rank"] = new_rank
    if old_lignin is not None:
        df_out["old_lignin_mu"] = old_lignin
        old_rank = (-old_lignin).argsort().argsort()
        df_out["old_rank"] = old_rank
        df_out["rank_change"] = old_rank - new_rank

    cat_col = "cation" if "cation" in df_out.columns else None
    an_col = "anion" if "anion" in df_out.columns else None
    cols = ([cat_col] if cat_col else []) + ([an_col] if an_col else []) + [smi_col]
    if old_lignin is not None: cols.append("old_lignin_mu")
    cols += ["new_lignin_mu", "new_lignin_sigma"]
    cols += (["old_rank", "new_rank", "rank_change"] if old_lignin is not None else ["new_rank"])
    df_sorted = df_out.sort_values("new_lignin_mu", ascending=False)
    df_sorted[cols].to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    spearman, top10_overlap = float("nan"), None
    if old_lignin is not None:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(old_lignin, new_lignin).correlation)
        top10_overlap = len(set(np.argsort(-old_lignin)[:10].tolist())
                            & set(np.argsort(-new_lignin)[:10].tolist()))

    validation = {
        "candidate_source": src, "n_candidates": int(N),
        "n_overlap_ils_with_dft": int(M),
        "proxy_vs_dft_overall_pearson_r": overall_r,
        "proxy_vs_dft_per_dim_pearson_r": [None if not np.isfinite(r) else round(float(r), 4)
                                           for r in per_dim_r],
        "proxy_vs_dft_median_per_dim_r": float(np.median(finite_r)),
        "proxy_vs_dft_mean_per_dim_r": float(np.mean(finite_r)),
        "calibration_dims_identity_fallback": [int(x) for x in np.where(used_identity)[0]],
        "candidate_has_cos_coverage": float(has_cos.mean()),
        "old_lignin_mu_column": old_col,
        "old_lignin_mu_range": ([float(old_lignin.min()), float(old_lignin.max())]
                                if old_lignin is not None else None),
        "new_lignin_mu_range": [float(new_lignin.min()), float(new_lignin.max())],
        "new_lignin_mu_mean": float(new_lignin.mean()),
        "new_lignin_mu_std": float(new_lignin.std()),
        "spearman_old_vs_new": spearman,
        "top10_overlap_old_vs_new": top10_overlap,
        "top10_new": df_sorted[cols].head(10).to_dict(orient="records"),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(validation, f, indent=2, default=str)
    print(f"Wrote {OUT_JSON}")

    print("\n=== SANITY ===")
    if old_lignin is not None:
        print(f"  old lignin range: [{old_lignin.min():.4f}, {old_lignin.max():.4f}]")
    print(f"  new lignin range: [{new_lignin.min():.4f}, {new_lignin.max():.4f}] "
          f"(mean {new_lignin.mean():.4f}, std {new_lignin.std():.4f})")
    print(f"  overall proxy-vs-DFT r = {overall_r:.4f} (n={M})")
    if old_lignin is not None:
        print(f"  Spearman(old,new) = {spearman:.4f}  top-10 overlap = {top10_overlap}/10")
    print("\nNew top-10:")
    print(df_sorted[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
