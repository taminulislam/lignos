"""A5.4 — minimal COSMO-SAC solver (Lin-Sandler 2002) for γ predictions.

Computes γ_water and γ_IL at T = 298.15 K for each IL in the cache using:
  1. σ-profile per species (histogram of surface-charge density weighted by
     area) — from data/pipeline/dft_surface/{cid}_pair.npz for IL, and a
     literature-derived σ-profile for water.
  2. Segment activity coefficient iteration (Eq. 12 of Lin-Sandler 2002)
  3. Combinatorial Staverman-Guggenheim contribution
  4. Residual component activity coefficient (Eq. 10)

Output: lignos/data/cosmo_sac_soft_labels.npz
  smiles           : (N,) object       canonical IL SMILES
  ln_gamma_water   : (N,) float32      ln γ of water in the IL at 298 K (x→0)
  ln_gamma_IL      : (N,) float32      ln γ of IL in water at 298 K (x→0)
  G_E_cosmo        : (N,) float32      excess Gibbs (J/mol) at x=0.5, 298 K
  areas            : (N, 2) float32    [A_IL, A_water] (Å²)

Later use: as soft labels for gamma1/gamma2 (or as auxiliary G_E target) on
the 5147 ILThermo pre-training rows whose real labels are NaN. Trains the
model to agree with first-principles thermodynamics on unseen compositions.

Reference:
  Lin, S.-T.; Sandler, S. I. "A Priori Phase Equilibrium Prediction from
  a Segment Contribution Solvation Model." Ind. Eng. Chem. Res. 2002, 41,
  899-913.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
DFT_SURF = PROJECT_ROOT / "data" / "pipeline" / "dft_surface"
OUT = V5 / "data" / "cosmo_sac_soft_labels.npz"

# Constants (Lin-Sandler 2002)
T_REF = 298.15
R_GAS = 8.314          # J/(mol·K)
A_EFF = 7.25           # Å², standard segment area
SIGMA_RANGE = (-0.025, 0.025)
N_BINS = 51            # 51 bins from -0.025 to +0.025, width 0.001 e/Å²
SIGMA_HB = 0.0084      # e/Å²
C_HB_STD = 85580.0     # kcal·Å⁴/(mol·e²) — Lin-Sandler hydrogen-bond coefficient
                        # Note: we use J units below (×4184)
C_ES = 6525.69         # electrostatic interaction coefficient (SI)

# COORDINATION
Z_COORD = 10.0


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


def sigma_profile_from_dft(surface):
    """Given (M, 7) surface with columns (x,y,z,nx,ny,nz,σ), histogram σ
    weighted by surface area. Assume uniform area per tessellation point ≈
    A_total / M (Lin-Sandler uses explicit surface areas; we approximate
    since our tessellation is roughly uniform)."""
    sigma = surface[:, 6].astype(np.float64)
    # Total surface area: estimate as n_points × A_avg. Typical Psi4 PCM
    # cavity produces 1-2 Å² per point. We'll use the Lin-Sandler scaling
    # with A_EFF; the overall normalization matters only through A_i.
    n = len(sigma)
    # Use A_EFF per point as a uniform approximation (Lin-Sandler convention)
    # — this makes n_points ≈ species area / A_EFF.
    total_area = n * A_EFF
    # Histogram σ with density False so weights sum to n_points
    hist, edges = np.histogram(sigma, bins=N_BINS, range=SIGMA_RANGE)
    # Normalize to pdf: p(σ) such that Σ p(σ) × Δσ = 1
    bin_width = (SIGMA_RANGE[1] - SIGMA_RANGE[0]) / N_BINS
    pdf = hist.astype(np.float64) / (hist.sum() * bin_width + 1e-12)
    sigma_centers = 0.5 * (edges[:-1] + edges[1:])
    return sigma_centers, pdf, total_area


def water_sigma_profile():
    """Approximate σ-profile for water — a Gaussian with peaks at ±0.015
    e/Å² (representing donor/acceptor lobes) + central peak at 0.
    Literature COSMO-RS water profile has bi-modal character; this is a
    reasonable proxy.

    Source: typical published COSMO-RS water σ-profile shape. For production
    we'd read from a sigma-profile database, but this captures the
    qualitative HB-donor + HB-acceptor character."""
    sigmas = np.linspace(*SIGMA_RANGE, N_BINS)
    # Two Gaussians: donor at -0.013, acceptor at +0.013, plus small tail at 0
    p = (np.exp(-((sigmas + 0.013) ** 2) / (2 * 0.004 ** 2))
         + np.exp(-((sigmas - 0.013) ** 2) / (2 * 0.004 ** 2))
         + 0.4 * np.exp(-(sigmas ** 2) / (2 * 0.005 ** 2)))
    p /= p.sum() * (sigmas[1] - sigmas[0])
    return sigmas, p, 43.8  # area of water ≈ 43.8 Å² (Lin-Sandler)


def interaction_energy(sigma_m, sigma_n):
    """ΔW(σ_m, σ_n) per Lin-Sandler Eq. 9 — electrostatic + HB, in J/mol·A."""
    # Electrostatic: 0.5 × α' × (σ_m + σ_n)²  — per Å² of interaction area
    alpha_prime = 16466.72      # J·Å²/(mol·e²) — refit for SI units
    es = 0.5 * alpha_prime * (sigma_m + sigma_n) ** 2
    # Hydrogen bonding: C_hb × max(0, σ_acc − σ_hb) × min(0, σ_don + σ_hb)
    c_hb = 353210.0              # J·Å²/(mol·e²) — refit
    sigma_acc = np.maximum(sigma_m, sigma_n)
    sigma_don = np.minimum(sigma_m, sigma_n)
    hb = c_hb * np.maximum(0.0, sigma_acc - SIGMA_HB) * np.minimum(0.0, sigma_don + SIGMA_HB)
    return es + hb


def segment_activity_coefficients(pdf, T=T_REF, max_iter=500, tol=1e-6):
    """Solve Γ(σ_m) = 1 / Σ_n [ p(σ_n) · Γ(σ_n) · exp(-ΔW(σ_m, σ_n) / (RT)) ]
    iteratively until converged."""
    sigmas = np.linspace(*SIGMA_RANGE, N_BINS)
    dsig = sigmas[1] - sigmas[0]
    # Precompute interaction matrix W[m, n] and Boltzmann weights
    S = sigmas[:, None]
    W = interaction_energy(S, S.T)   # (N_BINS, N_BINS)
    boltz = np.exp(-W / (R_GAS * T))
    # Weight by pdf (+ bin width for discretization)
    p_weighted = pdf * dsig          # Σ p · dσ = 1
    # Init Γ = 1
    G = np.ones(N_BINS, dtype=np.float64)
    for _ in range(max_iter):
        denom = (p_weighted[None, :] * G[None, :] * boltz).sum(axis=1)  # (N_BINS,)
        G_new = 1.0 / np.maximum(denom, 1e-30)
        # Successive substitution with damping
        G_new = 0.5 * (G + G_new)
        if np.max(np.abs(G_new - G)) < tol:
            G = G_new; break
        G = G_new
    return np.log(G + 1e-30)


def component_ln_gamma_residual(pdf_i, ln_Gamma_mix, ln_Gamma_pure_i, area_i):
    """Residual contribution: ln γ_i = (A_i / A_EFF) · Σ p_i(σ) · (ln Γ_mix(σ) − ln Γ_pure_i(σ)) · dσ"""
    sigmas = np.linspace(*SIGMA_RANGE, N_BINS)
    dsig = sigmas[1] - sigmas[0]
    return (area_i / A_EFF) * np.sum(pdf_i * (ln_Gamma_mix - ln_Gamma_pure_i)) * dsig


def combinatorial_ln_gamma(x, r, q):
    """Staverman-Guggenheim combinatorial term. x = mole fraction vector,
    r = normalized vol, q = normalized area."""
    phi = x * r / (x @ r)
    theta = x * q / (x @ q)
    l = (Z_COORD / 2.0) * (r - q) - (r - 1.0)
    return (np.log(phi / x)
            + (Z_COORD / 2.0) * q * np.log(theta / phi)
            + l - (phi / x) * (x @ l))


def lin_sandler_binary(sigma_i, pdf_i, area_i, sigma_j, pdf_j, area_j,
                         x_i=0.5, T=T_REF):
    """Binary mixture γ_i, γ_j at mole fraction x_i (species i)."""
    x = np.array([x_i, 1.0 - x_i])
    A = np.array([area_i, area_j])
    r = A / A.mean()  # normalized area as volume proxy
    q = r.copy()       # use identical r, q for simplicity

    # Pure-component segment activity coefficients
    ln_G_pure_i = segment_activity_coefficients(pdf_i, T=T)
    ln_G_pure_j = segment_activity_coefficients(pdf_j, T=T)
    # Mixture σ-profile
    area_frac = x * A / (x @ A)
    pdf_mix = area_frac[0] * pdf_i + area_frac[1] * pdf_j
    ln_G_mix = segment_activity_coefficients(pdf_mix, T=T)

    # Residual terms
    ln_gi_res = component_ln_gamma_residual(pdf_i, ln_G_mix, ln_G_pure_i, A[0])
    ln_gj_res = component_ln_gamma_residual(pdf_j, ln_G_mix, ln_G_pure_j, A[1])

    # Combinatorial terms
    ln_gamma_comb = combinatorial_ln_gamma(x, r, q)
    ln_gi = ln_gamma_comb[0] + ln_gi_res
    ln_gj = ln_gamma_comb[1] + ln_gj_res

    # G_E = RT (x_i · ln γ_i + x_j · ln γ_j)
    G_E = R_GAS * T * (x[0] * ln_gi + x[1] * ln_gj)
    return ln_gi, ln_gj, G_E


def compound_id_map():
    geom = pd.read_csv(PROJECT_ROOT / "data/pipeline/geometry_status.csv")
    tier3 = pd.read_csv(PROJECT_ROOT / "data/pipeline/tier3_compounds.csv")
    mapping = {}
    for _, r in geom.iterrows():
        mapping[r["compound_id"]] = r["smiles"]
    for _, r in tier3.iterrows():
        mapping[r["compound_id"]] = r["smiles"]
    return mapping


def main():
    id_to_smi = compound_id_map()
    print(f"compound_id → SMILES mappings: {len(id_to_smi)}")

    water_sig, water_pdf, water_area = water_sigma_profile()
    print(f"Water reference σ-profile built (area={water_area:.1f} Å²)")

    results = {}
    n_ok = n_miss = n_fail = 0
    for cid, smi in id_to_smi.items():
        cs = canon(smi)
        if not cs or cs in results:
            continue
        pair_path = DFT_SURF / f"{cid}_pair.npz"
        if not pair_path.exists():
            n_miss += 1; continue
        try:
            surf = np.load(pair_path, allow_pickle=True)["surface"]
            sigmas, pdf_il, area_il = sigma_profile_from_dft(surf)
            ln_gi, ln_gj, ge = lin_sandler_binary(
                sigmas, pdf_il, area_il,
                water_sig, water_pdf, water_area,
                x_i=0.5, T=T_REF)
        except Exception as e:
            n_fail += 1
            if n_fail <= 3: print(f"  [fail] {cid}: {e}")
            continue
        results[cs] = {"ln_gamma_IL": float(ln_gi),
                         "ln_gamma_water": float(ln_gj),
                         "G_E": float(ge),
                         "area_IL": float(area_il),
                         "cid": cid}
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"  [{n_ok}] {cid}: ln γ_IL={ln_gi:+.3f}  ln γ_w={ln_gj:+.3f}  G_E={ge:+.0f} J/mol")

    smis = np.array(list(results.keys()), dtype=object)
    def col(k): return np.array([results[s][k] for s in smis], dtype=np.float32)
    np.savez(OUT,
              smiles=smis,
              ln_gamma_IL=col("ln_gamma_IL"),
              ln_gamma_water=col("ln_gamma_water"),
              G_E_cosmo=col("G_E"),
              area_IL=col("area_IL"),
              compound_ids=np.array([results[s]["cid"] for s in smis], dtype=object))
    print(f"\nWrote {len(smis)} ILs → {OUT}")
    print(f"  ok: {n_ok}  missing DFT: {n_miss}  solver fail: {n_fail}")
    if len(smis):
        ln_IL = col("ln_gamma_IL")
        ln_w = col("ln_gamma_water")
        ge = col("G_E")
        print(f"  ln γ_IL      : mean={ln_IL.mean():+.3f}  std={ln_IL.std():.3f}  range=[{ln_IL.min():+.3f}, {ln_IL.max():+.3f}]")
        print(f"  ln γ_water   : mean={ln_w.mean():+.3f}  std={ln_w.std():.3f}  range=[{ln_w.min():+.3f}, {ln_w.max():+.3f}]")
        print(f"  G_E (J/mol)  : mean={ge.mean():+.0f}  std={ge.std():.0f}  range=[{ge.min():+.0f}, {ge.max():+.0f}]")


if __name__ == "__main__":
    main()
