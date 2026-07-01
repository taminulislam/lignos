#!/usr/bin/env python3
"""Add lignin solubility as property #10 to the expanded dataset.

Two data sources:
  1. Experimental literature values (~20 ILs with wt% solubility)
  2. COSMO-SAC-derived ln(gamma) for ILs with existing sigma profiles

Outputs updated expanded_v2/cached_{split}.npz with 10D targets.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"

# Experimental lignin solubility data from literature
# Values normalized to [0, 1] scale: 0 = insoluble, 1 = highly soluble
# Original units are wt% (solubility) or % (extraction yield)
LIGNIN_DATA_EXPERIMENTAL = {
    # (canonical_SMILES_or_name_pattern, value_wt_pct, data_type)
    # Green Chemistry 2015 + J Wood Chem 2007 — kraft lignin solubility
    "1-ethyl-3-methylimidazolium acetate": (50.0, "solubility"),
    "1-ethyl-3-methylimidazolium methanesulfonate": (50.0, "solubility"),
    "1-ethyl-3-methylimidazolium trifluoromethanesulfonate": (50.0, "solubility"),
    "1-ethyl-3-methylimidazolium diethyl phosphate": (50.0, "solubility"),
    "1-ethyl-3-methylimidazolium thiocyanate": (50.0, "solubility"),
    "1-ethyl-3-methylimidazolium trifluoroacetate": (50.0, "solubility"),
    "1-butyl-3-methylimidazolium trifluoromethanesulfonate": (40.0, "solubility"),
    "1-butyl-3-methylimidazolium methyl sulfate": (20.0, "solubility"),
    "1-hexyl-3-methylimidazolium trifluoromethanesulfonate": (20.0, "solubility"),
    "1,3-dimethylimidazolium methyl sulfate": (20.0, "solubility"),
    "1-butyl-3-methylimidazolium chloride": (15.0, "solubility"),
    "1-butyl-3-methylimidazolium bromide": (15.0, "solubility"),
    "1-ethyl-3-methylimidazolium tetrafluoroborate": (0.5, "solubility"),
    "1-ethyl-3-methylimidazolium bis((trifluoromethyl)sulfonyl)imide": (0.5, "solubility"),
    # PMC12196028 review — extraction yields from biomass
    "1,1,3,3-tetramethylguanidinium hydrogen sulfate": (81.0, "extraction"),
    "triethylammonium hydrogen sulfate": (80.0, "extraction"),
    "1-butyl-3-methylimidazolium acetate": (71.0, "extraction"),
}


def match_il_name(il_name_db, target_name):
    """Fuzzy match IL names (case-insensitive, substring)."""
    il_lower = il_name_db.lower().strip()
    target_lower = target_name.lower().strip()
    if target_lower in il_lower or il_lower in target_lower:
        return True
    # Handle common abbreviation differences
    il_clean = il_lower.replace("-", "").replace(" ", "")
    target_clean = target_lower.replace("-", "").replace(" ", "")
    if target_clean in il_clean or il_clean in target_clean:
        return True
    return False


def cosmo_sac_lngamma_simple(sigma_il, sigma_lignin, T=298.15):
    """Simplified COSMO-SAC: compute ln(gamma) of lignin in IL.
    
    Uses the restoring free energy approach with misfit and hydrogen-bonding
    contributions. sigma_il and sigma_lignin are 50-bin sigma profiles.
    """
    # COSMO-SAC parameters (Lin & Sandler 2002)
    R = 8.314  # J/(mol·K)
    a_eff = 7.5e-20  # effective segment area (m²)
    
    # Charge density grid: -0.025 to +0.025 e/Å²
    n_bins = len(sigma_il)
    sigma_grid = np.linspace(-0.025, 0.025, n_bins)
    dsigma = sigma_grid[1] - sigma_grid[0]
    
    # Normalize profiles to probability distributions
    p_il = sigma_il / (sigma_il.sum() * dsigma + 1e-12)
    p_lig = sigma_lignin / (sigma_lignin.sum() * dsigma + 1e-12)
    
    # Misfit energy: e_mf(sigma_m, sigma_n) = (a_eff / 2) * f_pol * (sigma_m + sigma_n)²
    f_pol = 2.36e18  # polarization factor (J·m²/C²)
    
    # Hydrogen bonding: e_hb(sigma_m, sigma_n) for sigma_m < -sigma_hb and sigma_n > sigma_hb
    sigma_hb = 0.0084  # e/Å² threshold
    c_hb = 85580.0  # HB constant (J·Å⁴/(mol·e²))
    
    # Solve for segment activity coefficients Gamma in the IL mixture
    # Using the iterative approach: ln(Gamma(sigma_n)) = -ln(sum_m p(sigma_m) * Gamma(sigma_m) * exp(-W(sigma_m, sigma_n)/(RT)))
    
    ln_Gamma_il = np.zeros(n_bins)
    ln_Gamma_lig = np.zeros(n_bins)  # pure lignin reference
    
    for iteration in range(50):
        Gamma_il_old = np.exp(ln_Gamma_il)
        Gamma_lig_old = np.exp(ln_Gamma_lig)
        
        for n in range(n_bins):
            # Exchange energy W(m, n)
            W = np.zeros(n_bins)
            for m in range(n_bins):
                e_mf = (a_eff / 2) * f_pol * (sigma_grid[m] + sigma_grid[n]) ** 2
                e_hb_val = 0.0
                if sigma_grid[m] < -sigma_hb and sigma_grid[n] > sigma_hb:
                    e_hb_val = c_hb * (sigma_grid[m] + sigma_hb) * (sigma_grid[n] - sigma_hb)
                elif sigma_grid[n] < -sigma_hb and sigma_grid[m] > sigma_hb:
                    e_hb_val = c_hb * (sigma_grid[n] + sigma_hb) * (sigma_grid[m] - sigma_hb)
                W[m] = (e_mf + e_hb_val) * 6.022e23  # convert to J/mol
            
            # In IL mixture
            sum_il = np.sum(p_il * Gamma_il_old * np.exp(-W / (R * T))) * dsigma
            ln_Gamma_il[n] = -np.log(max(sum_il, 1e-30))
            
            # In pure lignin
            sum_lig = np.sum(p_lig * Gamma_lig_old * np.exp(-W / (R * T))) * dsigma
            ln_Gamma_lig[n] = -np.log(max(sum_lig, 1e-30))
    
    # ln(gamma) of lignin = sum over lignin segments of [ln(Gamma_IL) - ln(Gamma_pure)]
    ln_gamma = np.sum(p_lig * (ln_Gamma_il - ln_Gamma_lig)) * dsigma
    return float(ln_gamma)


def generate_lignin_sigma_profile(n_bins=50):
    """Generate approximate sigma profile for a lignin model compound.
    
    Uses guaiacylglycerol-β-guaiacyl ether (G-β-O-4 dimer), the most
    common lignin linkage model compound.
    """
    from rdkit.Chem import AllChem, Descriptors
    
    lignin_smi = "COc1cc(C(O)C(CO)Oc2ccccc2OC)ccc1O"  # G-β-O-4 dimer
    mol = Chem.MolFromSmiles(lignin_smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.ComputeGasteigerCharges(mol)
    
    charges = []
    for atom in mol.GetAtoms():
        c = float(atom.GetProp("_GasteigerCharge"))
        if np.isnan(c):
            c = 0.0
        charges.append(c)
    charges = np.array(charges)
    
    # Map Gasteiger charges to sigma-profile bins
    # Gasteiger charges are in e (partial), sigma-profile is in e/Å²
    # Approximate: divide by typical atomic surface area (~10 Å²)
    sigma_vals = charges / 10.0
    
    # Histogram into 50 bins from -0.025 to +0.025 e/Å²
    sigma_grid = np.linspace(-0.025, 0.025, n_bins)
    profile, _ = np.histogram(sigma_vals, bins=np.linspace(-0.025, 0.025, n_bins + 1))
    profile = profile.astype(np.float32)
    
    return profile, sigma_grid


def main():
    print("=" * 60)
    print("Adding lignin targets to expanded dataset")
    print("=" * 60)
    
    # Load current expanded data
    splits = {}
    for split in ["train", "val", "test"]:
        p = V5 / "data" / "expanded_v2" / f"cached_{split}.npz"
        d = np.load(p, allow_pickle=True)
        splits[split] = {k: d[k] for k in d.files}
    
    # Load ILThermo data for name matching
    ilthermo = pd.read_csv(V5 / "data" / "ilthermo_expanded_v2.csv")
    
    # Build SMILES -> IL name mapping from ILThermo
    smiles_to_name = {}
    for _, row in ilthermo.iterrows():
        s = row["il_smiles"]
        n = row["il_name"]
        if isinstance(s, str) and isinstance(n, str):
            smiles_to_name[s] = n
    
    # Also get names from the cached data
    for split in splits:
        for i, smi in enumerate(splits[split]["smiles"]):
            smi_str = smi.decode() if isinstance(smi, bytes) else smi
            il_id = splits[split]["il_ids"][i]
            il_id_str = il_id.decode() if isinstance(il_id, bytes) else str(il_id)
            if smi_str not in smiles_to_name and il_id_str:
                smiles_to_name[smi_str] = il_id_str
    
    # Match experimental lignin data to our ILs
    print(f"\nMatching experimental lignin data ({len(LIGNIN_DATA_EXPERIMENTAL)} entries)...")
    lignin_by_smiles = {}  # smiles -> lignin_solubility_wt_pct
    
    for target_name, (value, dtype) in LIGNIN_DATA_EXPERIMENTAL.items():
        matched = False
        for smi, db_name in smiles_to_name.items():
            if match_il_name(db_name, target_name):
                lignin_by_smiles[smi] = value
                print(f"  MATCH: {target_name[:45]:45s} -> {db_name[:45]:45s} = {value:.1f}%")
                matched = True
                break
        if not matched:
            print(f"  MISS:  {target_name}")
    
    print(f"\nMatched {len(lignin_by_smiles)} ILs with experimental lignin data")
    
    # COSMO-SAC path: compute ln(gamma) for ILs with sigma profiles
    print(f"\nComputing COSMO-SAC lignin ln(gamma) for ILs with sigma profiles...")
    sigma_data = np.load(V5 / "data" / "sigma_profiles.npz", allow_pickle=True)
    
    lignin_profile, sigma_grid = generate_lignin_sigma_profile(n_bins=50)
    print(f"  Lignin model sigma profile: sum={lignin_profile.sum():.1f}, "
          f"n_bins={len(lignin_profile)}")
    
    # Map samples to sigma profiles
    cosmo_lngamma = {}
    for split_name in ["train", "val", "test"]:
        sigma_split = sigma_data[split_name]  # (N, 50)
        smiles_arr = splits[split_name]["smiles"]
        for i, smi in enumerate(smiles_arr):
            smi_str = smi.decode() if isinstance(smi, bytes) else smi
            if smi_str in cosmo_lngamma:
                continue
            sigma_il = sigma_split[i]
            if sigma_il.sum() < 0.1:
                continue
            try:
                lng = cosmo_sac_lngamma_simple(sigma_il, lignin_profile)
                if np.isfinite(lng):
                    cosmo_lngamma[smi_str] = lng
            except Exception:
                pass
    
    print(f"  Computed COSMO-SAC ln(gamma) for {len(cosmo_lngamma)} ILs")
    if cosmo_lngamma:
        vals = list(cosmo_lngamma.values())
        print(f"  Range: [{min(vals):.3f}, {max(vals):.3f}]  mean={np.mean(vals):.3f}")
    
    # Add lignin column to each split
    # Use experimental data where available, COSMO-SAC as fallback
    for split_name in splits:
        data = splits[split_name]
        targets = data["targets"]  # (N, 9) currently
        n = targets.shape[0]
        
        lignin_col = np.full(n, np.nan, dtype=np.float32)
        n_exp = n_cosmo = 0
        
        for i, smi in enumerate(data["smiles"]):
            smi_str = smi.decode() if isinstance(smi, bytes) else smi
            if smi_str in lignin_by_smiles:
                lignin_col[i] = lignin_by_smiles[smi_str]
                n_exp += 1
            elif smi_str in cosmo_lngamma:
                # Convert ln(gamma) to a proxy wt% scale
                # More negative ln(gamma) = better solubility
                # Map roughly: ln(gamma)=-5 -> 50 wt%, ln(gamma)=0 -> 10 wt%, ln(gamma)=5 -> 0 wt%
                lng = cosmo_lngamma[smi_str]
                proxy_wt = max(0, 50 - 8 * lng)
                lignin_col[i] = proxy_wt
                n_cosmo += 1
        
        # Normalize lignin values (0-100 wt% -> standardized)
        valid = lignin_col[~np.isnan(lignin_col)]
        if len(valid) > 1:
            mean_lig = valid.mean()
            std_lig = valid.std()
            if std_lig < 1e-8:
                std_lig = 1.0
            lignin_col_normed = (lignin_col - mean_lig) / std_lig
        else:
            lignin_col_normed = lignin_col
        
        # Append as column 10
        new_targets = np.concatenate([targets, lignin_col_normed[:, None]], axis=1)
        
        # Also extend preds_fusion and preds_chemprop to 10D
        pf = data["preds_fusion"]
        pc = data["preds_chemprop"]
        pf_ext = np.concatenate([pf, np.zeros((n, 1), dtype=np.float32)], axis=1)
        pc_ext = np.concatenate([pc, np.zeros((n, 1), dtype=np.float32)], axis=1)
        
        data["targets"] = new_targets
        data["preds_fusion"] = pf_ext
        data["preds_chemprop"] = pc_ext
        
        n_valid = (~np.isnan(lignin_col)).sum()
        print(f"\n{split_name}: {n_valid}/{n} lignin targets ({n_exp} experimental, {n_cosmo} COSMO-SAC)")
    
    # Save updated files
    out_dir = V5 / "data" / "expanded_v2"
    for split_name, data in splits.items():
        out_path = out_dir / f"cached_{split}.npz"
        np.savez(out_path, **data)
    
    print(f"\nSaved updated files with lignin as property #10")
    print(f"Target shape: {splits['train']['targets'].shape}")


if __name__ == "__main__":
    main()
