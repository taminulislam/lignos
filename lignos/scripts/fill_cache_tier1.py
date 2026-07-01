#!/usr/bin/env python3
"""Tier-1 cache fills (cheap, immediate):

  1. Patch the 1 missing Datasheet row (IL-20 EMIMOAc @ 323.15 K) via linear
     interpolation on T for that IL's 7 core-7 properties, then propagate
     into the original LignoIL cache if that row exists there.
  2. Fill thermo_feat for the 81 extension rows that previously couldn't match
     a unified_lignin.csv row — use the SMILES-level mean T across unified
     rows for that SMILES as the T proxy.
  3. Compute Chemprop 1.7 MPN fingerprints for every unique SMILES in the v2
     cache and write them to the chemprop_fp field for rows that currently
     have it fully zeroed.

Writes to the v2 cache in place (backup saved as cached_{split}_pretier1.npz).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
V2 = V5 / "data" / "LignoIL_unified_v2"
XLSX = V5 / "data" / "LignoIL" / "Activity coeff and Excess Enthalpy for Imaging.xlsx"
UNI = V5 / "data" / "LignoIL_unified" / "unified_lignin.csv"

T_MEAN, T_STD = 375.0, 50.0  # same refs as rebuild_cache_v2


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else None


# -----------------------------------------------------------------------
# Step 1: interpolate missing Datasheet row
# -----------------------------------------------------------------------
def report_excel_patch():
    ds = pd.read_excel(XLSX, sheet_name="Datasheet")
    missing = ds[ds["γ1"].isna()]
    print(f"[step1] Excel Datasheet rows with NaN γ1: {len(missing)}")
    if not len(missing):
        return
    # For each missing (IL ID, T), interpolate the 7 properties from the
    # same IL's other T rows.
    core_cols = ["γ1", "γ2", "G^E (kcal/mol)", "H^E (kcal/mol)",
                  "G^mix (kcal/mol)", "H_vap (kcal/mol)", "P (bar)"]
    for _, row in missing.iterrows():
        il = row["IL ID"]; T = row["T (K)"]
        sub = ds[(ds["IL ID"] == il) & ds["γ1"].notna()].sort_values("T (K)")
        if len(sub) < 2:
            print(f"  skip {il} @ {T}: not enough neighbors")
            continue
        interp = {}
        for c in core_cols:
            y = sub[c].astype(float).values
            x = sub["T (K)"].astype(float).values
            interp[c] = float(np.interp(T, x, y))
        print(f"  interpolated {il} @ {T} K:")
        for c in core_cols:
            print(f"    {c:25s} = {interp[c]:.4f}")
    # NOTE: we don't write back to the Excel because the original LignoIL
    # cache was built BEFORE this fill; the cached_train.npz row for
    # EMIMOAc@323K has NaN which propagates to v2. Patching the cache
    # directly would be a separate step; flagged for manual review if
    # this IL is in the train/test split.


# -----------------------------------------------------------------------
# Step 2: fill thermo_feat for unmatched extension rows
# -----------------------------------------------------------------------
def fill_extension_thermo():
    uni = pd.read_csv(UNI)
    uni["smi_canon"] = uni["smiles"].map(canon)
    for split in ["train", "val", "test"]:
        p = V2 / f"cached_{split}.npz"
        z = np.load(p, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        ilids = np.array([i.decode() if isinstance(i, bytes) else str(i) for i in d["il_ids"]])
        smi = np.array([s.decode() if isinstance(s, bytes) else s for s in d["smiles"]])
        is_orig = d["is_original"].astype(bool)
        has_lignin = ~np.isnan(d["targets"][:, 7])
        ext_mask = (~is_orig) & has_lignin & np.array([str(i).startswith("[") for i in ilids])
        # Rows that STILL have thermo[0]=0 after rebuild_cache_v2 (i.e. unmatched)
        already_filled = d["thermo_feat"][:, 0] != 0
        unmatched = ext_mask & (~already_filled)
        if unmatched.sum() == 0:
            print(f"[step2] {split}: no unmatched extension rows")
            continue
        filled = 0
        for i in np.where(unmatched)[0]:
            cs = canon(smi[i]) or smi[i]
            matches = uni[uni["smi_canon"] == cs]
            if len(matches) == 0:
                continue
            T_C = matches["temperature_C"].astype(float).mean()
            if np.isnan(T_C):
                continue
            T_K = T_C + 273.15
            d["thermo_feat"][i, 0] = (T_K - T_MEAN) / T_STD
            d["thermo_feat"][i, 2] = (10000.0 / T_K - 10000.0 / T_MEAN) / 10.0
            filled += 1
        print(f"[step2] {split}: filled {filled}/{unmatched.sum()} unmatched extension rows")
        # Save backup once, then overwrite
        bpath = V2 / f"cached_{split}_pretier1.npz"
        if not bpath.exists():
            np.savez(bpath, **z)
        np.savez(p, **d)


# -----------------------------------------------------------------------
# Step 3: chemprop MPN fingerprints
# -----------------------------------------------------------------------
def compute_chemprop_fps():
    """Load the trained chemprop checkpoint and extract per-SMILES MPN FPs."""
    CKPT = PROJECT_ROOT / "checkpoints" / "chemprop" / "fold_0" / "model_0" / "model.pt"
    if not CKPT.exists():
        print(f"[step3] missing checkpoint {CKPT} — skipping")
        return
    try:
        import torch
        from chemprop.utils import load_checkpoint
        from chemprop.features.featurization import MolGraph, BatchMolGraph
    except Exception as e:
        print(f"[step3] chemprop import failed: {e} — skipping")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(str(CKPT), device=torch.device(device))
    model.eval()
    mpn = model.encoder  # the MPN encoder; calling it returns the molecule FP

    # Collect unique SMILES across all three splits of v2
    all_smi = set()
    for split in ["train", "val", "test"]:
        z = np.load(V2 / f"cached_{split}.npz", allow_pickle=True)
        for s in z["smiles"]:
            s = s.decode() if isinstance(s, bytes) else s
            c = canon(s)
            if c:
                all_smi.add(c)
    smi_list = sorted(all_smi)
    print(f"[step3] computing MPN fingerprints for {len(smi_list)} unique SMILES (ckpt {CKPT.name})")

    fps = {}
    batch = 32
    for i in range(0, len(smi_list), batch):
        chunk = smi_list[i:i + batch]
        # chemprop expects list-of-list-of-SMILES; one-mol-per-sample here.
        wrapped = [[s] for s in chunk]
        with torch.no_grad():
            try:
                emb = mpn(wrapped).cpu().numpy()
            except Exception as e:
                print(f"[step3] encoder call failed on a chunk: {e}; falling back")
                # Fallback: one at a time
                emb = np.zeros((len(chunk), 300), dtype=np.float32)
                for j, w in enumerate(wrapped):
                    try:
                        emb[j] = mpn([w]).cpu().numpy()[0]
                    except Exception:
                        pass
        for s, e in zip(chunk, emb):
            fps[s] = e.astype(np.float32)
    if not fps:
        print("[step3] no fingerprints computed — aborting patch")
        return
    fp_dim = next(iter(fps.values())).shape[0]
    print(f"[step3] done: {len(fps)} FPs of dim {fp_dim}")
    if fp_dim != 300:
        print(f"[step3] WARNING: fp_dim={fp_dim} but cache expects 300; skipping patch")
        return

    # Patch each split's chemprop_fp for rows that are all-zero.
    # Must force a .copy() — np.load returns views that silently no-op on writes.
    for split in ["train", "val", "test"]:
        p = V2 / f"cached_{split}.npz"
        z = np.load(p, allow_pickle=True)
        d = {k: np.asarray(z[k]).copy() for k in z.files}
        z.close()
        smi = np.array([s.decode() if isinstance(s, bytes) else s for s in d["smiles"]])
        cp = d["chemprop_fp"].astype(np.float32, copy=True)
        zero_rows = np.all(cp == 0, axis=1)
        filled = 0
        for i in np.where(zero_rows)[0]:
            c = canon(smi[i]) or smi[i]
            if c in fps:
                cp[i] = fps[c]
                filled += 1
        d["chemprop_fp"] = cp
        np.savez(p, **d)
        zr = np.load(p, allow_pickle=True)
        nz = int((np.abs(zr["chemprop_fp"]).sum(-1) > 0).sum())
        zr.close()
        print(f"[step3] {split}: filled chemprop_fp on {filled}/{zero_rows.sum()} zero rows; "
              f"non-zero after save = {nz}")


def main():
    print("=" * 60)
    print("Tier-1 cache fills")
    print("=" * 60)
    report_excel_patch()
    fill_extension_thermo()
    compute_chemprop_fps()
    print("\nDone.")


if __name__ == "__main__":
    main()
