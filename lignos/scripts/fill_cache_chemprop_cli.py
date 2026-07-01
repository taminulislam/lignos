#!/usr/bin/env python3
"""Robust chemprop-CLI-based cache fill.

Replaces the earlier in-process chemprop Python-API attempt (which produced
all-zero outputs due to MPN-input format mismatch). Uses the canonical
`chemprop_predict` and `chemprop_fingerprint` CLI entry points — same
executables that produced the original cache's teacher features.

Outputs:
  * preds_chemprop, preds_fusion (7-D):   from chemprop_predict
  * chemprop_fp (300-D):                  from chemprop_fingerprint

Patches both LignoIL_unified and LignoIL_unified_v2 caches in place.
"""
from __future__ import annotations
import subprocess, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
V2 = V5 / "data" / "LignoIL_unified_v2"
VU = V5 / "data" / "LignoIL_unified"
CKPT = PROJECT_ROOT / "checkpoints" / "chemprop" / "fold_0" / "model_0" / "model.pt"

CP_PREDICT = "/u/kahmed2/miniconda3/envs/mmseg/bin/chemprop_predict"
CP_FINGERPRINT = "/u/kahmed2/miniconda3/envs/mmseg/bin/chemprop_fingerprint"


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else None


def collect_unique_smiles():
    smi = set()
    for d in (V2, VU):
        for split in ["train", "val", "test"]:
            z = np.load(d / f"cached_{split}.npz", allow_pickle=True)
            for s in z["smiles"]:
                c = canon(s.decode() if isinstance(s, bytes) else s)
                if c:
                    smi.add(c)
    return sorted(smi)


def run_cli(tool, smiles_csv, out_csv):
    cmd = [tool, "--test_path", str(smiles_csv),
           "--checkpoint_path", str(CKPT),
           "--preds_path", str(out_csv),
           "--no_features_scaling"]
    print("  $ " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"{tool} failed")
    # Also print last few lines of stdout for sanity
    for line in result.stdout.splitlines()[-5:]:
        print("  " + line)


def patch_cache(cache_dir: Path, smi_to_pred: dict, smi_to_fp: dict,
                  pred_dim: int, fp_dim: int):
    for split in ["train", "val", "test"]:
        p = cache_dir / f"cached_{split}.npz"
        z = np.load(p, allow_pickle=True)
        d = {k: np.asarray(z[k]).copy() for k in z.files}
        z.close()
        smi = np.array([s.decode() if isinstance(s, bytes) else s for s in d["smiles"]])

        pc = d["preds_chemprop"].astype(np.float32, copy=True)
        pf = d["preds_fusion"].astype(np.float32, copy=True)
        cp = d["chemprop_fp"].astype(np.float32, copy=True)

        zero_pred = np.all(pc == 0, axis=1) & np.all(pf == 0, axis=1)
        zero_cp = np.all(cp == 0, axis=1)

        n_pred = n_cp = 0
        for i in np.where(zero_pred)[0]:
            c = canon(smi[i]) or smi[i]
            v = smi_to_pred.get(c)
            if v is not None and not np.all(np.isnan(v)):
                pc[i, :pred_dim] = np.nan_to_num(v, 0.0)
                pf[i, :pred_dim] = np.nan_to_num(v, 0.0)
                n_pred += 1
        for i in np.where(zero_cp)[0]:
            c = canon(smi[i]) or smi[i]
            v = smi_to_fp.get(c)
            if v is not None:
                cp[i] = v
                n_cp += 1

        d["preds_chemprop"] = pc
        d["preds_fusion"] = pf
        d["chemprop_fp"] = cp
        np.savez(p, **d)

        # Verify
        zr = np.load(p, allow_pickle=True)
        nz_pred = int((np.abs(zr["preds_chemprop"]).sum(-1) > 0).sum())
        nz_cp = int((np.abs(zr["chemprop_fp"]).sum(-1) > 0).sum())
        zr.close()
        print(f"  {cache_dir.name}/{split}: filled preds={n_pred}/{zero_pred.sum()}  "
              f"chemprop_fp={n_cp}/{zero_cp.sum()}  |  "
              f"after: preds_nonzero={nz_pred}, fp_nonzero={nz_cp}")


def main():
    if not CKPT.exists():
        print(f"Missing {CKPT}")
        return

    smi_list = collect_unique_smiles()
    print(f"Unique SMILES to predict: {len(smi_list)}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_csv = td / "smiles.csv"
        pd.DataFrame({"smiles": smi_list}).to_csv(in_csv, index=False)

        pred_csv = td / "predictions.csv"
        print("\n[1/2] chemprop_predict ...")
        run_cli(CP_PREDICT, in_csv, pred_csv)
        pd_df = pd.read_csv(pred_csv)
        pd_cols = [c for c in pd_df.columns if c != "smiles"]
        print(f"  got prediction cols: {pd_cols}")
        smi_to_pred = {}
        for _, r in pd_df.iterrows():
            c = canon(r["smiles"])
            if c:
                smi_to_pred[c] = np.array([r[cc] for cc in pd_cols], dtype=np.float32)
        pred_dim = len(pd_cols)
        print(f"  pred_dim={pred_dim}, got {len(smi_to_pred)} mappings")

        fp_csv = td / "fingerprints.csv"
        print("\n[2/2] chemprop_fingerprint ...")
        run_cli(CP_FINGERPRINT, in_csv, fp_csv)
        fp_df = pd.read_csv(fp_csv)
        fp_cols = [c for c in fp_df.columns if c != "smiles"]
        smi_to_fp = {}
        for _, r in fp_df.iterrows():
            c = canon(r["smiles"])
            if c:
                smi_to_fp[c] = np.array([r[cc] for cc in fp_cols], dtype=np.float32)
        fp_dim = len(fp_cols)
        print(f"  fp_dim={fp_dim}, got {len(smi_to_fp)} mappings")

        if fp_dim != 300:
            print(f"  WARNING: fp_dim={fp_dim} != 300 (cache expected 300). Patching anyway with truncation/padding.")
            for k, v in list(smi_to_fp.items()):
                if len(v) < 300:
                    smi_to_fp[k] = np.concatenate([v, np.zeros(300 - len(v), dtype=np.float32)])
                else:
                    smi_to_fp[k] = v[:300]

        print("\nPatching caches...")
        for d in (V2, VU):
            patch_cache(d, smi_to_pred, smi_to_fp, pred_dim, 300)

    print("\nDone.")


if __name__ == "__main__":
    main()
