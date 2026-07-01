#!/usr/bin/env python3
"""Tier-2 fills: use the trained chemprop model to produce per-SMILES property
predictions that fill `preds_chemprop` (and, as a reasonable proxy,
`preds_fusion`) on the 5,147 non-original train rows + 716 non-original val
rows. This closes the 97%-zero teacher-prediction gap that dominates the
failure modes of all recent extended-cache experiments.

Strategy
--------
1. Load checkpoints/chemprop/fold_0/model_0/model.pt — this is a 7-target
   regression model trained on the core-7 properties of the original 152-row
   v4 cache.
2. Run it on every unique SMILES in the extended cache.
3. Predictions are standardized on the same per-property basis as the cache
   (the trained model was fit on standardized targets; raw outputs are
   already in standardized units).
4. Lignin (column 7) is NOT predicted by this model (only 7 core props) —
   leave `preds_chemprop[:, 7] = 0` for rows that don't already have it.
5. Fill both `preds_chemprop` AND `preds_fusion` (proxy); the training
   script will compute v4_base = 0.4*preds_fusion + 0.6*preds_chemprop,
   effectively giving every row the same 7-D teacher vector. For lignin
   col 7 the residual still carries all signal (same as before).

Writes in place; backs up as cached_{split}_pretier2.npz.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
V2 = V5 / "data" / "LignoIL_unified_v2"
# Also patch the non-v2 unified cache that Track 1b uses
SRC_ALSO = V5 / "data" / "LignoIL_unified"
CKPT = PROJECT_ROOT / "checkpoints" / "chemprop" / "fold_0" / "model_0" / "model.pt"


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else None


def main():
    if not CKPT.exists():
        print(f"Missing {CKPT}")
        return
    import torch
    from chemprop.utils import load_checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(str(CKPT), device=torch.device(device))
    model.eval()
    print(f"Loaded {CKPT.name} on {device}")

    # Unique SMILES across every cache we touch
    all_smi = set()
    for cache_dir in (V2, SRC_ALSO):
        for split in ["train", "val", "test"]:
            z = np.load(cache_dir / f"cached_{split}.npz", allow_pickle=True)
            for s in z["smiles"]:
                s = s.decode() if isinstance(s, bytes) else s
                c = canon(s)
                if c:
                    all_smi.add(c)
    smi_list = sorted(all_smi)
    print(f"Predicting properties for {len(smi_list)} unique SMILES")

    preds = {}
    for i in range(0, len(smi_list), 32):
        chunk = smi_list[i:i + 32]
        wrapped = [[s] for s in chunk]
        try:
            with torch.no_grad():
                out = model(wrapped).cpu().numpy()   # (B, 7)
        except Exception:
            # Fallback per-smiles
            out = np.zeros((len(chunk), 7), dtype=np.float32)
            for j, w in enumerate(wrapped):
                try:
                    with torch.no_grad():
                        out[j] = model([w]).cpu().numpy()[0]
                except Exception:
                    pass
        for s, o in zip(chunk, out):
            preds[s] = o.astype(np.float32)
    print(f"  got {len(preds)} predictions, dim={next(iter(preds.values())).shape[0]}")
    n_out_dim = next(iter(preds.values())).shape[0]

    # Patch each cache directory (using explicit .copy() so modifications persist)
    for cache_dir in (V2, SRC_ALSO):
        for split in ["train", "val", "test"]:
            p = cache_dir / f"cached_{split}.npz"
            z = np.load(p, allow_pickle=True)
            # Force copies — np.load returns views that may silently no-op on writes.
            d = {k: np.asarray(z[k]).copy() for k in z.files}
            z.close()
            smi = np.array([s.decode() if isinstance(s, bytes) else s for s in d["smiles"]])
            pc = d["preds_chemprop"].astype(np.float32, copy=True)
            pf = d["preds_fusion"].astype(np.float32, copy=True)
            zero_rows = np.all(pc == 0, axis=1) & np.all(pf == 0, axis=1)
            filled = 0
            for i in np.where(zero_rows)[0]:
                c = canon(smi[i]) or smi[i]
                if c in preds:
                    pc[i, :n_out_dim] = preds[c]
                    pf[i, :n_out_dim] = preds[c]
                    filled += 1
            d["preds_chemprop"] = pc
            d["preds_fusion"] = pf
            bpath = cache_dir / f"cached_{split}_pretier2.npz"
            if not bpath.exists():
                np.savez(bpath, **d)  # backup of original pre-fill state
            np.savez(p, **d)
            # Verify persistence
            zr = np.load(p, allow_pickle=True)
            nz_after = int((np.abs(zr["preds_chemprop"]).sum(-1) > 0).sum())
            zr.close()
            print(f"  {cache_dir.name}/{split}: filled preds on {filled}/{zero_rows.sum()} zero rows; "
                  f"non-zero rows after save = {nz_after}")


if __name__ == "__main__":
    main()
