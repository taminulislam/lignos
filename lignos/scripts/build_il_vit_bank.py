"""W2 — Build an IL-level ViT feature bank.

For every compound_id in {geometry_status.csv, tier3_compounds.csv} that has a
rendered `{compound_id}_frames/frame_*.png` directory under data/pipeline/
cosmo_images/, run the existing MultiViewViT encoder over 36 rotation frames
and save a single 192-D embedding. Output is a lookup {canon_SMILES: 192D}.

A5 (train_a5_surface_frames.py) loads this bank and uses the resulting
`vit_feat` as the input to a zero-init gated residual branch. Bank rows whose
SMILES doesn't exist get `has_frames=0` and the branch contributes nothing.

Output: lignos/data/il_vit_bank.npz
  smiles    : (N,) object      canonical SMILES
  vit_feat  : (N, 192) float32 per-IL pooled ViT embedding
  compound_ids : (N,) object   source compound_id for traceability
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
FRAMES_ROOT = PROJECT_ROOT / "data" / "pipeline" / "cosmo_images"
CKPT = V5 / "checkpoints" / "vjepa" / "vit_pretrained_vjepa.pt"
OUT = V5 / "data" / "il_vit_bank.npz"

sys.path.insert(0, str(V5))
from models.multiview_vit import MultiViewViT  # noqa: E402

N_VIEWS = 36
IMG_SIZE = 224


def canon(smi: Optional[str]):
    if not isinstance(smi, str):
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def load_vit(device):
    vit = MultiViewViT(n_views=N_VIEWS, embed_dim=192, dropout=0.0)
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    encoder_state = state.get("encoder_state_dict", state)
    missing, _ = vit.load_state_dict(encoder_state, strict=False)
    print(f"ViT loaded from {CKPT.name} ({len(missing)} missing keys)")
    return vit.to(device).eval()


def load_frames(frames_dir: Path) -> Optional[torch.Tensor]:
    """Load 36 frames from {compound_id}_frames/frame_*.png → (36, 3, 224, 224)."""
    pngs = sorted(frames_dir.glob("frame_*.png"))
    if len(pngs) < N_VIEWS:
        return None
    pngs = pngs[:N_VIEWS]
    frames = []
    for p in pngs:
        try:
            img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            frames.append(arr.transpose(2, 0, 1))  # (3, H, W)
        except Exception:
            return None
    return torch.from_numpy(np.stack(frames))  # (36, 3, 224, 224)


def compound_id_map():
    """Merge geometry_status + tier3 compounds → {compound_id: SMILES}."""
    geom = pd.read_csv(PROJECT_ROOT / "data" / "pipeline" / "geometry_status.csv")
    tier3 = pd.read_csv(PROJECT_ROOT / "data" / "pipeline" / "tier3_compounds.csv")
    mapping = {}
    for _, r in geom.iterrows():
        mapping[r["compound_id"]] = r["smiles"]
    for _, r in tier3.iterrows():
        mapping[r["compound_id"]] = r["smiles"]
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1,
                    help="MultiViewViT runs one compound at a time "
                         "(36 frames = one batch). Set >1 only if mem allows.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    vit = load_vit(device)
    id_to_smi = compound_id_map()
    print(f"compound_id → SMILES mappings: {len(id_to_smi)}")

    results = {}  # canon_smi → (vit_feat, compound_id)
    n_ok = n_skip = 0
    for cid, smi in id_to_smi.items():
        cs = canon(smi)
        if not cs:
            continue
        if cs in results:
            # Already covered by another compound_id; skip re-encoding.
            continue
        frames_dir = FRAMES_ROOT / f"{cid}_frames"
        if not frames_dir.exists():
            n_skip += 1
            continue
        frames = load_frames(frames_dir)
        if frames is None:
            n_skip += 1
            continue
        frames = frames.unsqueeze(0).to(device)  # (1, 36, 3, 224, 224)
        with torch.no_grad():
            out = vit(frames)
            # MultiViewViT.forward returns (embedding, view_weights).
            embedding = out[0] if isinstance(out, tuple) else out
            feat = embedding.cpu().numpy()  # (1, 192)
        results[cs] = (feat[0].astype(np.float32), cid)
        n_ok += 1
        if n_ok % 25 == 0:
            print(f"  [{n_ok}] latest: {cid} → {cs[:60]}")

    smis = np.array(list(results.keys()), dtype=object)
    feats = np.stack([results[s][0] for s in smis])
    cids = np.array([results[s][1] for s in smis], dtype=object)
    np.savez(OUT, smiles=smis, vit_feat=feats, compound_ids=cids)
    print(f"\nWrote {len(smis)} ILs × {feats.shape[1]}D → {OUT}")
    print(f"  encoded: {n_ok}   skipped (no frames): {n_skip}")


if __name__ == "__main__":
    main()
