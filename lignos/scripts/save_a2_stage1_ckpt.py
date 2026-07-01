"""Train A2_chemprop Stage-1 for N seeds, pick the seed with best validation
masked-MSE, save its state_dict. Produces the warm-start checkpoint that
A5_sf and A5_cosmo need in order to train ONLY their new branches on top of
a converged A2 backbone (instead of retraining A2 from scratch with extra
parameters, which destabilized both A5 runs on 2026-04-20).

Output: lignos/checkpoints/a2/stage1_best.pt
  {"state_dict": ..., "seed": <int>, "val_loss": <float>, "core7": <float>}
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import r2_per_prop, PROPS  # noqa
from train_a2_two_stage import (
    A2Head, train_stage1_a2,
    build_chemprop_40d, predict_stage1, v4_base,
)
from sklearn.decomposition import PCA

CACHE = V5 / "data" / "LignoIL_A1"
CKPT_DIR = V5 / "checkpoints" / "a2"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CKPT_DIR / "stage1_best.pt"


def _load_split(s):
    p_dft = CACHE / f"cached_{s}_dft.npz"
    p_std = CACHE / f"cached_{s}.npz"
    p = p_dft if p_dft.exists() else p_std
    return {k: v for k, v in np.load(p, allow_pickle=True).items()}


def _val_loss(m, v4, f, th, cp, y, device):
    with torch.no_grad():
        pred = m(torch.from_numpy(v4).to(device),
                 torch.from_numpy(f).to(device),
                 torch.from_numpy(th).to(device),
                 torch.from_numpy(cp).to(device)).cpu().numpy()
    mask = ~np.isnan(y)
    err = (pred - np.nan_to_num(y)) ** 2 * mask
    return float((err.sum(0) / mask.sum(0).clip(min=1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tr, va, te = _load_split("train"), _load_split("val"), _load_split("test")
    pca = PCA(40).fit(tr["morgan_fp"])
    m_tr = pca.transform(tr["morgan_fp"]).astype(np.float32)
    m_va = pca.transform(va["morgan_fp"]).astype(np.float32)
    m_te = pca.transform(te["morgan_fp"]).astype(np.float32)
    cp_tr, cp_te = build_chemprop_40d(tr["chemprop_fp"], te["chemprop_fp"])
    _, cp_va = build_chemprop_40d(tr["chemprop_fp"], va["chemprop_fp"])
    v4_tr, v4_va, v4_te = v4_base(tr), v4_base(va), v4_base(te)
    th_tr, th_va, th_te = tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]
    y_tr, y_va, y_te = [x["targets"].astype(np.float32) for x in (tr, va, te)]

    best = None
    for seed in range(args.n_seeds):
        print(f"\n[seed {seed}] Training A2 Stage-1...")
        m = train_stage1_a2(seed, v4_tr, m_tr, th_tr, cp_tr, y_tr, device,
                             epochs=args.epochs)
        vl = _val_loss(m, v4_va, m_va, th_va, cp_va, y_va, device)
        te_pred = predict_stage1(m, v4_te, m_te, th_te, cp_te, device)
        r = r2_per_prop(te_pred, y_te)
        print(f"  seed {seed}: val_loss={vl:.6f}  test core7={r['avg_core7']:.4f}")
        if best is None or vl < best["val_loss"]:
            best = {"state_dict": {k: v.detach().cpu() for k, v in m.state_dict().items()},
                    "seed": seed, "val_loss": vl, "core7": r["avg_core7"],
                    "per_prop": {p: r.get(p) for p in PROPS}}

    # Save
    torch.save(best, CKPT_PATH)
    print(f"\nSaved best (seed={best['seed']}, val_loss={best['val_loss']:.6f}, "
          f"core7={best['core7']:.4f}) → {CKPT_PATH}")


if __name__ == "__main__":
    main()
