"""Chemprop D-MPNN literature baseline for Task 1 (same-chemistry split).

Task 1 uses the fixed LignoIL_A1 test split: train on cached_train +
cached_val (lignin-labeled rows), evaluate on cached_test (39 rows, all
lignin-labeled). This is the protocol used by the A2 two-stage and
LIGNOS +#5+#6 headline numbers (0.706 / 0.750 ± 0.037 lignin R²).

Same feature recipe as `train_chemprop_task2.py`: cation.anion SMILES +
5-dim process conditions (T, time, IL_conc, biomass) via --features_path.
Per-seed R² is logged for std; the final prediction is the seed-averaged
mean. Defaults to 10 seeds to match LIGNOS +#5+#6's reporting protocol.

Writes `results/lignos_chemprop_task1.json` with per-seed R² and the
aggregate mean ± std.
"""
from __future__ import annotations
import argparse, csv, json, shutil, subprocess, sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
CACHE = V5 / "data" / "LignoIL"
RESULTS = V5 / "results"
SCRATCH = V5 / "scratch" / "chemprop_task1"
IDX_LIGNIN = 7
N_PROCESS = 5


def _write_chemprop_csv(path: Path, smiles: np.ndarray, y: np.ndarray) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "y_lig"])
        for s, yi in zip(smiles, y):
            w.writerow([s, "" if np.isnan(yi) else f"{float(yi):.8f}"])


def _run(cmd: list[str]) -> None:
    print(f"    $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Subprocess failed with code {r.returncode}: {cmd[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Fraction of train pool to hold out for Chemprop's "
                         "internal val (early stopping). cached_val is merged "
                         "into the train pool first.")
    ap.add_argument("--smiles-only", action="store_true",
                    help="Drop process features — pure literature D-MPNN.")
    ap.add_argument("--keep-scratch", action="store_true")
    args = ap.parse_args()

    tr = {k: v for k, v in np.load(CACHE / "cached_train.npz", allow_pickle=True).items()}
    va = {k: v for k, v in np.load(CACHE / "cached_val.npz", allow_pickle=True).items()}
    te = {k: v for k, v in np.load(CACHE / "cached_test.npz", allow_pickle=True).items()}

    pool_smi = np.concatenate([tr["smiles"], va["smiles"]])
    pool_y = np.concatenate([tr["targets"], va["targets"]])[:, IDX_LIGNIN].astype(np.float32)
    pool_feat = np.concatenate([tr["thermo_feat"], va["thermo_feat"]])[:, :N_PROCESS].astype(np.float32)

    ok_tr = ~np.isnan(pool_y)
    smi_tr = pool_smi[ok_tr]
    y_tr = pool_y[ok_tr]
    feat_tr = pool_feat[ok_tr]

    smi_te = te["smiles"]
    y_te = te["targets"][:, IDX_LIGNIN].astype(np.float32)
    feat_te = te["thermo_feat"][:, :N_PROCESS].astype(np.float32)
    ok_te = ~np.isnan(y_te)
    smi_te = smi_te[ok_te]
    y_te = y_te[ok_te]
    feat_te = feat_te[ok_te]

    print(f"Task 1: n_train_pool={len(y_tr)}  n_test={len(y_te)}")
    print(f"Features: {'SMILES-only' if args.smiles_only else f'SMILES + {N_PROCESS} process cols'}")
    print(f"Seeds: {args.n_seeds}  epochs: {args.epochs}  batch: {args.batch_size}")

    root = SCRATCH
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    # Deterministic train/val split used for Chemprop's internal early-stopping val
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y_tr))
    n_val = max(1, int(round(len(y_tr) * args.val_frac)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    train_csv = root / "train.csv"
    val_csv = root / "val.csv"
    test_csv = root / "test.csv"
    _write_chemprop_csv(train_csv, smi_tr[tr_idx], y_tr[tr_idx])
    _write_chemprop_csv(val_csv, smi_tr[val_idx], y_tr[val_idx])
    _write_chemprop_csv(test_csv, smi_te, np.full(len(smi_te), np.nan))

    if not args.smiles_only:
        train_npz = root / "train_feats.npz"
        val_npz = root / "val_feats.npz"
        test_npz = root / "test_feats.npz"
        np.savez(train_npz, features=feat_tr[tr_idx].astype(np.float32))
        np.savez(val_npz, features=feat_tr[val_idx].astype(np.float32))
        np.savez(test_npz, features=feat_te.astype(np.float32))

    from sklearn.metrics import r2_score, mean_absolute_error

    seed_preds = []
    seed_r2 = []
    for s in range(args.n_seeds):
        seed_dir = root / f"seed{s}"
        seed_dir.mkdir()
        train_cmd = [
            "chemprop_train",
            "--data_path", str(train_csv),
            "--separate_val_path", str(val_csv),
            "--dataset_type", "regression",
            "--smiles_columns", "smiles",
            "--target_columns", "y_lig",
            "--save_dir", str(seed_dir),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--seed", str(s),
            "--num_workers", "0",
            "--quiet",
        ]
        if not args.smiles_only:
            train_cmd += [
                "--features_path", str(train_npz),
                "--separate_val_features_path", str(val_npz),
                "--no_features_scaling",
            ]
        _run(train_cmd)

        preds_csv = seed_dir / "preds.csv"
        pred_cmd = [
            "chemprop_predict",
            "--test_path", str(test_csv),
            "--smiles_columns", "smiles",
            "--checkpoint_dir", str(seed_dir),
            "--preds_path", str(preds_csv),
            "--num_workers", "0",
        ]
        if not args.smiles_only:
            pred_cmd += [
                "--features_path", str(test_npz),
                "--no_features_scaling",
            ]
        _run(pred_cmd)

        with open(preds_csv) as fh:
            rows = list(csv.DictReader(fh))
        p = np.array([float(r["y_lig"]) for r in rows], dtype=np.float32)
        r2_s = float(r2_score(y_te, p))
        seed_preds.append(p)
        seed_r2.append(r2_s)
        print(f"    seed {s}: R² = {r2_s:+.4f}  mean={p.mean():+.3f}", flush=True)

    pred_avg = np.stack(seed_preds).mean(axis=0)
    r2_mean_of_preds = float(r2_score(y_te, pred_avg))
    mae_of_avg = float(mean_absolute_error(y_te, pred_avg))
    r2_mu = float(np.mean(seed_r2))
    r2_sd = float(np.std(seed_r2))

    print()
    print(f"Task 1 Chemprop D-MPNN  —  n_seeds={args.n_seeds}")
    print(f"  per-seed R² mean ± std : {r2_mu:+.4f} ± {r2_sd:.4f}")
    print(f"  R² on seed-averaged preds: {r2_mean_of_preds:+.4f}")
    print(f"  MAE on seed-averaged preds: {mae_of_avg:.4f}")

    out = {
        "task": "task1",
        "n_train_pool": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_seeds": args.n_seeds,
        "epochs": args.epochs,
        "smiles_only": bool(args.smiles_only),
        "per_seed_r2": seed_r2,
        "r2_per_seed_mean": r2_mu,
        "r2_per_seed_std": r2_sd,
        "r2_on_avg_preds": r2_mean_of_preds,
        "mae_on_avg_preds": mae_of_avg,
    }
    out_json = RESULTS / "lignos_chemprop_task1.json"
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_json.relative_to(PROJECT_ROOT)}")

    if not args.keep_scratch:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
