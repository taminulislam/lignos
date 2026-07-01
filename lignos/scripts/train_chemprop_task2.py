"""Chemprop D-MPNN literature baseline for Task 2 leave-IL-out CV.

Strong literature DL baseline that trains a stock Chemprop D-MPNN per fold
on lignin yield only, using the cation.anion SMILES already stored in the
cache plus the same 5-dim process features (T, time, IL_conc, biomass)
consumed by ProcSpec and the two-stage backbone. No COSMO-SAC, no DFT
surfaces — just molecular graph + process conditions, which is the fair
literature comparator for graph-based regressors on this benchmark.

Reproduces the same fold construction as `compare_a59_baran_feat_meta.py`.
Trains `--n-seeds` independent Chemprop models per fold and averages their
predictions. Appends `pred_chemprop_lig` to each per-fold row CSV in place.
Also writes `results/lignos_chemprop_fold_{k}.json` with the per-fold R².

Reference: Yang et al., J. Chem. Inf. Model. 2019 (Chemprop D-MPNN).
"""
from __future__ import annotations
import argparse, csv, json, shutil, subprocess, sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from compare_a2_vs_baran import _load_baran_matched  # noqa

RESULTS = V5 / "results"
SCRATCH = V5 / "scratch" / "chemprop"
IDX_LIGNIN = 7
N_PROCESS = 5  # thermo_feat[:, 0:5] = T, time, IL_conc, biomass C/H


def _write_chemprop_csv(path: Path, smiles: np.ndarray, y: np.ndarray) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "y_lig"])
        for s, yi in zip(smiles, y):
            w.writerow([s, "" if np.isnan(yi) else f"{float(yi):.8f}"])


def _run(cmd: list[str]) -> None:
    """Run a subprocess, stream stdout/stderr, raise on non-zero exit."""
    print(f"    $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Subprocess failed with code {r.returncode}: {cmd[0]}")


def train_and_predict_fold(
    fold_k: int,
    smiles_tr: np.ndarray, feat_tr: np.ndarray, y_tr: np.ndarray,
    smiles_te: np.ndarray, feat_te: np.ndarray,
    n_seeds: int, epochs: int, batch_size: int,
    smiles_only: bool, val_frac: float, keep_scratch: bool,
) -> np.ndarray:
    """Train n_seeds Chemprop models on (smiles_tr, feat_tr, y_tr) and return
    averaged predictions on (smiles_te, feat_te). Uses Chemprop 1.7 CLI.
    """
    # Filter training rows with valid lignin labels
    ok = ~np.isnan(y_tr)
    smiles_tr = smiles_tr[ok]
    feat_tr = feat_tr[ok]
    y_tr = y_tr[ok]

    fold_dir = SCRATCH / f"fold{fold_k}"
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    fold_dir.mkdir(parents=True)

    # Train/val split (deterministic on base seed 42)
    rng = np.random.default_rng(42 + fold_k)
    n = len(y_tr)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    train_csv = fold_dir / "train.csv"
    val_csv = fold_dir / "val.csv"
    test_csv = fold_dir / "test.csv"
    _write_chemprop_csv(train_csv, smiles_tr[tr_idx], y_tr[tr_idx])
    _write_chemprop_csv(val_csv, smiles_tr[val_idx], y_tr[val_idx])
    _write_chemprop_csv(test_csv, smiles_te, np.full(len(smiles_te), np.nan))

    if not smiles_only:
        train_npz = fold_dir / "train_feats.npz"
        val_npz = fold_dir / "val_feats.npz"
        test_npz = fold_dir / "test_feats.npz"
        np.savez(train_npz, features=feat_tr[tr_idx].astype(np.float32))
        np.savez(val_npz, features=feat_tr[val_idx].astype(np.float32))
        np.savez(test_npz, features=feat_te.astype(np.float32))

    preds = []
    for s in range(n_seeds):
        seed_dir = fold_dir / f"seed{s}"
        seed_dir.mkdir()
        train_cmd = [
            "chemprop_train",
            "--data_path", str(train_csv),
            "--separate_val_path", str(val_csv),
            "--dataset_type", "regression",
            "--smiles_columns", "smiles",
            "--target_columns", "y_lig",
            "--save_dir", str(seed_dir),
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--seed", str(s),
            "--num_workers", "0",
            "--quiet",
        ]
        if not smiles_only:
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
        if not smiles_only:
            pred_cmd += [
                "--features_path", str(test_npz),
                "--no_features_scaling",
            ]
        _run(pred_cmd)

        with open(preds_csv) as fh:
            rows = list(csv.DictReader(fh))
        p = np.array([float(r["y_lig"]) for r in rows], dtype=np.float32)
        if len(p) != len(smiles_te):
            raise RuntimeError(
                f"Fold {fold_k} seed {s}: pred len {len(p)} != test len {len(smiles_te)}"
            )
        preds.append(p)
        print(f"    seed {s}: {len(p)} preds, mean={p.mean():+.3f}", flush=True)

    if not keep_scratch:
        shutil.rmtree(fold_dir)

    return np.stack(preds).mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None,
                    help="If given, run only this fold (for SLURM arrays).")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--smiles-only", action="store_true",
                    help="Drop process features — pure literature D-MPNN.")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="Keep per-fold scratch dir for debugging.")
    args = ap.parse_args()

    tr, va, te, lig_ils, baran_lig_mask = _load_baran_matched()
    np.random.seed(42)
    il_order = np.random.permutation(lig_ils)
    fold_size = max(1, len(il_order) // args.n_splits)
    folds = [il_order[i * fold_size : (i + 1) * fold_size if i < args.n_splits - 1 else None]
             for i in range(args.n_splits)]

    pool_il = np.concatenate([tr["il_ids"], va["il_ids"], te["il_ids"]])
    pool_y = np.concatenate([tr["targets"], va["targets"], te["targets"]]).astype(np.float32)
    pool_smi = np.concatenate([tr["smiles"], va["smiles"], te["smiles"]])
    pool_th = np.concatenate([tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]]).astype(np.float32)
    pool_feat = pool_th[:, :N_PROCESS]

    print(f"Pool: {len(pool_il)} rows, {len(lig_ils)} Baran-matched ILs")
    print(f"Features: {'SMILES-only' if args.smiles_only else f'SMILES + {N_PROCESS} process cols'}")
    print(f"Seeds: {args.n_seeds}  epochs: {args.epochs}  batch: {args.batch_size}")

    target_folds = [args.fold] if args.fold is not None else list(range(args.n_splits))
    for k in target_folds:
        held = folds[k]
        te_mask = np.isin(pool_il, held) & baran_lig_mask
        tr_mask = ~np.isin(pool_il, held)
        if te_mask.sum() == 0:
            print(f"Fold {k}: 0 test rows — skip")
            continue
        print(f"\n=== Fold {k}: held ILs = {list(held)} ===")
        print(f"    n_train={int(tr_mask.sum())}, n_test={int(te_mask.sum())}")

        pred = train_and_predict_fold(
            k,
            pool_smi[tr_mask], pool_feat[tr_mask], pool_y[tr_mask, IDX_LIGNIN],
            pool_smi[te_mask], pool_feat[te_mask],
            n_seeds=args.n_seeds, epochs=args.epochs, batch_size=args.batch_size,
            smiles_only=args.smiles_only, val_frac=args.val_frac,
            keep_scratch=args.keep_scratch,
        )

        # Mutate per-fold row CSV in place
        csv_path = RESULTS / f"lignos_baran_feat_meta_fold_{k}_rows.csv"
        if not csv_path.exists():
            print(f"Fold {k}: CSV not found at {csv_path} — skip CSV update")
            y_te = pool_y[te_mask, IDX_LIGNIN]
        else:
            with open(csv_path) as fh:
                rows = list(csv.DictReader(fh))
            if len(rows) != len(pred):
                raise RuntimeError(
                    f"Fold {k}: row count mismatch (csv={len(rows)} vs pred={len(pred)})"
                )
            for i, row in enumerate(rows):
                row["pred_chemprop_lig"] = float(pred[i])
            fieldnames = list(rows[0].keys())
            if "pred_chemprop_lig" not in fieldnames:
                fieldnames.append("pred_chemprop_lig")
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            y_te = np.array([float(r["y_true"]) for r in rows], dtype=np.float32)

        from sklearn.metrics import r2_score, mean_squared_error
        r2 = float(r2_score(y_te, pred))
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        print(f"Fold {k}: Chemprop R² = {r2:+.4f}  RMSE = {rmse:.4f}  "
              f"(n_test={len(pred)}, n_seeds={args.n_seeds})")

        summary = {
            "fold": k,
            "n_train": int(tr_mask.sum()),
            "n_test": int(te_mask.sum()),
            "n_seeds": args.n_seeds,
            "epochs": args.epochs,
            "smiles_only": bool(args.smiles_only),
            "r2": r2,
            "rmse": rmse,
            "held_ils": [str(x) for x in held],
        }
        out_json = RESULTS / f"lignos_chemprop_fold_{k}.json"
        with open(out_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"    wrote {out_json.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
