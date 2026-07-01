"""Chemprop D-MPNN fine-tuned from an ILThermo-pretrained checkpoint.

Fair D-MPNN comparator to LIGNOS: the pretrained checkpoint under
`checkpoints/chemprop_unified_v3/` was trained on the same 60k-row
ILThermo multi-task dataset LIGNOS's A2 backbone consumes, so this
baseline has access to the same thermodynamic prior. Only the lignin
head is newly trained.

Chemprop 1.x `--checkpoint_path` semantics: MPNN encoder weights load
from the pretrained 7-target checkpoint; the FFN output layer is
reinitialized because the num_tasks changes from 7 to 1.

Supports both Task 1 (single test split, 10 seeds) and Task 2 (5-fold
leave-IL-out, 5 seeds). Mutates per-fold CSVs with `pred_chempretrained_lig`.
"""
from __future__ import annotations
import argparse, csv, json, shutil, subprocess, sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from compare_a2_vs_baran import _load_baran_matched  # noqa

CACHE = V5 / "data" / "LignoIL"
RESULTS = V5 / "results"
SCRATCH = V5 / "scratch" / "chemprop_pretrained"
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "chemprop_unified_v3" / "fold_0" / "model_0" / "model.pt"
IDX_LIGNIN = 7
N_PROCESS = 25  # full thermo_feat to match pretrained checkpoint's 325-D readout


def _write_chemprop_csv(path: Path, smiles: np.ndarray, y: np.ndarray) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "y_lig"])
        for s, yi in zip(smiles, y):
            w.writerow([s, "" if np.isnan(yi) else f"{float(yi):.8f}"])


def _run(cmd: list[str]) -> None:
    print(f"    $ {' '.join(cmd[:4])} ... ({len(cmd)} tokens)", flush=True)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Subprocess failed with code {r.returncode}: {cmd[0]}")


def train_and_predict(
    tag: str, ckpt_path: Path,
    smi_tr: np.ndarray, feat_tr: np.ndarray, y_tr: np.ndarray,
    smi_va: np.ndarray, feat_va: np.ndarray, y_va: np.ndarray,
    smi_te: np.ndarray, feat_te: np.ndarray,
    n_seeds: int, epochs: int, batch_size: int,
    smiles_only: bool, keep_scratch: bool,
    init_lr: float = 1e-5, max_lr: float = 1e-4, final_lr: float = 1e-5,
) -> np.ndarray:
    work = SCRATCH / tag
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    train_csv = work / "train.csv"
    val_csv = work / "val.csv"
    test_csv = work / "test.csv"
    _write_chemprop_csv(train_csv, smi_tr, y_tr)
    _write_chemprop_csv(val_csv, smi_va, y_va)
    _write_chemprop_csv(test_csv, smi_te, np.full(len(smi_te), np.nan))

    if not smiles_only:
        train_npz = work / "train_feats.npz"
        val_npz = work / "val_feats.npz"
        test_npz = work / "test_feats.npz"
        np.savez(train_npz, features=feat_tr.astype(np.float32))
        np.savez(val_npz, features=feat_va.astype(np.float32))
        np.savez(test_npz, features=feat_te.astype(np.float32))

    preds = []
    for s in range(n_seeds):
        seed_dir = work / f"seed{s}"
        seed_dir.mkdir()
        train_cmd = [
            "chemprop_train",
            "--data_path", str(train_csv),
            "--separate_val_path", str(val_csv),
            "--dataset_type", "regression",
            "--smiles_columns", "smiles",
            "--target_columns", "y_lig",
            "--checkpoint_path", str(ckpt_path),
            "--save_dir", str(seed_dir),
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--seed", str(s),
            "--num_workers", "0",
            "--init_lr", str(init_lr),
            "--max_lr", str(max_lr),
            "--final_lr", str(final_lr),
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
        preds.append(p)
        print(f"      seed {s}: mean={p.mean():+.3f}", flush=True)

    if not keep_scratch:
        shutil.rmtree(work)
    return np.stack(preds).mean(axis=0), preds


def task1(args):
    from sklearn.metrics import r2_score, mean_absolute_error
    tr = {k: v for k, v in np.load(CACHE / "cached_train.npz", allow_pickle=True).items()}
    va = {k: v for k, v in np.load(CACHE / "cached_val.npz", allow_pickle=True).items()}
    te = {k: v for k, v in np.load(CACHE / "cached_test.npz", allow_pickle=True).items()}

    pool_smi = np.concatenate([tr["smiles"], va["smiles"]])
    pool_y = np.concatenate([tr["targets"], va["targets"]])[:, IDX_LIGNIN].astype(np.float32)
    pool_f = np.concatenate([tr["thermo_feat"], va["thermo_feat"]])[:, :N_PROCESS].astype(np.float32)
    ok = ~np.isnan(pool_y)
    pool_smi = pool_smi[ok]; pool_y = pool_y[ok]; pool_f = pool_f[ok]

    smi_te = te["smiles"]
    y_te = te["targets"][:, IDX_LIGNIN].astype(np.float32)
    feat_te = te["thermo_feat"][:, :N_PROCESS].astype(np.float32)
    ok_te = ~np.isnan(y_te)
    smi_te = smi_te[ok_te]; y_te = y_te[ok_te]; feat_te = feat_te[ok_te]

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(pool_y))
    n_val = max(1, int(round(len(pool_y) * 0.15)))
    va_idx = perm[:n_val]; tr_idx = perm[n_val:]

    print(f"Task 1 Chemprop-pretrained  n_pool={len(pool_y)}  n_test={len(y_te)}  "
          f"ckpt={args.ckpt}")

    pred_avg, seed_preds = train_and_predict(
        "task1", Path(args.ckpt),
        pool_smi[tr_idx], pool_f[tr_idx], pool_y[tr_idx],
        pool_smi[va_idx], pool_f[va_idx], pool_y[va_idx],
        smi_te, feat_te,
        n_seeds=args.n_seeds, epochs=args.epochs, batch_size=args.batch_size,
        smiles_only=args.smiles_only, keep_scratch=args.keep_scratch,
        init_lr=args.init_lr, max_lr=args.max_lr, final_lr=args.final_lr,
    )
    seed_r2 = [float(r2_score(y_te, p)) for p in seed_preds]
    r2_avg = float(r2_score(y_te, pred_avg))
    mae_avg = float(mean_absolute_error(y_te, pred_avg))
    for s, r in enumerate(seed_r2):
        print(f"  seed {s}: R² = {r:+.4f}")
    print(f"\nTask 1 Chemprop-pretrained  per-seed R² = {np.mean(seed_r2):+.4f} ± {np.std(seed_r2):.4f}")
    print(f"                            R² on avg preds = {r2_avg:+.4f}   MAE = {mae_avg:.4f}")

    out = {
        "task": "task1", "method": "chemprop_pretrained_ft",
        "ckpt": str(args.ckpt),
        "n_train": int(len(pool_y)), "n_test": int(len(y_te)),
        "n_seeds": args.n_seeds, "epochs": args.epochs,
        "smiles_only": bool(args.smiles_only),
        "per_seed_r2": seed_r2,
        "r2_per_seed_mean": float(np.mean(seed_r2)),
        "r2_per_seed_std": float(np.std(seed_r2)),
        "r2_on_avg_preds": r2_avg, "mae_on_avg_preds": mae_avg,
    }
    with open(RESULTS / "lignos_chemprop_pretrained_task1.json", "w") as fh:
        json.dump(out, fh, indent=2)


def task2(args):
    from sklearn.metrics import r2_score
    tr, va, te, lig_ils, baran_lig_mask = _load_baran_matched()
    np.random.seed(42)
    il_order = np.random.permutation(lig_ils)
    fold_size = max(1, len(il_order) // args.n_splits)
    folds = [il_order[i * fold_size : (i + 1) * fold_size if i < args.n_splits - 1 else None]
             for i in range(args.n_splits)]

    pool_il = np.concatenate([tr["il_ids"], va["il_ids"], te["il_ids"]])
    pool_y = np.concatenate([tr["targets"], va["targets"], te["targets"]]).astype(np.float32)
    pool_smi = np.concatenate([tr["smiles"], va["smiles"], te["smiles"]])
    pool_f = np.concatenate([tr["thermo_feat"], va["thermo_feat"], te["thermo_feat"]])[:, :N_PROCESS].astype(np.float32)

    target_folds = [args.fold] if args.fold is not None else list(range(args.n_splits))
    for k in target_folds:
        held = folds[k]
        te_mask = np.isin(pool_il, held) & baran_lig_mask
        tr_mask = (~np.isin(pool_il, held)) & (~np.isnan(pool_y[:, IDX_LIGNIN]))
        if te_mask.sum() == 0:
            print(f"Fold {k}: 0 test rows — skip"); continue
        smi_full = pool_smi[tr_mask]; y_full = pool_y[tr_mask, IDX_LIGNIN]; f_full = pool_f[tr_mask]
        rng = np.random.default_rng(42 + k)
        perm = rng.permutation(len(y_full))
        n_val = max(1, int(round(len(y_full) * 0.15)))
        va_idx = perm[:n_val]; tr_idx = perm[n_val:]

        print(f"\n=== Fold {k}: held ILs = {list(held)} ===  n_train={tr_mask.sum()} n_test={te_mask.sum()}")
        pred_avg, _ = train_and_predict(
            f"fold{k}", Path(args.ckpt),
            smi_full[tr_idx], f_full[tr_idx], y_full[tr_idx],
            smi_full[va_idx], f_full[va_idx], y_full[va_idx],
            pool_smi[te_mask], pool_f[te_mask],
            n_seeds=args.n_seeds, epochs=args.epochs, batch_size=args.batch_size,
            smiles_only=args.smiles_only, keep_scratch=args.keep_scratch,
        )

        csv_path = RESULTS / f"lignos_baran_feat_meta_fold_{k}_rows.csv"
        if not csv_path.exists():
            print(f"Fold {k}: CSV missing"); continue
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != len(pred_avg):
            raise RuntimeError(f"Fold {k}: row count mismatch")
        for i, row in enumerate(rows):
            row["pred_chempretrained_lig"] = float(pred_avg[i])
        fieldnames = list(rows[0].keys())
        if "pred_chempretrained_lig" not in fieldnames:
            fieldnames.append("pred_chempretrained_lig")
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        y_true_arr = np.array([float(r["y_true"]) for r in rows], dtype=np.float32)
        r2 = float(r2_score(y_true_arr, pred_avg))
        print(f"Fold {k}: Chemprop-pretrained R² = {r2:+.4f}  (n_test={len(rows)}, n_seeds={args.n_seeds})")
        summary = {
            "fold": k, "method": "chemprop_pretrained_ft", "ckpt": str(args.ckpt),
            "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
            "n_seeds": args.n_seeds, "epochs": args.epochs, "r2": r2,
            "held_ils": [str(x) for x in held],
        }
        with open(RESULTS / f"lignos_chempretrained_fold_{k}.json", "w") as fh:
            json.dump(summary, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["task1", "task2"], required=True)
    ap.add_argument("--fold", type=int, default=None, help="Task 2 only; SLURM array.")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT),
                    help="Path to pretrained Chemprop model.pt.")
    ap.add_argument("--init-lr", type=float, default=1e-5,
                    help="Chemprop warmup LR; default 1e-5 for fine-tuning.")
    ap.add_argument("--max-lr", type=float, default=1e-4,
                    help="Chemprop peak LR; default 1e-4 for fine-tuning "
                         "(vs Chemprop default 1e-3, which overwrites pretrained).")
    ap.add_argument("--final-lr", type=float, default=1e-5,
                    help="Chemprop cooldown LR.")
    ap.add_argument("--smiles-only", action="store_true")
    ap.add_argument("--keep-scratch", action="store_true")
    args = ap.parse_args()

    if args.task == "task1":
        task1(args)
    else:
        task2(args)


if __name__ == "__main__":
    main()
