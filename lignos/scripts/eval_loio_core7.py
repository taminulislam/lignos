"""Leave-one-IL-out CV on all 28 original ILs for core 7 properties.

Uses the expanded LignoIL training data (167 ILs) for each fold,
but evaluates on each held-out original IL. This gives a robust
R² estimate across all 28 ILs with proper confidence intervals.

Best config from experiments: shallow head + narrow thermo + unbalanced loss.

Usage:
    # Worker mode (array job): run one chunk of folds, save per-fold .npz
    python eval_loio_core7.py --chunk-idx 0 --n-chunks 4

    # Aggregator: read all per-fold .npz and emit final JSON + summary
    python eval_loio_core7.py --aggregate

    # Legacy: run all folds + aggregate in a single process
    python eval_loio_core7.py
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, CORE_PROPS, train_one_seed, predict, r2_per_prop

PARTIAL_DIR = V5 / "results" / "loio_partial"
FINAL_JSON = V5 / "results" / "loio_cv_core7.json"


def load_data():
    tr_d = np.load(V5 / "data/LignoIL/cached_train.npz", allow_pickle=True)
    va_d = np.load(V5 / "data/LignoIL/cached_val.npz", allow_pickle=True)
    te_d = np.load(V5 / "data/LignoIL/cached_test.npz", allow_pickle=True)

    def merge(*dicts):
        return {k: np.concatenate([d[k] for d in dicts], axis=0) for k in dicts[0].keys()}

    all_data = merge(
        {k: tr_d[k] for k in tr_d.files},
        {k: va_d[k] for k in va_d.files},
        {k: te_d[k] for k in te_d.files},
    )
    all_smiles = np.array([s.decode() if isinstance(s, bytes) else s for s in all_data["smiles"]])
    is_original = all_data["is_original"].astype(bool)
    targets = all_data["targets"].astype(np.float32)
    morgan_fp = all_data["morgan_fp"]
    thermo = all_data["thermo_feat"]
    preds_f = all_data["preds_fusion"].astype(np.float32)
    preds_c = all_data["preds_chemprop"].astype(np.float32)
    v4_base = (0.4 * preds_f + 0.6 * preds_c).astype(np.float32)
    unique_orig_ils = sorted(set(all_smiles[is_original]))
    return {
        "all_smiles": all_smiles, "is_original": is_original, "targets": targets,
        "morgan_fp": morgan_fp, "thermo": thermo, "v4_base": v4_base,
        "unique_orig_ils": unique_orig_ils,
    }


def run_fold(fold_i, held_out_il, d, device):
    all_smiles = d["all_smiles"]
    is_original = d["is_original"]
    te_mask = (all_smiles == held_out_il) & is_original
    tr_mask = all_smiles != held_out_il

    pca = PCA(40).fit(d["morgan_fp"][tr_mask])
    f_tr = pca.transform(d["morgan_fp"][tr_mask]).astype(np.float32)
    f_te = pca.transform(d["morgan_fp"][te_mask]).astype(np.float32)

    preds_seeds = []
    for seed in range(5):
        model = train_one_seed(
            seed, d["v4_base"][tr_mask], f_tr, d["thermo"][tr_mask],
            d["targets"][tr_mask], device=device,
            balance_props=False, depth="shallow", wide_thermo=False,
        )
        p = predict(model, d["v4_base"][te_mask], f_te, d["thermo"][te_mask], device)
        preds_seeds.append(p)
    pred_avg = np.mean(preds_seeds, axis=0)
    tgt = d["targets"][te_mask]
    fold_r2 = r2_per_prop(pred_avg, tgt)
    return pred_avg, tgt, fold_r2, int(te_mask.sum())


def run_chunk(chunk_idx, n_chunks):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

    d = load_data()
    unique_orig_ils = d["unique_orig_ils"]
    all_fold_indices = np.arange(len(unique_orig_ils))
    my_folds = np.array_split(all_fold_indices, n_chunks)[chunk_idx]

    print(f"Total samples: {len(d['all_smiles'])}")
    print(f"Original samples: {d['is_original'].sum()} ({len(unique_orig_ils)} unique ILs)")
    print(f"Chunk {chunk_idx}/{n_chunks}: folds {list(my_folds)}")
    print(f"{'='*60}")

    for fold_i in my_folds:
        held_out_il = unique_orig_ils[fold_i]
        pred_avg, tgt, fold_r2, n_te = run_fold(fold_i, held_out_il, d, device)

        out_path = PARTIAL_DIR / f"fold_{fold_i:02d}.npz"
        np.savez(
            out_path,
            fold_idx=fold_i,
            il_smiles=held_out_il,
            preds=pred_avg,
            targets=tgt,
            fold_r2_keys=np.array(list(fold_r2.keys())),
            fold_r2_vals=np.array([float(v) for v in fold_r2.values()], dtype=np.float32),
        )
        il_short = held_out_il[:45]
        print(f"  Fold {fold_i+1:>2}/{len(unique_orig_ils)}: {il_short:45s} "
              f"n={n_te:>3}  core7={fold_r2['avg_core7']:.4f}  -> {out_path.name}")


def aggregate():
    d = load_data()
    unique_orig_ils = d["unique_orig_ils"]
    n_folds = len(unique_orig_ils)

    partials = sorted(PARTIAL_DIR.glob("fold_*.npz"))
    if len(partials) < n_folds:
        missing = set(range(n_folds)) - {int(p.stem.split("_")[1]) for p in partials}
        raise RuntimeError(
            f"Incomplete: found {len(partials)}/{n_folds} fold files. Missing: {sorted(missing)}"
        )

    all_preds_list, all_true_list, fold_r2s = [], [], [None] * n_folds
    for p in partials:
        z = np.load(p, allow_pickle=True)
        fold_i = int(z["fold_idx"])
        fold_r2 = dict(zip([k.item() if hasattr(k, "item") else k for k in z["fold_r2_keys"]],
                           [float(v) for v in z["fold_r2_vals"]]))
        fold_r2s[fold_i] = fold_r2
        all_preds_list.append(z["preds"])
        all_true_list.append(z["targets"])

    all_preds_loio = np.concatenate(all_preds_list, axis=0)
    all_true_loio = np.concatenate(all_true_list, axis=0)
    overall = r2_per_prop(all_preds_loio, all_true_loio)

    core7_per_fold = [f["avg_core7"] for f in fold_r2s]
    mean_core7 = float(np.mean(core7_per_fold))
    std_core7 = float(np.std(core7_per_fold))
    median_core7 = float(np.median(core7_per_fold))

    print(f"\n{'='*60}")
    print(f"LOIO-CV RESULTS ({n_folds} folds, 5 seeds each)")
    print(f"{'='*60}")
    print(f"\nOverall (pooled predictions):")
    print(f"  core7 R² = {overall['avg_core7']:.4f}")
    for p in CORE_PROPS:
        print(f"    {p:15s}: {overall.get(p, float('nan')):.4f}")

    print(f"\nPer-fold statistics:")
    print(f"  core7 R²: mean={mean_core7:.4f} ± {std_core7:.4f}  median={median_core7:.4f}")
    print(f"  Per-property mean ± std:")
    for p in CORE_PROPS:
        vals = [f[p] for f in fold_r2s if not np.isnan(f.get(p, float("nan")))]
        if vals:
            print(f"    {p:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    sorted_folds = sorted(zip(core7_per_fold, unique_orig_ils), reverse=True)
    print(f"\n  Best 5 folds:")
    for r2, il in sorted_folds[:5]:
        print(f"    {r2:.4f}  {il[:50]}")
    print(f"  Worst 5 folds:")
    for r2, il in sorted_folds[-5:]:
        print(f"    {r2:.4f}  {il[:50]}")

    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    print(f"  Fixed 5-IL test split:      R² = 0.8430 (best, 10 seeds)")
    print(f"  LOIO-CV pooled ({n_folds} ILs):    R² = {overall['avg_core7']:.4f}")
    print(f"  LOIO-CV per-fold mean:      R² = {mean_core7:.4f} ± {std_core7:.4f}")

    results = {
        "overall_pooled": {p: float(overall.get(p, float("nan"))) for p in CORE_PROPS + ["avg_core7"]},
        "per_fold_mean": mean_core7,
        "per_fold_std": std_core7,
        "per_fold_median": median_core7,
        "per_fold": [{
            "il": il,
            "core7": float(fold_r2s[i]["avg_core7"]),
            **{p: float(fold_r2s[i].get(p, float("nan"))) for p in CORE_PROPS}
        } for i, il in enumerate(unique_orig_ils)],
        "n_folds": n_folds,
        "n_seeds": 5,
        "config": "shallow+narrow_thermo+unbalanced+expanded_LignoIL",
    }
    json.dump(results, open(FINAL_JSON, "w"), indent=2)
    print(f"\nSaved: {FINAL_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-idx", type=int, default=None, help="0-based chunk index for array worker")
    ap.add_argument("--n-chunks", type=int, default=4, help="total number of chunks in the array")
    ap.add_argument("--aggregate", action="store_true", help="aggregate per-fold .npz files into final JSON")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
    elif args.chunk_idx is not None:
        run_chunk(args.chunk_idx, args.n_chunks)
    else:
        # Legacy: all folds in one process, then aggregate
        run_chunk(0, 1)
        aggregate()


if __name__ == "__main__":
    main()
