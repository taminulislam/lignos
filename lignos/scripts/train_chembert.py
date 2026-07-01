"""ChemBERTa-77M-MLM fine-tuned on lignin yield — transformer baseline.

Architecture: pretrained DeepChem/ChemBERTa-77M-MLM (RoBERTa-style SMILES
encoder, 384-D hidden, ~3.4M params) + mean-pool over non-pad tokens +
concat 5-D process features + 2-layer MLP regression head (384+5 -> 64 -> 1).

Fair apples-to-apples with the Chemprop baselines: same SMILES
(cation.anion) and same 5-dim process conditions (`thermo_feat[:, :5]`)
as Task 1/2 Chemprop runs. Internal 15% val for early stopping.

Two protocols:
- task1: cached_train+val pool, predict cached_test (39 rows).
- task2: 5-fold leave-IL-out on Baran-matched 13-IL subset; mutates each
  per-fold row CSV with `pred_chembert_lig`.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from compare_a2_vs_baran import _load_baran_matched  # noqa

CACHE = V5 / "data" / "LignoIL"
RESULTS = V5 / "results"
IDX_LIGNIN = 7
N_PROCESS = 5
MODEL_ID = "DeepChem/ChemBERTa-77M-MLM"
MAX_LEN = 128


class SmilesDS(Dataset):
    def __init__(self, smiles, feats, y, tok):
        self.smiles = list(smiles)
        self.feats = torch.as_tensor(feats, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.tok = tok

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        enc = self.tok(self.smiles[i], truncation=True, max_length=MAX_LEN,
                        padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "feat": self.feats[i],
            "y": self.y[i],
        }


class ChemBERTaReg(nn.Module):
    def __init__(self, backbone, feat_dim):
        super().__init__()
        self.backbone = backbone
        H = backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(H + feat_dim, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask, feat):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # (B, L, H)
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        x = torch.cat([pooled, feat], dim=-1)
        return self.head(x).squeeze(-1)


def train_one(smi_tr, feat_tr, y_tr, smi_va, feat_va, y_va,
              smi_te, feat_te, seed, epochs, batch_size, lr, device,
              frozen_warmup_epochs=5, ft_encoder_lr=5e-5):
    """Two-phase transfer-learning recipe for small-data pretrained ChemBERTa.

    Phase 1 (frozen warmup, `frozen_warmup_epochs` epochs): freeze the
    RoBERTa encoder, train the regression head at `lr`. This anchors the
    head to the pretrained representation without destroying it.
    Phase 2 (full fine-tune, remaining epochs): unfreeze encoder, train
    encoder at `ft_encoder_lr` (≈5e-5) and head at `lr`, no warmup.
    """
    from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    backbone = AutoModel.from_pretrained(MODEL_ID)
    model = ChemBERTaReg(backbone, N_PROCESS).to(device)

    ds_tr = SmilesDS(smi_tr, feat_tr, y_tr, tok)
    ds_va = SmilesDS(smi_va, feat_va, y_va, tok)
    ds_te = SmilesDS(smi_te, feat_te, np.zeros(len(smi_te)), tok)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0)
    loss_fn = nn.MSELoss()

    def run_epochs(n_ep, opt, sch, best_va, best_state):
        for _ in range(n_ep):
            model.train()
            for batch in dl_tr:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(batch["input_ids"], batch["attention_mask"], batch["feat"])
                loss = loss_fn(pred, batch["y"])
                opt.zero_grad(); loss.backward(); opt.step()
                if sch is not None: sch.step()
            model.eval()
            with torch.no_grad():
                vp = []; vy = []
                for batch in dl_va:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    vp.append(model(batch["input_ids"], batch["attention_mask"],
                                    batch["feat"]).cpu().numpy())
                    vy.append(batch["y"].cpu().numpy())
                vp = np.concatenate(vp); vy = np.concatenate(vy)
                va_mse = float(((vp - vy) ** 2).mean())
            if va_mse < best_va:
                best_va = va_mse
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        return best_va, best_state

    best_va = float("inf"); best_state = None

    # Phase 1: frozen encoder, train head
    for p in model.backbone.parameters():
        p.requires_grad = False
    phase1_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(phase1_params, lr=lr, weight_decay=1e-2)
    best_va, best_state = run_epochs(frozen_warmup_epochs, opt1, None,
                                       best_va, best_state)

    # Phase 2: unfrozen encoder with a lower LR, head at same lr
    for p in model.backbone.parameters():
        p.requires_grad = True
    phase2_params = [
        {"params": model.backbone.parameters(), "lr": ft_encoder_lr},
        {"params": model.head.parameters(),     "lr": lr},
    ]
    opt2 = torch.optim.AdamW(phase2_params, weight_decay=1e-2)
    remaining = max(0, epochs - frozen_warmup_epochs)
    if remaining > 0:
        n_steps2 = remaining * len(dl_tr)
        sch2 = get_cosine_schedule_with_warmup(opt2,
            num_warmup_steps=max(10, n_steps2 // 10),
            num_training_steps=n_steps2)
        best_va, best_state = run_epochs(remaining, opt2, sch2, best_va, best_state)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = []
        for batch in dl_te:
            batch = {k: v.to(device) for k, v in batch.items()}
            tp.append(model(batch["input_ids"], batch["attention_mask"],
                            batch["feat"]).cpu().numpy())
        tp = np.concatenate(tp)
    return tp, best_va


def task1(args, device):
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

    print(f"Task 1 ChemBERTa  n_pool={len(pool_y)}  n_test={len(y_te)}")

    seed_r2 = []; seed_preds = []
    for s in range(args.n_seeds):
        rng = np.random.default_rng(42 + s)
        perm = rng.permutation(len(pool_y))
        n_val = max(1, int(round(len(pool_y) * args.val_frac)))
        va_idx = perm[:n_val]; tr_idx = perm[n_val:]
        pred, best_va = train_one(
            pool_smi[tr_idx], pool_f[tr_idx], pool_y[tr_idx],
            pool_smi[va_idx], pool_f[va_idx], pool_y[va_idx],
            smi_te, feat_te, seed=s, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, device=device,
            frozen_warmup_epochs=args.frozen_warmup_epochs,
            ft_encoder_lr=args.ft_encoder_lr,
        )
        r2 = float(r2_score(y_te, pred))
        seed_r2.append(r2); seed_preds.append(pred)
        print(f"  seed {s}: R² = {r2:+.4f}  val_mse={best_va:.4f}", flush=True)

    pred_avg = np.stack(seed_preds).mean(axis=0)
    r2_avg = float(r2_score(y_te, pred_avg))
    mae_avg = float(mean_absolute_error(y_te, pred_avg))
    r2_mu = float(np.mean(seed_r2)); r2_sd = float(np.std(seed_r2))
    print(f"\nTask 1 ChemBERTa  per-seed R² = {r2_mu:+.4f} ± {r2_sd:.4f}")
    print(f"                  R² on seed-averaged preds = {r2_avg:+.4f}   MAE = {mae_avg:.4f}")

    out = {
        "task": "task1", "method": "chembert_ft",
        "n_train": int(len(pool_y)), "n_test": int(len(y_te)),
        "n_seeds": args.n_seeds, "epochs": args.epochs, "lr": args.lr,
        "per_seed_r2": seed_r2,
        "r2_per_seed_mean": r2_mu, "r2_per_seed_std": r2_sd,
        "r2_on_avg_preds": r2_avg, "mae_on_avg_preds": mae_avg,
    }
    with open(RESULTS / "lignos_chembert_task1.json", "w") as fh:
        json.dump(out, fh, indent=2)


def task2(args, device):
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
        smi_pool = pool_smi[tr_mask]; y_pool = pool_y[tr_mask, IDX_LIGNIN]; f_pool = pool_f[tr_mask]
        smi_te = pool_smi[te_mask]; f_te = pool_f[te_mask]

        print(f"\n=== Fold {k}: held ILs = {list(held)} ===  n_train={tr_mask.sum()} n_test={te_mask.sum()}")
        preds = []; seed_r2 = []
        for s in range(args.n_seeds):
            rng = np.random.default_rng(42 + s + 100 * k)
            perm = rng.permutation(len(y_pool))
            n_val = max(1, int(round(len(y_pool) * args.val_frac)))
            va_idx = perm[:n_val]; tr_idx = perm[n_val:]
            pred, best_va = train_one(
                smi_pool[tr_idx], f_pool[tr_idx], y_pool[tr_idx],
                smi_pool[va_idx], f_pool[va_idx], y_pool[va_idx],
                smi_te, f_te, seed=s, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, device=device,
                frozen_warmup_epochs=args.frozen_warmup_epochs,
                ft_encoder_lr=args.ft_encoder_lr,
            )
            preds.append(pred)
            print(f"  seed {s}: mean_pred={pred.mean():+.3f}  val_mse={best_va:.4f}", flush=True)
        pred_avg = np.stack(preds).mean(axis=0)

        csv_path = RESULTS / f"lignos_baran_feat_meta_fold_{k}_rows.csv"
        if not csv_path.exists():
            print(f"Fold {k}: CSV missing"); continue
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != len(pred_avg):
            raise RuntimeError(f"Fold {k}: row count mismatch")
        for i, row in enumerate(rows):
            row["pred_chembert_lig"] = float(pred_avg[i])
        fieldnames = list(rows[0].keys())
        if "pred_chembert_lig" not in fieldnames:
            fieldnames.append("pred_chembert_lig")
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        y_true_arr = np.array([float(r["y_true"]) for r in rows], dtype=np.float32)
        r2 = float(r2_score(y_true_arr, pred_avg))
        print(f"Fold {k}: ChemBERTa R² = {r2:+.4f}  (n_test={len(rows)}, n_seeds={args.n_seeds})")

        summary = {
            "fold": k, "method": "chembert_ft",
            "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
            "n_seeds": args.n_seeds, "epochs": args.epochs,
            "r2": r2, "held_ils": [str(x) for x in held],
        }
        with open(RESULTS / f"lignos_chembert_fold_{k}.json", "w") as fh:
            json.dump(summary, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["task1", "task2"], required=True)
    ap.add_argument("--fold", type=int, default=None, help="Task 2 only; SLURM array.")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=2e-4, help="Head LR (both phases).")
    ap.add_argument("--frozen-warmup-epochs", type=int, default=5,
                    help="Phase 1: encoder frozen, head-only training.")
    ap.add_argument("--ft-encoder-lr", type=float, default=5e-5,
                    help="Phase 2: low LR for unfrozen encoder.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  task={args.task}")
    if args.task == "task1":
        task1(args, device)
    else:
        task2(args, device)


if __name__ == "__main__":
    main()
