"""Two-stage hardfreeze on the extended cache with configurable image backbone.

Variants
--------
  --image-source morgan       : Morgan-FP PCA (baseline; reproduces hardfreeze)
  --image-source vit          : ViT(DFT) PCA only
  --image-source both         : [Morgan_PCA, ViT_PCA] concat (nf doubles)
  --use-physchem              : concat 12-D physchem to ctx (gated by has_physchem)

ViT features come from lignos/data/cached_image_features_*_dft.npz
(152/32/39 rows). They're aggregated per-SMILES (mean-pool across T rows) then
joined by canonical SMILES to every row in the extended LignoIL_unified cache;
rows without a match get zero-filled and masked via `has_vit`.

Each submit should call this script once with a specific flag combination.
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
V4 = PROJECT_ROOT / "cosmobridge_v4"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, CORE_PROPS, r2_per_prop, set_seed  # noqa: E402

EXTENDED_DIR = V5 / "data" / "LignoIL_unified"
PHYSCHEM_DIM = 12
N_SEEDS = 10
MORGAN_DIM = 40
VIT_DIM = 40


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


def load_split(data_dir, split):
    d = np.load(data_dir / f"cached_{split}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def v4_base(c):
    f, p = c.get("preds_fusion"), c.get("preds_chemprop")
    return (0.4 * f + 0.6 * p).astype(np.float32)


def build_vit_smiles_map():
    """Aggregate ViT(DFT) features to one vector per canonical SMILES, using
    ALL three splits (train/val/test) so held-out test ILs also get coverage.
    Without this the test set has 0% ViT coverage and lignin predictions
    collapse (tested in 17702910/11/12: lignin went negative)."""
    acc = {}
    for split in ["train", "val", "test"]:
        cache = np.load(V4 / f"data/cached_{split}.npz", allow_pickle=True)
        smi = np.array([s.decode() if isinstance(s, bytes) else s for s in cache["smiles"]])
        vit = np.load(V5 / f"data/cached_image_features_{split}_dft.npz")["vit_feat"]
        assert len(smi) == len(vit), f"{split} smi {len(smi)} vs vit {len(vit)}"
        for s, f in zip(smi, vit):
            c = canon(s)
            if not c:
                continue
            acc.setdefault(c, []).append(f)
    return {k: np.mean(vs, axis=0).astype(np.float32) for k, vs in acc.items()}


def attach_vit(data, vit_map):
    """Add (has_vit, vit_feat) keys to a split dict."""
    smi = np.array([s.decode() if isinstance(s, bytes) else s for s in data["smiles"]])
    N = len(smi)
    vit_dim = next(iter(vit_map.values())).shape[0] if vit_map else 192
    vit_feat = np.zeros((N, vit_dim), dtype=np.float32)
    has_vit = np.zeros(N, dtype=bool)
    for i, s in enumerate(smi):
        c = canon(s) or s
        vec = vit_map.get(c)
        if vec is not None:
            vit_feat[i] = vec
            has_vit[i] = True
    return vit_feat, has_vit


def preprocess_physchem_train(phys_feat, has_physchem):
    x = phys_feat.astype(np.float32).copy()
    x[:, 3] = np.log1p(np.maximum(x[:, 3], 0.0))
    x[:, 5] = np.log1p(np.maximum(x[:, 5], 0.0))
    covered = has_physchem.astype(bool)
    if covered.sum() > 0:
        mu = x[covered].mean(axis=0); sd = x[covered].std(axis=0) + 1e-6
    else:
        mu = np.zeros(x.shape[1], dtype=np.float32); sd = np.ones(x.shape[1], dtype=np.float32)
    return ((x - mu) / sd * covered[:, None]).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def preprocess_physchem_apply(phys_feat, has_physchem, mu, sd):
    x = phys_feat.astype(np.float32).copy()
    x[:, 3] = np.log1p(np.maximum(x[:, 3], 0.0))
    x[:, 5] = np.log1p(np.maximum(x[:, 5], 0.0))
    covered = has_physchem.astype(bool)
    return (((x - mu) / sd) * covered[:, None]).astype(np.float32)


class PerPropHeadV(nn.Module):
    """Generic head: nf-D image slot `i`, optional physchem concat to ctx."""
    def __init__(self, nf, n_props=8, use_physchem=False, physchem_dim=PHYSCHEM_DIM):
        super().__init__()
        self.use_physchem = use_physchem
        self.gate = nn.Sequential(
            nn.Linear(5, 32), nn.GELU(), nn.Linear(32, nf), nn.Sigmoid()
        )
        ctx_dim = 5 + (physchem_dim if use_physchem else 0)
        head_in = nf + ctx_dim
        heads = []
        for _ in range(n_props):
            heads.append(nn.Sequential(
                nn.Linear(head_in, 32), nn.GELU(), nn.Linear(32, 1)
            ))
        self.heads = nn.ModuleList(heads)
        self.alphas = nn.Parameter(torch.full((n_props,), -3.0))
        for h in self.heads:
            with torch.no_grad():
                h[-1].weight.mul_(0.01); h[-1].bias.zero_()

    def forward(self, v, i, t, phys=None):
        tmp = t[:, :5]
        g = i * self.gate(tmp)
        ctx = tmp if not self.use_physchem else torch.cat([tmp, phys], -1)
        inp = torch.cat([g, ctx], -1)
        res = torch.cat([h(inp) for h in self.heads], -1)
        return v + torch.sigmoid(self.alphas) * res


def train_one_seed(seed, tr_v, tr_f, tr_th, tr_phys, tr_y, device, use_physchem,
                    epochs=300, patience=50):
    set_seed(seed)
    n_props = tr_y.shape[1]
    model = PerPropHeadV(tr_f.shape[1], n_props=n_props, use_physchem=use_physchem).to(device)
    opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    tr_v_t = torch.from_numpy(tr_v).to(device)
    tr_f_t = torch.from_numpy(tr_f).to(device)
    tr_th_t = torch.from_numpy(tr_th).to(device)
    tr_p_t = torch.from_numpy(tr_phys).to(device) if tr_phys is not None else None
    tr_y_t = torch.from_numpy(tr_y).to(device)

    valid_mask = ~torch.isnan(tr_y_t)
    y_fill = torch.nan_to_num(tr_y_t, nan=0.0)
    weights = torch.ones(n_props, device=device)

    ds = TensorDataset(
        tr_v_t.cpu(), tr_f_t.cpu(), tr_th_t.cpu(),
        (tr_p_t.cpu() if tr_p_t is not None else torch.zeros(len(tr_v_t), 1)),
        y_fill.cpu(), valid_mask.cpu()
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    best_loss, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for v, im, t, p, y, m in loader:
            v, im, t, y, m = v.to(device), im.to(device), t.to(device), y.to(device), m.to(device)
            p = p.to(device) if tr_p_t is not None else None
            pred = model(v, im, t, p)
            err2 = ((pred - y) ** 2) * m
            per_prop = err2.sum(0) / m.sum(0).clamp(min=1)
            loss = per_prop.mean()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(tr_v_t, tr_f_t, tr_th_t, tr_p_t)
            err2 = ((pred - y_fill) ** 2) * valid_mask
            tl = (err2.sum(0) / valid_mask.sum(0).clamp(min=1)).mean().item()
        if np.isfinite(tl) and tl < best_loss:
            best_loss, best_state, bad = tl, {k: vv.clone() for k, vv in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval(); return model


def predict(model, v, f, th, phys, device):
    v_t = torch.from_numpy(v).to(device)
    f_t = torch.from_numpy(f).to(device)
    th_t = torch.from_numpy(th).to(device)
    p_t = torch.from_numpy(phys).to(device) if phys is not None else None
    model.eval()
    with torch.no_grad():
        return model(v_t, f_t, th_t, p_t).cpu().numpy()


def train_stage2_hardfreeze(stage1, tr_v, tr_f, tr_th, tr_phys, tr_y, device, seed,
                              use_physchem, epochs=300, patience=50):
    set_seed(seed)
    model = copy.deepcopy(stage1).to(device)
    for p in model.parameters():
        p.requires_grad = False
    nf = model.gate[2].out_features
    ctx_dim = 5 + (PHYSCHEM_DIM if use_physchem else 0)
    model.heads[7] = nn.Sequential(
        nn.Linear(nf + ctx_dim, 128), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(64, 1),
    ).to(device)
    with torch.no_grad():
        model.heads[7][-1].weight.mul_(0.01); model.heads[7][-1].bias.zero_()
    for p in model.heads[7].parameters():
        p.requires_grad = True
    model.alphas.requires_grad = True
    mask = torch.zeros_like(model.alphas); mask[7] = 1.0
    model.alphas.register_hook(lambda g: g * mask)
    head7 = list(model.heads[7].parameters())
    opt = AdamW([
        {"params": head7, "weight_decay": 1e-2},
        {"params": [model.alphas], "weight_decay": 0.0},
    ], lr=1e-3)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    opt_params = head7 + [model.alphas]

    tr_v_t = torch.from_numpy(tr_v).to(device); tr_f_t = torch.from_numpy(tr_f).to(device)
    tr_th_t = torch.from_numpy(tr_th).to(device)
    tr_p_t = torch.from_numpy(tr_phys).to(device) if tr_phys is not None else None
    tr_y_t = torch.from_numpy(tr_y).to(device)
    ds = TensorDataset(
        tr_v_t.cpu(), tr_f_t.cpu(), tr_th_t.cpu(),
        (tr_p_t.cpu() if tr_p_t is not None else torch.zeros(len(tr_v_t), 1)),
        tr_y_t.cpu()
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    best_loss, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for v, im, t, p, y in loader:
            v, im, t, y = v.to(device), im.to(device), t.to(device), y.to(device)
            p = p.to(device) if tr_p_t is not None else None
            pred = model(v, im, t, p)
            lig = ~torch.isnan(y[:, 7])
            if lig.sum() == 0:
                continue
            loss = ((pred[lig, 7] - y[lig, 7].nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(tr_v_t, tr_f_t, tr_th_t, tr_p_t)
            lig = ~torch.isnan(tr_y_t[:, 7])
            tl = ((pred[lig, 7] - tr_y_t[lig, 7].nan_to_num(0)) ** 2).mean().item() if lig.any() else float("inf")
        if np.isfinite(tl) and tl < best_loss:
            best_loss, best_state, bad = tl, {k: vv.clone() for k, vv in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval(); return model


def summarize(name, r2s):
    core7 = [r["avg_core7"] for r in r2s]
    out = {"name": name, "avg_r2_core7": float(np.mean(core7)),
           "std_r2_core7": float(np.std(core7)), "per_prop": {}}
    for p in PROPS:
        vals = [r.get(p) for r in r2s if r.get(p) is not None and not np.isnan(r.get(p, float("nan")))]
        out["per_prop"][p] = float(np.mean(vals)) if vals else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-source", choices=["morgan", "vit", "both"], required=True)
    ap.add_argument("--use-physchem", action="store_true")
    ap.add_argument("--tag", default="vit_experiment")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label = f"{args.image_source}{'_physchem' if args.use_physchem else ''}"
    print(f"Device: {device}  |  image={args.image_source}  physchem={args.use_physchem}  label={label}")

    # Load extended cache
    tr = load_split(EXTENDED_DIR, "train")
    te = load_split(EXTENDED_DIR, "test")
    v4_tr, v4_te = v4_base(tr), v4_base(te)
    y_tr = tr["targets"].astype(np.float32); y_te = te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]

    # Image features
    feats_tr, feats_te = [], []
    if args.image_source in ("morgan", "both"):
        pca_m = PCA(MORGAN_DIM).fit(tr["morgan_fp"])
        feats_tr.append(pca_m.transform(tr["morgan_fp"]).astype(np.float32))
        feats_te.append(pca_m.transform(te["morgan_fp"]).astype(np.float32))
    if args.image_source in ("vit", "both"):
        vit_map = build_vit_smiles_map()
        vit_tr, has_vit_tr = attach_vit(tr, vit_map)
        vit_te, has_vit_te = attach_vit(te, vit_map)
        print(f"  ViT coverage train: {has_vit_tr.sum()}/{len(has_vit_tr)}  test: {has_vit_te.sum()}/{len(has_vit_te)}")
        # PCA on the covered train rows, then apply; zero-fill uncovered.
        pca_v = PCA(VIT_DIM).fit(vit_tr[has_vit_tr])
        vit_tr_p = pca_v.transform(vit_tr).astype(np.float32) * has_vit_tr[:, None]
        vit_te_p = pca_v.transform(vit_te).astype(np.float32) * has_vit_te[:, None]
        feats_tr.append(vit_tr_p); feats_te.append(vit_te_p)
    f_tr = np.concatenate(feats_tr, axis=1); f_te = np.concatenate(feats_te, axis=1)
    print(f"  Final image-feature dim: {f_tr.shape[1]}")

    # Physchem
    p_tr = p_te = None
    if args.use_physchem:
        p_tr, mu, sd = preprocess_physchem_train(tr["physchem_feat"], tr["has_physchem"])
        p_te = preprocess_physchem_apply(te["physchem_feat"], te["has_physchem"], mu, sd)
        print(f"  physchem coverage train: {int(tr['has_physchem'].sum())}/{len(tr['has_physchem'])}")

    # Stage 1
    print(f"\n{'='*60}\nStage 1: shallow head, {N_SEEDS} seeds\n{'='*60}")
    stage1_models, s1_r2 = [], []
    for seed in range(N_SEEDS):
        m = train_one_seed(seed, v4_tr, f_tr, th_tr, p_tr, y_tr, device, args.use_physchem)
        stage1_models.append(m)
        s1_r2.append(r2_per_prop(predict(m, v4_te, f_te, th_te, p_te, device), y_te))
    s1 = summarize(f"Stage1_{label}", s1_r2)
    print(f"Stage1 core7={s1['avg_r2_core7']:.4f} ± {s1['std_r2_core7']:.4f}")
    for p in PROPS:
        print(f"  {p:12s}: {s1['per_prop'][p]:.4f}")

    # Stage 2 hardfreeze
    print(f"\n{'='*60}\nStage 2 hardfreeze, {N_SEEDS} seeds\n{'='*60}")
    s2_r2 = []
    for seed in range(N_SEEDS):
        s2 = train_stage2_hardfreeze(stage1_models[seed], v4_tr, f_tr, th_tr, p_tr, y_tr,
                                       device, seed + 100, args.use_physchem)
        s2_r2.append(r2_per_prop(predict(s2, v4_te, f_te, th_te, p_te, device), y_te))
    s2s = summarize(f"Stage2_hardfreeze_{label}", s2_r2)
    print(f"Stage2 core7={s2s['avg_r2_core7']:.4f} ± {s2s['std_r2_core7']:.4f}")
    for p in PROPS:
        print(f"  {p:12s}: {s2s['per_prop'][p]:.4f}")

    results = {"label": label, "image_source": args.image_source,
               "use_physchem": args.use_physchem,
               "stage1": s1, "stage2_hardfreeze": s2s}
    out = V5 / "results" / f"vit_experiment_{label}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
