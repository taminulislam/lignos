"""A5.9 Stage-2 — add A2_2stg-style deep lignin head on top of the fused
scalar-router BMA model, giving us the combined headline:

  core7 (gated@50%) = 0.9344   (from A5.9 fused ensemble, held constant)
  lignin_wt         = 0.70+    (from Stage-2 deep head + physchem)

Pipeline:
  1. Load 3 frozen specialist checkpoints (A, B, C) from a5_bma/.
  2. Train scalar BMA router on frozen specialists (fast, ~1 min).
  3. Build Stage-2 wrapper:
       - `backbone` = (specialists + router) producing fused (μ, σ²)
       - `deep_lignin` = 3-layer head on [gated_morgan, thermo(5), physchem(12), has_phys(1)]
       - `alpha_lignin` = trainable per-row gate (inherited logic from A2StageTwoLigninWrapper)
  4. Train ONLY `deep_lignin` + `alpha_lignin` on lignin-labeled rows.
  5. At inference: core-7 fused μ unchanged; lignin column overridden with
     `v[:, 7] + sigmoid(alpha_lignin) * deep_lignin(ctx)`.
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V5 = PROJECT_ROOT / "lignos"
sys.path.insert(0, str(V5 / "scripts"))
from audit_residuals import PROPS, r2_per_prop, set_seed  # noqa
from train_a2_two_stage import (
    build_chemprop_40d, preprocess_physchem, v4_base,
)
from train_a5_bma_pipeline import (
    A5_BMA_Specialist, A5_BMA_Router, train_router,
    _assemble_bank, _standardize, _load_split,
    FRAME_DIM, SURFACE_DIM, COSMO_DIM, K_SPECIALISTS,
    VIT_BANK, COSMO_BANK, BMA_DIR,
)

CACHE = V5 / "data" / "LignoIL_A1"


class A5_BMA_Stage2(nn.Module):
    """Wraps frozen A5.9 fused ensemble + trainable Stage-2 deep lignin head."""

    def __init__(self, specialists_dict, router, nf, n_props=8, physchem_dim=12):
        super().__init__()
        # Freeze everything under `backbone` (specialists + router)
        self.spec_A = specialists_dict["A"]
        self.spec_B = specialists_dict["B"]
        self.spec_C = specialists_dict["C"]
        self.router = router
        for m in (self.spec_A, self.spec_B, self.spec_C, self.router):
            for p in m.parameters():
                p.requires_grad = False

        # Shared A2 gate/alpha come from any specialist's inherited A2Head
        self.gate_fn = self.spec_A.gate          # gated-morgan projection (shared)
        # alpha_lignin init = specialist A's alphas[7]
        self.alpha_lignin = nn.Parameter(self.spec_A.alphas.data[7].clone())

        head_in = nf + 5 + physchem_dim + 1
        self.deep_lignin = nn.Sequential(
            nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        with torch.no_grad():
            self.deep_lignin[-1].weight.mul_(0.01); self.deep_lignin[-1].bias.zero_()

    def _fused(self, v, i, t, chemprop, surface, vit, cos, hs, hv, hc):
        mus, lvs = [], []
        with torch.no_grad():
            for m in (self.spec_A, self.spec_B, self.spec_C):
                mu, lv = m.forward_with_lv(v, i, t, chemprop,
                                             surface=surface, vit=vit, cos=cos,
                                             has_surf=hs, has_vit=hv, has_cos=hc)
                mus.append(mu); lvs.append(lv)
            mu_s = torch.stack(mus, dim=1)                    # (B, K, P)
            lv_s = torch.stack(lvs, dim=1)
            w = self.router(chemprop, surface, t, lv_s)       # (B, K, P)
            mu_f = (w * mu_s).sum(dim=1)
            prec = torch.exp(-lv_s).sum(dim=1) + 1e-8
            lv_f = -torch.log(prec)
        return mu_f, lv_f

    def forward(self, v, i, t, chemprop, surface, vit, cos, hs, hv, hc,
                phys, has_phys):
        mu_f, lv_f = self._fused(v, i, t, chemprop, surface, vit, cos, hs, hv, hc)
        # Stage-2 lignin override (column 7)
        tmp = t[:, :5]
        g = i * self.gate_fn(tmp)
        hp = has_phys.float().unsqueeze(-1) if has_phys.ndim == 1 else has_phys.float()
        ctx = torch.cat([g, tmp, phys, hp], -1)
        res_lignin = self.deep_lignin(ctx).squeeze(-1)
        out = mu_f.clone()
        out[:, 7] = v[:, 7] + torch.sigmoid(self.alpha_lignin) * res_lignin
        return out, lv_f


def train_stage2(model, v4, morg, th, cp, surf, vit, cos, hs, hv, hc,
                  phys, hp, y, device, seed, epochs=300, patience=50):
    set_seed(seed)
    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW([{"params": model.deep_lignin.parameters(), "weight_decay": 1e-2},
                  {"params": [model.alpha_lignin], "weight_decay": 0.0}], lr=1e-3)
    sch = CosineAnnealingLR(opt, T_max=epochs)

    ts = {k: torch.from_numpy(x).to(device) for k, x in dict(
        v=v4, i=morg, t=th, cp=cp, surf=surf, vit=vit, cos=cos,
        hs=hs, hv=hv, hc=hc, p=phys, hp=hp, y=y).items()}

    ds = TensorDataset(*[ts[k].cpu() for k in
                          ("v","i","t","cp","surf","vit","cos","hs","hv","hc","p","hp","y")])
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = [x.to(device) for x in batch]
            vb, ib, tb, cpb, sub, vib, cob, hsb, hvb, hcb, pb, hpb, yb = batch
            pred, _ = model(vb, ib, tb, cpb, sub, vib, cob, hsb, hvb, hcb, pb, hpb)
            lg = ~torch.isnan(yb[:, 7])
            if lg.sum() == 0: continue
            loss = ((pred[lg, 7] - yb[lg, 7].nan_to_num(0)) ** 2).mean()
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            pred, _ = model(ts["v"], ts["i"], ts["t"], ts["cp"],
                              ts["surf"], ts["vit"], ts["cos"],
                              ts["hs"], ts["hv"], ts["hc"],
                              ts["p"], ts["hp"])
            lg = ~torch.isnan(ts["y"][:, 7])
            tl = ((pred[lg, 7] - ts["y"][lg, 7].nan_to_num(0)) ** 2).mean().item() if lg.any() else float("inf")
        if np.isfinite(tl) and tl < best:
            best = tl; state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if state is not None: model.load_state_dict(state)
    model.eval()
    return model


def predict_stage2(model, feats, device):
    with torch.no_grad():
        pred, lv = model(*(torch.from_numpy(feats[k]).to(device) for k in
                            ("v4","morg","thermo","chemprop","surface","vit","cos",
                             "has_surf","has_vit","has_cos","physchem","has_physchem")))
    return pred.cpu().numpy(), lv.cpu().numpy()


def conf_gated_r2(pred, y, logvar, quantile=0.5):
    sigma = np.exp(0.5 * logvar).mean(axis=-1)
    thr = np.quantile(sigma, quantile)
    keep = sigma <= thr
    r = {}
    for i, p in enumerate(PROPS):
        valid = ~np.isnan(y[:, i]) & keep
        if valid.sum() < 2: continue
        yk = y[valid, i]; pk = pred[valid, i]
        ss_res = ((yk - pk) ** 2).sum()
        ss_tot = ((yk - yk.mean()) ** 2).sum() + 1e-12
        r[p] = 1.0 - ss_res / ss_tot
    r["avg_core7"] = float(np.mean([r[p] for p in PROPS[:7] if p in r and np.isfinite(r[p])]))
    return r


def summarize(name, r2s):
    c = [r["avg_core7"] for r in r2s]
    out = {"name": name, "avg_r2_core7": float(np.mean(c)),
           "std_r2_core7": float(np.std(c)), "per_prop": {}}
    for p in PROPS:
        vs = [r.get(p) for r in r2s if r.get(p) is not None and not np.isnan(r.get(p, float("nan")))]
        out["per_prop"][p] = float(np.mean(vs)) if vs else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--router-mode", choices=["mlp", "scalar"], default="scalar")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  router_mode: {args.router_mode}")

    tr, va, te = _load_split("train"), _load_split("val"), _load_split("test")
    pca_m = PCA(40).fit(tr["morgan_fp"])
    m_tr, m_va, m_te = [pca_m.transform(x["morgan_fp"]).astype(np.float32)
                         for x in (tr, va, te)]
    cp_tr, cp_te = build_chemprop_40d(tr["chemprop_fp"], te["chemprop_fp"])
    _, cp_va = build_chemprop_40d(tr["chemprop_fp"], va["chemprop_fp"])
    p_tr, p_te = preprocess_physchem(tr["physchem_feat"], tr["has_physchem"],
                                      te["physchem_feat"], te["has_physchem"])
    hp_tr = tr["has_physchem"].astype(np.float32)
    hp_te = te["has_physchem"].astype(np.float32)

    surf_tr = tr["surface_fp"].astype(np.float32); hs_tr = (surf_tr != 0).any(axis=1).astype(np.float32)
    surf_te = te["surface_fp"].astype(np.float32); hs_te = (surf_te != 0).any(axis=1).astype(np.float32)
    surf_tr, mu_p, sd_p = _standardize(surf_tr, hs_tr)
    surf_te = ((surf_te - mu_p) / sd_p).astype(np.float32) * hs_te[:, None]

    vit_bank = dict(zip(*[np.load(VIT_BANK, allow_pickle=True)[k]
                             for k in ("smiles", "vit_feat")]))
    vit_tr, hv_tr = _assemble_bank(tr["smiles"], vit_bank, FRAME_DIM)
    vit_te, hv_te = _assemble_bank(te["smiles"], vit_bank, FRAME_DIM)
    vit_tr, mu_v, sd_v = _standardize(vit_tr, hv_tr)
    vit_te = ((vit_te - mu_v) / sd_v).astype(np.float32) * hv_te[:, None]

    cos_bank = dict(zip(*[np.load(COSMO_BANK, allow_pickle=True)[k]
                             for k in ("smiles", "cosmo_feat")]))
    cos_tr, hc_tr = _assemble_bank(tr["smiles"], cos_bank, COSMO_DIM)
    cos_te, hc_te = _assemble_bank(te["smiles"], cos_bank, COSMO_DIM)
    cos_tr, mu_c, sd_c = _standardize(cos_tr, hc_tr)
    cos_te = ((cos_te - mu_c) / sd_c).astype(np.float32) * hc_te[:, None]

    v4_tr, v4_te = v4_base(tr), v4_base(te)
    y_tr, y_te = tr["targets"].astype(np.float32), te["targets"].astype(np.float32)
    th_tr, th_te = tr["thermo_feat"], te["thermo_feat"]

    feats_tr = {"v4": v4_tr, "morg": m_tr, "thermo": th_tr, "chemprop": cp_tr,
                 "surface": surf_tr, "vit": vit_tr, "cos": cos_tr,
                 "has_surf": hs_tr, "has_vit": hv_tr, "has_cos": hc_tr}
    feats_te = {"v4": v4_te, "morg": m_te, "thermo": th_te, "chemprop": cp_te,
                 "surface": surf_te, "vit": vit_te, "cos": cos_te,
                 "has_surf": hs_te, "has_vit": hv_te, "has_cos": hc_te,
                 "physchem": p_te, "has_physchem": hp_te}

    # Load specialists
    specialists = {}
    for kind in ("A", "B", "C"):
        ck = torch.load(BMA_DIR / f"specialist_{kind}.pt",
                         map_location=device, weights_only=False)
        m = A5_BMA_Specialist(kind, m_tr.shape[1], y_tr.shape[1],
                                 chemprop_dim=cp_tr.shape[1]).to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        specialists[kind] = m
        print(f"[Sp {kind}] loaded core7={ck.get('test_core7'):.4f}")

    # Train router (quick — 24 params for scalar)
    print(f"\nTraining scalar router on frozen specialists...")
    set_seed(2026)
    router = train_router([specialists["A"], specialists["B"], specialists["C"]],
                            feats_tr, y_tr, device, epochs=args.epochs // 2,
                            mode=args.router_mode)
    router.eval()
    print(f"Router trained (mode={args.router_mode}, params="
           f"{sum(p.numel() for p in router.parameters())})")

    # Stage-2 training (deep lignin + alpha_lignin only)
    s2_r2s, s2_gated50, s2_gated25 = [], [], []
    nf = m_tr.shape[1]
    for seed in range(args.n_seeds):
        print(f"\n[Stage-2] seed {seed} ...")
        model = A5_BMA_Stage2(specialists, router, nf, y_tr.shape[1]).to(device)
        model = train_stage2(model, v4_tr, m_tr, th_tr, cp_tr,
                              surf_tr, vit_tr, cos_tr, hs_tr, hv_tr, hc_tr,
                              p_tr, hp_tr, y_tr,
                              device, seed=seed, epochs=args.epochs)
        pred_te, lv_te = predict_stage2(model, feats_te, device)
        r = r2_per_prop(pred_te, y_te)
        g50 = conf_gated_r2(pred_te, y_te, lv_te, quantile=0.5)
        g25 = conf_gated_r2(pred_te, y_te, lv_te, quantile=0.25)
        s2_r2s.append(r); s2_gated50.append(g50); s2_gated25.append(g25)
        print(f"  core7={r['avg_core7']:.4f}  lignin={r.get('lignin_wt', float('nan')):.4f}  "
              f"gated@50% core7={g50['avg_core7']:.4f}  alpha_lignin={model.alpha_lignin.item():+.3f}")

    s = summarize("Stage2_A5_BMA_fused", s2_r2s)
    s50 = summarize("Stage2_A5_BMA_fused_gated50", s2_gated50)
    s25 = summarize("Stage2_A5_BMA_fused_gated25", s2_gated25)
    print(f"\n{'='*72}\nA5.9 Stage-2 Summary ({args.router_mode} router)\n{'='*72}")
    print(f"{'Arm':<42}{'core7':>10}{'std':>10}{'lignin':>10}")
    for r in [s, s50, s25]:
        lig = r["per_prop"].get("lignin_wt", float("nan"))
        print(f"{r['name']:<42}{r['avg_r2_core7']:>10.4f}{r['std_r2_core7']:>10.4f}{lig:>10.4f}")

    outf = V5 / "results" / f"a5_bma_stage2_{args.router_mode}.json"
    json.dump([s, s50, s25], open(outf, "w"), indent=2)
    print(f"\nSaved: {outf}")


if __name__ == "__main__":
    main()
