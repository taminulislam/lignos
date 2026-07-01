#!/bin/bash
#SBATCH --job-name=stilt_v5
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/stilt_v5_%j.out
#SBATCH --error=../jobs/stilt_v5_%j.err

# ============================================================
# STILT Training (v4 paper methodology applied to v5)
# 3 safeguards: gamma1 masking + distribution filter + 48x oversample
# ============================================================

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "STILT Training (v4 methodology)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python -c "
import sys, time, json, numpy as np, torch, torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path('$PROJECT_ROOT')
V5_ROOT = Path('$V5_ROOT')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V5_ROOT))

from data.stilt_dataset import STILTDataset, masked_mse_loss

PROPS = ['gamma1', 'gamma2', 'G_E', 'H_E', 'G_mix', 'H_vap', 'P']

class V4Model(nn.Module):
    def __init__(self, graph_dim=300, surface_dim=256, thermo_dim=25,
                 fused_dim=256, n_props=7, dropout=0.3):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, fused_dim)
        self.surface_proj = nn.Linear(surface_dim, fused_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.LayerNorm(fused_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.fused_head = nn.Sequential(
            nn.Linear(fused_dim + thermo_dim, 128), nn.BatchNorm1d(128),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(128, n_props))
        self.direct_head = nn.Sequential(
            nn.Linear(graph_dim + thermo_dim, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_props))
        self.gate = nn.Parameter(torch.tensor([2.,2.,-2.,-2.,-2.,0.,1.5]))
    def forward(self, b):
        g,s,t = b['graph_feat'], b['surface_feat'], b['thermo_feat']
        fused = self.fusion_mlp(self.graph_proj(g) * self.surface_proj(s))
        pf = self.fused_head(torch.cat([fused,t],-1))
        pd = self.direct_head(torch.cat([g,t],-1))
        a = torch.sigmoid(self.gate)
        return a*pf + (1-a)*pd

def metrics(p, t, m=None):
    r = {}
    for i,n in enumerate(PROPS):
        if m is not None:
            mk = m[:,i].bool()
            if mk.sum()<2: r[n]=float('nan'); continue
            pi,ti = p[mk,i],t[mk,i]
        else: pi,ti = p[:,i],t[:,i]
        ss_r = ((ti-pi)**2).sum()
        ss_t = ((ti-ti.mean())**2).sum()
        r[n] = (1-ss_r/(ss_t+1e-8)).item()
    v = [x for x in r.values() if not np.isnan(x)]
    r['avg'] = np.mean(v) if v else 0
    return r

def set_seed(s):
    import random; random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
precomp = str(V5_ROOT / 'data/precomputed_chemprop_features.npz')
ilthermo = str(PROJECT_ROOT / 'data/augmented/ilthermo_data.csv')

all_results = {}
for seed in range(10):
    set_seed(seed)
    print(f'\\n{\"#\"*60}\\n  SEED {seed}\\n{\"#\"*60}')

    train_ds = STILTDataset(
        str(PROJECT_ROOT / 'cosmobridge_v4/data/cached_train.npz'),
        ilthermo_csv=ilthermo,
        precomputed_features_path=precomp,
        project_root=str(PROJECT_ROOT),
        include_ilthermo=True,
        oversample_original=48,
        mask_ilthermo_gamma1=True,
        filter_sigma=2.0,
    )
    val_ds = STILTDataset(
        str(PROJECT_ROOT / 'cosmobridge_v4/data/cached_val.npz'),
        project_root=str(PROJECT_ROOT), include_ilthermo=False)
    test_ds = STILTDataset(
        str(PROJECT_ROOT / 'cosmobridge_v4/data/cached_test.npz'),
        project_root=str(PROJECT_ROOT), include_ilthermo=False)

    tl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    vl = torch.utils.data.DataLoader(val_ds, batch_size=128, num_workers=0)
    tel = torch.utils.data.DataLoader(test_ds, batch_size=128, num_workers=0)

    model = V4Model(dropout=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)

    best_val, patience = -1e9, 0
    ckpt = V5_ROOT / f'checkpoints/stilt/seed_{seed}'
    ckpt.mkdir(parents=True, exist_ok=True)

    for ep in range(1, 151):
        model.train()
        loss_sum, nb = 0, 0
        for b in tl:
            b = {k: v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
            p = model(b)
            loss = masked_mse_loss(p, b['targets'], b['mask'])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item(); nb += 1
        sched.step()

        model.eval()
        vp,vt = [],[]
        with torch.no_grad():
            for b in vl:
                b = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
                vp.append(model(b).cpu()); vt.append(b['targets'].cpu())
        vp,vt = torch.cat(vp),torch.cat(vt)
        vm = metrics(vp,vt)

        if ep<=5 or ep%20==0:
            print(f'  Ep {ep:3d} | loss={loss_sum/nb:.4f} | val R2={vm[\"avg\"]:.4f}')

        if vm['avg'] > best_val:
            best_val = vm['avg']; patience = 0
            torch.save(model.state_dict(), ckpt/'best.pt')
        else:
            patience += 1
            if patience >= 30:
                print(f'  Early stop ep {ep} (best={best_val:.4f})')
                break

    model.load_state_dict(torch.load(ckpt/'best.pt', weights_only=True))
    model.eval()
    tp,tt = [],[]
    with torch.no_grad():
        for b in tel:
            b = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
            tp.append(model(b).cpu()); tt.append(b['targets'].cpu())
    tp,tt = torch.cat(tp),torch.cat(tt)
    tm = metrics(tp,tt)

    print(f'\\n  Test (seed {seed}):')
    for n in PROPS:
        print(f'    {n:8s}: R2={tm[n]:.4f}')
    print(f'    avg     : R2={tm[\"avg\"]:.4f}')

    all_results[seed] = tm

    pd_dir = V5_ROOT / 'results/stilt/seed_predictions'
    pd_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pd_dir/f'seed_{seed}.npz', predictions=tp.numpy(), targets=tt.numpy())

print(f'\\n{\"=\"*60}')
print('STILT SUMMARY')
print(f'{\"=\"*60}')
avgs = [m['avg'] for m in all_results.values()]
print(f'  avg R2: {np.mean(avgs):.4f} +/- {np.std(avgs):.4f}')
print(f'  (v4 original: 0.810, v5+SimCLR: 0.657, Path2: 0.642)')

for n in PROPS:
    vals = [m[n] for m in all_results.values() if not np.isnan(m[n])]
    print(f'  {n:8s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')

out = V5_ROOT / 'results/stilt'
out.mkdir(exist_ok=True)
with open(out/'summary.json', 'w') as f:
    json.dump({str(k):v for k,v in all_results.items()}, f, indent=2)
print(f'\\nSaved: {out}/summary.json')
"

echo ""
echo "Done: $(date)"
