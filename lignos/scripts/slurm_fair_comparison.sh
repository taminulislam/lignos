#!/bin/bash
#SBATCH --job-name=fair_cmp
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/fair_cmp_%j.out
#SBATCH --error=../jobs/fair_cmp_%j.err

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry

echo "FAIR COMPARISON: All feature sets x Both architectures | $(date)"

python3 -c "
import numpy as np, torch, torch.nn as nn, json
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.decomposition import PCA
from pathlib import Path

V5=Path('lignos')
PROPS=['gamma1','gamma2','G_E','H_E','G_mix','H_vap','P']
device=torch.device('cuda')

def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def metrics(p,t):
    m={}
    for i,n in enumerate(PROPS):
        sr=((t[:,i]-p[:,i])**2).sum(); st=((t[:,i]-t[:,i].mean())**2).sum()
        m[f'{n}_r2']=(1-sr/(st+1e-8)).item()
    m['avg_r2']=np.mean(list(m.values())); return m

# Load
tc=np.load('cosmobridge_v4/data/cached_train.npz',allow_pickle=True)
tsc=np.load('cosmobridge_v4/data/cached_test.npz',allow_pickle=True)

vj_tr=np.load(V5/'data/cached_image_features_train.npz')['vit_feat']
vj_te=np.load(V5/'data/cached_image_features_test.npz')['vit_feat']
sup=np.load(V5/'data/supervised_vit_features.npz')['features']
sup_tr=sup[:152]; sup_te=sup[152+32:]

pca_vj=PCA(20).fit(vj_tr); pca_sup=PCA(20).fit(sup_tr)
vj_tr_p=pca_vj.transform(vj_tr).astype(np.float32)
vj_te_p=pca_vj.transform(vj_te).astype(np.float32)
sup_tr_p=pca_sup.transform(sup_tr).astype(np.float32)
sup_te_p=pca_sup.transform(sup_te).astype(np.float32)
combo_tr=np.concatenate([vj_tr_p,sup_tr_p],1).astype(np.float32)
combo_te=np.concatenate([vj_te_p,sup_te_p],1).astype(np.float32)

pdir=Path('cosmobridge_v4/results/seed_predictions')
sf=sorted(pdir.glob('seed_*.npz'))
v4p=np.mean([np.load(f)['preds' if 'preds' in np.load(f) else 'predictions'] for f in sf],0)
v4_tr=(0.4*tc['preds_fusion']+0.6*tc['preds_chemprop']).astype(np.float32)
tr_th=tc['thermo_feat'].astype(np.float32)
te_th=tsc['thermo_feat'].astype(np.float32)
tr_tgt=tc['targets'].astype(np.float32)
te_tgt=tsc['targets'].astype(np.float32)

mv4=metrics(v4p,te_tgt)
print(f'v4 router: {mv4[\"avg_r2\"]:.4f}')

# Two architectures
class SharedHead(nn.Module):
    def __init__(self,nf):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,nf),nn.Sigmoid())
        self.head=nn.Sequential(nn.Linear(nf+5,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(0.3),nn.Linear(32,7))
        self.alpha=nn.Parameter(torch.full((7,),-3.0))
        with torch.no_grad(): self.head[-1].weight.mul_(0.01); self.head[-1].bias.zero_()
    def forward(self,v,i,t):
        m=i*self.gate(t[:,:5]); r=self.head(torch.cat([m,t[:,:5]],-1))
        return v+torch.sigmoid(self.alpha)*r

class PerPropHead(nn.Module):
    def __init__(self,nf):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,nf),nn.Sigmoid())
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(nf+5,16),nn.GELU(),nn.Linear(16,1)) for _ in range(7)])
        self.alphas=nn.Parameter(torch.full((7,),-3.0))
        for h in self.heads:
            with torch.no_grad(): h[-1].weight.mul_(0.01); h[-1].bias.zero_()
    def forward(self,v,i,t):
        tmp=t[:,:5]; g=i*self.gate(tmp); inp=torch.cat([g,tmp],-1)
        res=torch.cat([h(inp) for h in self.heads],-1)
        return v+torch.sigmoid(self.alphas)*res

def train_eval(ModelClass, trf, tef, name, seeds=range(10)):
    sm=[]
    for seed in seeds:
        set_seed(seed)
        m=ModelClass(trf.shape[1]).to(device)
        o=AdamW(m.parameters(),lr=5e-4,weight_decay=1e-2)
        s=CosineAnnealingLR(o,T_max=300)
        ldr=DataLoader(TensorDataset(torch.from_numpy(v4_tr),torch.from_numpy(trf),torch.from_numpy(tr_th),torch.from_numpy(tr_tgt)),batch_size=32,shuffle=True)
        b,bs,p=float('inf'),None,0
        for ep in range(300):
            m.train()
            for v,i,t,y in ldr:
                v,i,t,y=[x.to(device) for x in [v,i,t,y]]
                l=((m(v,i,t)-y)**2).mean(); o.zero_grad(); l.backward(); o.step()
            s.step()
            m.eval()
            with torch.no_grad():
                tl=((m(torch.from_numpy(v4_tr).to(device),torch.from_numpy(trf).to(device),torch.from_numpy(tr_th).to(device))-torch.from_numpy(tr_tgt).to(device))**2).mean().item()
            if tl<b: b=tl; bs={k:v.clone() for k,v in m.state_dict().items()}; p=0
            else:
                p+=1
                if p>=50: break
        m.load_state_dict(bs); m.eval()
        with torch.no_grad():
            fn=m(torch.from_numpy(v4p.astype(np.float32)).to(device),torch.from_numpy(tef).to(device),torch.from_numpy(te_th).to(device)).cpu().numpy()
        sm.append(metrics(fn,te_tgt))
    return sm

# Run ALL combinations
print(f'\n{\"=\"*70}')
print('COMPLETE FAIR COMPARISON (3 features x 2 architectures = 6 experiments)')
print(f'{\"=\"*70}')

all_results = {}
for feat_name, trf, tef in [('V-JEPA',vj_tr_p,vj_te_p),('Supervised',sup_tr_p,sup_te_p),('Combined',combo_tr,combo_te)]:
    for arch_name, ModelClass in [('Shared',SharedHead),('PerProp',PerPropHead)]:
        key = f'{feat_name}_{arch_name}'
        sm = train_eval(ModelClass, trf, tef, key)
        avgs = [x['avg_r2'] for x in sm]
        all_results[key] = sm
        print(f'  {key:25s}: {np.mean(avgs):.4f}+/-{np.std(avgs):.4f} (D={np.mean(avgs)-mv4[\"avg_r2\"]:+.4f})')

# Summary table
print(f'\n{\"=\"*70}')
print(f'{\"\":25s} {\"Shared Head\":>15s} {\"Per-Prop Head\":>15s}')
print(f'{\"=\"*70}')
for feat in ['V-JEPA','Supervised','Combined']:
    sh=[x['avg_r2'] for x in all_results[f'{feat}_Shared']]
    pp=[x['avg_r2'] for x in all_results[f'{feat}_PerProp']]
    best_sh = '***' if np.mean(sh) == max(np.mean([x['avg_r2'] for x in v]) for v in all_results.values()) else ''
    best_pp = '***' if np.mean(pp) == max(np.mean([x['avg_r2'] for x in v]) for v in all_results.values()) else ''
    print(f'{feat:25s} {np.mean(sh):.4f}+/-{np.std(sh):.4f} {np.mean(pp):.4f}+/-{np.std(pp):.4f}')

best_key = max(all_results.keys(), key=lambda k: np.mean([x['avg_r2'] for x in all_results[k]]))
best_avg = np.mean([x['avg_r2'] for x in all_results[best_key]])
print(f'\nBEST: {best_key} = {best_avg:.4f} (v4={mv4[\"avg_r2\"]:.4f}, paper=0.818)')

# Per-property for the best
print(f'\nPer-property ({best_key}):')
for prop in PROPS:
    vs=[x[f'{prop}_r2'] for x in all_results[best_key]]
    print(f'  {prop:8s}: {np.mean(vs):.4f}+/-{np.std(vs):.4f} (v4={mv4[f\"{prop}_r2\"]:.4f} D={np.mean(vs)-mv4[f\"{prop}_r2\"]:+.4f})')

out=V5/'results/fair_comparison'
out.mkdir(exist_ok=True)
with open(out/'summary.json','w') as f:
    json.dump({k:[x for x in v] for k,v in all_results.items()},f,indent=2,default=float)
"

echo "Done: $(date)"
