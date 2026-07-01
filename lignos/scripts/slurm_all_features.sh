#!/bin/bash
#SBATCH --job-name=all_feat
#SBATCH --account=bgte-delta-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/all_feat_%j.out
#SBATCH --error=../jobs/all_feat_%j.err

module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg
cd /work/nvme/bgte/kahmed2/Dataset_Chemistry

echo "ALL FEATURES COMPARISON (SimCLR + V-JEPA + Supervised + Sigma) | $(date)"

python3 -c "
import numpy as np, torch, torch.nn as nn, json, hashlib
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from PIL import Image
from torchvision import transforms

V5=Path('lignos')
PROJECT=Path('.')
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

def smiles_to_hash(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]

# ── Extract SimCLR features ──
print('Extracting SimCLR features...')
from lignos.models.multiview_vit import MultiViewViT
simclr_vit = MultiViewViT(n_views=36, embed_dim=192, dropout=0.0).to(device)
ckpt = torch.load(V5/'checkpoints/simclr/vit_pretrained.pt', map_location=device, weights_only=True)
enc_state = ckpt.get('encoder_state_dict', {})
if enc_state:
    simclr_vit.load_state_dict(enc_state, strict=False)
    print('  SimCLR encoder loaded')
simclr_vit.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

cosmo_dirs = [V5/'data/cosmo_images', PROJECT/'data/pipeline/cosmo_images']

def extract_feats(vit_model, smiles_list):
    feats = []
    for smi in smiles_list:
        h = smiles_to_hash(smi)
        frames_dir = None
        for d in cosmo_dirs:
            c = d / f'{h}_frames'
            if c.exists() and len(list(c.glob('frame_*.png'))) >= 2:
                frames_dir = c; break
        if frames_dir is None:
            feats.append(np.zeros(192, dtype=np.float32)); continue
        frames = sorted(frames_dir.glob('frame_*.png'))
        indices = np.linspace(0, len(frames)-1, 6, dtype=int)
        views = torch.stack([transform(Image.open(frames[i]).convert('RGB')) for i in indices])
        views = views.unsqueeze(0).to(device)
        with torch.no_grad():
            emb, _ = vit_model.encode_views_chunked(views, chunk_size=3)
        feats.append(emb.cpu().numpy()[0])
    return np.array(feats, dtype=np.float32)

tc = np.load('cosmobridge_v4/data/cached_train.npz', allow_pickle=True)
tsc = np.load('cosmobridge_v4/data/cached_test.npz', allow_pickle=True)

simclr_train = extract_feats(simclr_vit, list(tc['smiles']))
simclr_test = extract_feats(tsc, list(tsc['smiles'])) if False else None

# Actually extract for test too
simclr_test = extract_feats(simclr_vit, list(tsc['smiles']))
print(f'  SimCLR features: train={simclr_train.shape}, test={simclr_test.shape}')
np.savez(V5/'data/cached_simclr_features.npz', train=simclr_train, test=simclr_test)

del simclr_vit; torch.cuda.empty_cache()

# ── Load all feature sets ──
vj_tr=np.load(V5/'data/cached_image_features_train.npz')['vit_feat']
vj_te=np.load(V5/'data/cached_image_features_test.npz')['vit_feat']
sup=np.load(V5/'data/supervised_vit_features.npz')['features']
sup_tr=sup[:152]; sup_te=sup[152+32:]
sigma=np.load(V5/'data/sigma_profiles.npz')
sig_tr=sigma['train']; sig_te=sigma['test']

# PCA/normalize
pca_vj=PCA(20).fit(vj_tr); pca_sup=PCA(20).fit(sup_tr); pca_sim=PCA(20).fit(simclr_train)
sig_sc=StandardScaler().fit(sig_tr)

vj_tr_p=pca_vj.transform(vj_tr).astype(np.float32)
vj_te_p=pca_vj.transform(vj_te).astype(np.float32)
sup_tr_p=pca_sup.transform(sup_tr).astype(np.float32)
sup_te_p=pca_sup.transform(sup_te).astype(np.float32)
sim_tr_p=pca_sim.transform(simclr_train).astype(np.float32)
sim_te_p=pca_sim.transform(simclr_test).astype(np.float32)
sig_tr_n=sig_sc.transform(sig_tr).astype(np.float32)
sig_te_n=sig_sc.transform(sig_te).astype(np.float32)

# Build all feature combinations
features = {
    'SimCLR(20D)': (sim_tr_p, sim_te_p),
    'V-JEPA(20D)': (vj_tr_p, vj_te_p),
    'Supervised(20D)': (sup_tr_p, sup_te_p),
    'Sigma(50D)': (sig_tr_n, sig_te_n),
    'V-JEPA+Sup(40D)': (np.concatenate([vj_tr_p,sup_tr_p],1).astype(np.float32),
                         np.concatenate([vj_te_p,sup_te_p],1).astype(np.float32)),
    'V-JEPA+Sup+Sigma(90D)': (np.concatenate([vj_tr_p,sup_tr_p,sig_tr_n],1).astype(np.float32),
                                np.concatenate([vj_te_p,sup_te_p,sig_te_n],1).astype(np.float32)),
    'ALL(110D)': (np.concatenate([sim_tr_p,vj_tr_p,sup_tr_p,sig_tr_n],1).astype(np.float32),
                   np.concatenate([sim_te_p,vj_te_p,sup_te_p,sig_te_n],1).astype(np.float32)),
}

# v4 predictions
pdir=Path('cosmobridge_v4/results/seed_predictions')
sf=sorted(pdir.glob('seed_*.npz'))
v4p=np.mean([np.load(f)['preds' if 'preds' in np.load(f) else 'predictions'] for f in sf],0)
v4_tr=(0.4*tc['preds_fusion']+0.6*tc['preds_chemprop']).astype(np.float32)
tr_th=tc['thermo_feat'].astype(np.float32)
te_th=tsc['thermo_feat'].astype(np.float32)
tr_tgt=tc['targets'].astype(np.float32)
te_tgt=tsc['targets'].astype(np.float32)

mv4=metrics(v4p,te_tgt)
print(f'\nv4 router: {mv4[\"avg_r2\"]:.4f}')

class PerPropHead(nn.Module):
    def __init__(self,nf):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,nf),nn.Sigmoid())
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(nf+5,32),nn.GELU(),nn.Linear(32,1)) for _ in range(7)])
        self.alphas=nn.Parameter(torch.full((7,),-3.0))
        for h in self.heads:
            with torch.no_grad(): h[-1].weight.mul_(0.01); h[-1].bias.zero_()
    def forward(self,v,i,t):
        tmp=t[:,:5]; g=i*self.gate(tmp); inp=torch.cat([g,tmp],-1)
        res=torch.cat([h(inp) for h in self.heads],-1)
        return v+torch.sigmoid(self.alphas)*res

results={}
for name,(trf,tef) in features.items():
    sm=[]
    for seed in range(10):
        set_seed(seed)
        m=PerPropHead(trf.shape[1]).to(device)
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
    avgs=[x['avg_r2'] for x in sm]
    results[name]=sm
    print(f'\n{name}: {np.mean(avgs):.4f}+/-{np.std(avgs):.4f} (D={np.mean(avgs)-mv4[\"avg_r2\"]:+.4f})')
    for prop in PROPS:
        vs=[x[f'{prop}_r2'] for x in sm]
        print(f'  {prop:8s}: {np.mean(vs):.4f} (v4={mv4[f\"{prop}_r2\"]:.4f} D={np.mean(vs)-mv4[f\"{prop}_r2\"]:+.4f})')

print(f'\n{\"=\"*70}')
print('DEFINITIVE RANKING')
print(f'{\"=\"*70}')
print(f'v4 router:       {mv4[\"avg_r2\"]:.4f}')
print(f'v4 paper:        0.818')
ranked = sorted(results.items(), key=lambda x: np.mean([m['avg_r2'] for m in x[1]]), reverse=True)
for i,(n,ms) in enumerate(ranked):
    a=[x['avg_r2'] for x in ms]
    print(f'{i+1}. {n:30s}: {np.mean(a):.4f}+/-{np.std(a):.4f} (D={np.mean(a)-mv4[\"avg_r2\"]:+.4f})')

out=V5/'results/all_features'
out.mkdir(exist_ok=True)
with open(out/'summary.json','w') as f:
    json.dump({n:[x for x in ms] for n,ms in results.items()},f,indent=2,default=float)
print(f'\nSaved: {out}/summary.json')
"

echo "Done: $(date)"
