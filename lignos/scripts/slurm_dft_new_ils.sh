#!/bin/bash
#SBATCH --job-name=dft_new
#SBATCH --account=bgte-delta-gpu
#SBATCH --array=1-42%20
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpuA100x4
#SBATCH --output=../jobs/dft_new_%A_%a.out
#SBATCH --error=../jobs/dft_new_%A_%a.err

# ============================================================
# DFT COSMO computation for 42 new ILThermoPy ILs
# Generates: geometry → DFT ESP → point cloud → COSMO images
# ============================================================

module load nwchem/7.2 2>/dev/null || true
module load python/3.10 2>/dev/null || true
source /u/kahmed2/miniconda3/bin/activate mmseg

PROJECT_ROOT="/work/nvme/bgte/kahmed2/Dataset_Chemistry"
V5_ROOT="${PROJECT_ROOT}/lignos"

SMILES=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${V5_ROOT}/scripts/need_dft_smiles.txt")

if [ -z "$SMILES" ]; then
    echo "ERROR: No SMILES for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# Generate a hash ID for this SMILES
HASH_ID=$(python3 -c "import hashlib; print(hashlib.md5('${SMILES}'.encode()).hexdigest()[:12])")

echo "============================================"
echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "SMILES: ${SMILES}"
echo "Hash: ${HASH_ID}"
echo "Date: $(date)"
echo "============================================"

cd "${PROJECT_ROOT}"

# Step 1: Generate 3D conformer + Gasteiger charges → point cloud
python3 -c "
import numpy as np, hashlib
from rdkit import Chem
from rdkit.Chem import AllChem

smiles = '${SMILES}'
h = '${HASH_ID}'

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
AllChem.ComputeGasteigerCharges(mol)

conf = mol.GetConformer()
positions = conf.GetPositions()
charges = [float(a.GetProp('_GasteigerCharge')) if not np.isnan(float(a.GetProp('_GasteigerCharge'))) else 0.0 for a in mol.GetAtoms()]

vdw = {1:1.20, 6:1.70, 7:1.55, 8:1.52, 9:1.47, 15:1.80, 16:1.80, 17:1.75, 35:1.85}
pts, norms, esps = [], [], []
for i, atom in enumerate(mol.GetAtoms()):
    pos = positions[i]
    r = vdw.get(atom.GetAtomicNum(), 1.70)
    n = max(8, 1024 // len(positions))
    phi = np.random.uniform(0, 2*np.pi, n)
    cos_t = np.random.uniform(-1, 1, n)
    theta = np.arccos(cos_t)
    pts.append(np.stack([r*np.sin(theta)*np.cos(phi)+pos[0], r*np.sin(theta)*np.sin(phi)+pos[1], r*np.cos(theta)+pos[2]], 1))
    norms.append(np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], 1))
    esps.append(np.full(n, charges[i]))

pts = np.concatenate(pts); norms = np.concatenate(norms); esps = np.concatenate(esps)
if len(pts) > 1024:
    idx = np.random.choice(len(pts), 1024, replace=False)
    pts, norms, esps = pts[idx], norms[idx], esps[idx]

points = np.column_stack([pts, norms, esps]).astype(np.float32)
np.savez_compressed(f'data/pipeline/point_clouds/{h}.npz', points=points)
print(f'Point cloud saved: {h}.npz ({points.shape})')
"

# Step 2: Try GFN2-xTB geometry optimization if available
if command -v xtb &> /dev/null; then
    echo "Running GFN2-xTB optimization..."
    python scripts/pipeline/step2_geometry_optimization.py --smiles "${SMILES}" --output_id "${HASH_ID}" 2>&1 || echo "xTB failed, using RDKit geometry"
fi

# Step 3: Try NWChem DFT+COSMO if available
if command -v nwchem &> /dev/null; then
    echo "Running NWChem DFT+COSMO..."
    timeout 14400 python scripts/pipeline/step3_dft_esp.py --smiles "${SMILES}" --output_id "${HASH_ID}" 2>&1 || echo "NWChem failed, using Gasteiger charges"
fi

# Step 4: Render 36-view COSMO images
echo "Rendering COSMO views..."
python "${V5_ROOT}/scripts/render_cosmo_views.py" \
    --compound_id "${HASH_ID}" \
    --point_cloud_dir data/pipeline/point_clouds \
    --output_dir "${V5_ROOT}/data/cosmo_images" \
    --n_views 36 --resolution 224 --render_ep

# Step 5: Update point cloud index
python3 -c "
import csv
with open('data/pipeline/point_clouds/index.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(['${SMILES}', '${HASH_ID}.npz', ''])
"

echo "Done: ${HASH_ID} ($(date))"
