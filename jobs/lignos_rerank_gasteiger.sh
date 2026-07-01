#!/bin/bash
#SBATCH --job-name=lignos_rerank
#SBATCH --account=bgte-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --qos=bgte-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/lignos_rerank_%j.out
#SBATCH --error=jobs/logs/lignos_rerank_%j.err

# LIGNOS revision C3: re-rank top lignin candidates with a Gasteiger proxy
# sigma-profile so Specialist C contributes real signal (not masked out).
export PATH=/work/nvme/bgte/tislam6/envs/cosmo/bin:$PATH
cd /work/nvme/bgte/Dataset_Chemistry

echo "=== env check ==="
which python
python -c "import torch, numpy, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'sklearn', sklearn.__version__)"

# Preferred location (move PENDING copy here once scripts/screen is writable):
SCRIPT=lignos/scripts/screen/rerank_gasteiger_sigma.py
if [ ! -f "$SCRIPT" ]; then
  SCRIPT=rerank_gasteiger_sigma.PENDING.py   # staged copy at project root
fi
echo "=== running $SCRIPT ==="
python "$SCRIPT"
