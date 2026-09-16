#!/bin/bash
#SBATCH --job-name=radiology_embed
#SBATCH --output=logs/radiology_%j.out
#SBATCH --error=logs/radiology_%j.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH -p b40x4

# Always run relative to this script's own folder
cd "$SLURM_SUBMIT_DIR"

set -euo pipefail
mkdir -p logs results/radiology_embeddings

# --- Activate the environment (has torch, radimagenet, etc.) ---
source /lustre/nvwulf/software/miniconda3/etc/profile.d/conda.sh
conda activate honeybee

# Redirect torch's model cache to scratch -- same quota lesson as HF_HOME and
# OLLAMA_MODELS earlier. Only 30MB for densenet121 in testing, but no reason
# to risk it against the 20GB home quota.
export TORCH_HOME=/lustre/nvwulf/scratch/cnwakeze/torch_cache
mkdir -p "$TORCH_HOME"

python radiology_generate_embeddings.py \
    --imaging-metadata ../00_raw_data/radiology/radiology_metadata.csv \
    --clinical ../00_raw_data/clinical/clinical_data.tsv \
    --model radimagenet-densenet121 \
    --outdir results/radiology_embeddings \
    --max-filesize 300000000

echo "Done."
