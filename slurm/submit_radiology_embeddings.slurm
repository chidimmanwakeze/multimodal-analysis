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

set -euo pipefail
mkdir -p logs results/radiology_embeddings

# Redirect torch's model cache to scratch -- same quota lesson as HF_HOME and
# OLLAMA_MODELS earlier. Only 30MB for densenet121 in testing, but no reason
# to risk it against the 20GB home quota.
export TORCH_HOME=/lustre/nvwulf/scratch/cnwakeze/torch_cache
mkdir -p "$TORCH_HOME"

python scripts/radiology_generate_embeddings.py \
    --imaging-metadata imaging-metadata.csv \
    --clinical clinical-data.tsv \
    --model radimagenet-densenet121 \
    --outdir results/radiology_embeddings \
    --max-filesize 300000000

echo "Done."
