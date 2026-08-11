#!/bin/bash
#SBATCH --job-name=wsi_embed
#SBATCH --output=logs/wsi_embed_%j.out
#SBATCH --error=logs/wsi_embed_%j.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH -p b40x4

set -euo pipefail
mkdir -p logs results/wsi_embeddings

export HF_HOME=/lustre/nvwulf/scratch/cnwakeze/hf_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

python scripts/wsi_generate_embeddings.py \
    --manifest results/wsi/wsi_manifest.csv \
    --model uni \
    --outdir results/wsi_embeddings \
    --max-patches-per-slide 500

echo "Done."
