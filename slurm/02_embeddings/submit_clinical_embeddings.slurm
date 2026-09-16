#!/bin/bash
#SBATCH --job-name=clinical_embed
#SBATCH --array=0-4                       # one task per model (5 models)
#SBATCH --output=logs/embed_%A_%a.out
#SBATCH --error=logs/embed_%A_%a.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=25g
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH -p b40x4                          # 11 nodes as of last check; switch to h200x4/h200x8 if you need more VRAM

# Always run relative to this script's own folder
cd "$SLURM_SUBMIT_DIR"

set -euo pipefail
mkdir -p logs results/embeddings

# --- Activate the environment ---
source /lustre/nvwulf/software/miniconda3/etc/profile.d/conda.sh
conda activate honeybee

MODELS=(bioclinicalbert pubmedbert biobert scibert sentence-transformers)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID -> model: $MODEL"

python clinical_generate_embeddings.py \
    --model "$MODEL" \
    --tsv ../00_raw_data/clinical/clinical_data.tsv \
    --outdir results/embeddings

echo "Finished model: $MODEL"
