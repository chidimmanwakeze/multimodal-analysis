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
#SBATCH -p b40x4                          # 9 nodes as of last check; switch to h200x4/h200x8 if you need more VRAM

set -euo pipefail
mkdir -p logs results/embeddings

# --- EDIT: load/activate your environment ---
# module load anaconda3
# conda activate honeybee-env
# or, if HoneyBee lives in your home dir venv:
# source /lustre/nvwulf/home/cnwakeze/venvs/honeybee/bin/activate

MODELS=(bioclinicalbert pubmedbert biobert scibert sentence-transformers)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID -> model: $MODEL"

python scripts/clinical/generate_embeddings.py \
    --model "$MODEL" \
    --tsv clinical-data.tsv \
    --outdir results/embeddings

echo "Finished model: $MODEL"
