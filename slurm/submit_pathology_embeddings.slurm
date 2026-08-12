#!/bin/bash
#SBATCH --job-name=pathology_embed
#SBATCH --array=0-9                       # 2 serializations x 5 models
#SBATCH --output=logs/pathembed_%A_%a.out
#SBATCH --error=logs/pathembed_%A_%a.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=25g
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH -p b40x4

set -euo pipefail
mkdir -p logs results/pathology_embeddings

# --- EDIT: activate your environment ---
# (same environment used for generate_embeddings.py -- HoneyBee etc. already installed)

export HF_HOME=/lustre/nvwulf/scratch/cnwakeze/hf_cache

MODELS=(bioclinicalbert pubmedbert biobert scibert sentence-transformers)
SERIALIZATIONS=(json template)

# Map SLURM_ARRAY_TASK_ID (0-9) to (serialization, model) pairs:
# 0-4 -> json + each model, 5-9 -> template + each model
SER_IDX=$((SLURM_ARRAY_TASK_ID / 5))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID % 5))
SERIALIZATION=${SERIALIZATIONS[$SER_IDX]}
MODEL=${MODELS[$MODEL_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID -> serialization=$SERIALIZATION model=$MODEL"

python scripts/pathology/pathology_generate_embeddings.py \
    --extracted results/pathology_json/pathology_extracted.jsonl \
    --clinical clinical-data.tsv \
    --serialization "$SERIALIZATION" \
    --model "$MODEL" \
    --outdir "results/pathology_embeddings/$SERIALIZATION"

echo "Finished: serialization=$SERIALIZATION model=$MODEL"
