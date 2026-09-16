#!/bin/bash
#SBATCH --job-name=pathology_embed
#SBATCH --array=0-9%3                     # 2 serializations x 5 models; %3 caps at 3 concurrent tasks
#SBATCH --output=logs/pathembed_%A_%a.out
#SBATCH --error=logs/pathembed_%A_%a.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=25g
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH -p b40x4

# Always run relative to this script's own folder
cd "$SLURM_SUBMIT_DIR"

set -euo pipefail
mkdir -p logs results/pathology_embeddings

# --- Activate the environment ---
source /lustre/nvwulf/software/miniconda3/etc/profile.d/conda.sh
conda activate honeybee

# Same HF rate-limit lesson as clinical embeddings: redirect cache to scratch,
# and this job's --array=0-9%3 throttle (max 3 concurrent tasks, instead of
# all 10 at once) keeps simultaneous HuggingFace API calls low enough to
# avoid the 429 "Too Many Requests" error we hit earlier.
export HF_HOME=/lustre/nvwulf/scratch/cnwakeze/hf_cache
mkdir -p "$HF_HOME"

MODELS=(bioclinicalbert pubmedbert biobert scibert sentence-transformers)
SERIALIZATIONS=(json template)

# Map SLURM_ARRAY_TASK_ID (0-9) to (serialization, model) pairs:
# 0-4 -> json + each model, 5-9 -> template + each model
SER_IDX=$((SLURM_ARRAY_TASK_ID / 5))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID % 5))
SERIALIZATION=${SERIALIZATIONS[$SER_IDX]}
MODEL=${MODELS[$MODEL_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID -> serialization=$SERIALIZATION model=$MODEL"

python pathology_generate_embeddings.py \
    --extracted ../01_extraction/results/pathology_json/pathology_extracted.jsonl \
    --clinical ../00_raw_data/clinical/clinical_data.tsv \
    --serialization "$SERIALIZATION" \
    --model "$MODEL" \
    --outdir "results/pathology_embeddings/$SERIALIZATION"

echo "Finished: serialization=$SERIALIZATION model=$MODEL"

