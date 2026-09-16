#!/bin/bash
#SBATCH --job-name=pathembed_remaining
#SBATCH --output=logs/pathembed_remaining_%j.out
#SBATCH --error=logs/pathembed_remaining_%j.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=25g
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH -p b40x4

set -euo pipefail
mkdir -p logs

export HF_HOME=/lustre/nvwulf/scratch/cnwakeze/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# EDIT this list to match whatever's actually still missing after checking
# `ls results/pathology_embeddings/json/` and `.../template/`
python scripts/pathology_generate_embeddings.py --extracted results/pathology_json/pathology_extracted.jsonl --clinical clinical-data.tsv --serialization json --model biobert --outdir results/pathology_embeddings/json
python scripts/pathology_generate_embeddings.py --extracted results/pathology_json/pathology_extracted.jsonl --clinical clinical-data.tsv --serialization json --model scibert --outdir results/pathology_embeddings/json
python scripts/pathology_generate_embeddings.py --extracted results/pathology_json/pathology_extracted.jsonl --clinical clinical-data.tsv --serialization template --model biobert --outdir results/pathology_embeddings/template

echo "Done."
