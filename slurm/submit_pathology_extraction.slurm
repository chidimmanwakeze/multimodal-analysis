#!/bin/bash
#SBATCH --job-name=pathology_extract
#SBATCH --output=logs/pathology_%j.out
#SBATCH --error=logs/pathology_%j.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH -p b40x4

set -euo pipefail
mkdir -p logs results/pathology_json

# --- EDIT: load Ollama module (check exact name with `module avail ollama`) ---
module load ollama

# Keep Ollama's model weights on scratch, NOT home -- home quota is 20GB and
# a single 8B model's weights alone can be several GB. Same lesson as the
# HuggingFace cache issue earlier.
export OLLAMA_MODELS=/lustre/nvwulf/scratch/cnwakeze/ollama_models
mkdir -p "$OLLAMA_MODELS"

# --- Start the Ollama server in the background ---
ollama serve > logs/ollama_server_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!

# Make sure the server (and any model still loaded in GPU memory) gets
# cleaned up when this script exits, whether it finishes, errors, or is
# killed for hitting the time limit.
cleanup() {
    echo "Stopping Ollama server (pid $OLLAMA_PID)..."
    kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the server to actually be ready before hitting it
echo "Waiting for Ollama server to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo "Ollama server is up."
        break
    fi
    sleep 2
done

MODEL="llama3.1:8b"
echo "Pulling model: $MODEL"
ollama pull "$MODEL"

echo "Starting pathology extraction..."
python scripts/pathology/pathology_extract.py \
    --input data/pathology-reports-brca.csv \
    --output results/pathology_json/pathology_extracted.jsonl \
    --model "$MODEL" \
    --host http://localhost:11434 \
    --log-every 25

echo "Done."
