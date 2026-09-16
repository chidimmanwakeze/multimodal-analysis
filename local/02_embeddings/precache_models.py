"""
precache_models.py

Downloads the 3 models that keep hitting HuggingFace's rate limit,
ONE AT A TIME with a pause between each, so they end up fully cached
locally before the real embedding array job ever runs.
"""
import os
import time

os.environ.setdefault("HF_HOME", "/lustre/nvwulf/scratch/cnwakeze/hf_cache")

from transformers import AutoModel, AutoTokenizer

MODELS = [
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",  # pubmedbert
    "dmis-lab/biobert-v1.1",                                          # biobert
    "sentence-transformers/all-MiniLM-L6-v2",                         # sentence-transformers
]

for i, model_id in enumerate(MODELS):
    print(f"\n[{i+1}/{len(MODELS)}] Downloading {model_id} ...")
    try:
        AutoTokenizer.from_pretrained(model_id)
        AutoModel.from_pretrained(model_id)
        print(f"  OK - cached.")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Waiting 60s before continuing to the next model...")
        time.sleep(60)
        continue

    if i < len(MODELS) - 1:
        print("  Pausing 15s before next model to stay well under the rate limit...")
        time.sleep(15)

print("\nDone pre-caching.")
