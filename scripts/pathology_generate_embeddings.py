"""
pathology_generate_embeddings.py

Takes the extracted pathology JSON (pathology_extracted.jsonl, produced by
pathology_extract.py) and generates embeddings using ONE of two text
serialization strategies, per your PI's request to compare which works
better:

  --serialization json
      Embeds the raw JSON object as text, e.g.:
      {"laterality": "left", "tumor_grade": 3, "tumor_size_cm": 2.5, ...}

  --serialization template
      Converts the JSON into a natural-language narrative, same LIFT-style
      pattern used for the clinical TSV in generate_embeddings.py, e.g.:
      "The laterality is left. The tumor grade is 3. The tumor size cm is 2.5."
      (matches your PI's example: "This patient is 55 (age = 55) years old"
      -- i.e. replace the column name with the value in a sentence)

Meant to be run once per (serialization, model) combination -- see
submit_pathology_embeddings.slurm for the array job that covers all
2 x 5 = 10 combinations.

Usage:
    python pathology_generate_embeddings.py \
        --extracted results/pathology_json/pathology_extracted.jsonl \
        --clinical clinical-data.tsv \
        --serialization template \
        --model scibert \
        --outdir results/pathology_embeddings/template

Output (matches the naming convention cluster_analysis.py already expects,
so it can be reused unmodified on either output directory):
    <outdir>/embeddings_<model>.npy
    <outdir>/patient_ids_<model>.txt
    <outdir>/subtypes_<model>.txt
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from honeybee import HoneyBee

VALID_MODELS = ["bioclinicalbert", "pubmedbert", "biobert", "scibert", "sentence-transformers"]

# Fields written by pathology_extract.py that are metadata, not clinical
# content, and should never be embedded.
NON_CONTENT_KEYS = {"Patient ID", "patient_filename", "_type_warnings"}


def humanize_key(key: str) -> str:
    """tumor_size_cm -> Tumor Size Cm"""
    return " ".join(w.capitalize() for w in key.split("_"))


def humanize_value(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    return value


def to_json_text(record: dict) -> str:
    content = {k: v for k, v in record.items() if k not in NON_CONTENT_KEYS}
    return json.dumps(content)


def to_template_text(record: dict) -> str:
    content = {k: v for k, v in record.items() if k not in NON_CONTENT_KEYS}
    parts = []
    for key, value in content.items():
        if value is None:
            continue
        parts.append(f"The {humanize_key(key)} is {humanize_value(value)}.")
    return " ".join(parts)


def load_extracted(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_subtypes(clinical_path: str) -> dict:
    df = pd.read_csv(clinical_path, sep="\t")
    df["Subtype"] = df["Subtype"].fillna("Unknown")
    return dict(zip(df["Patient ID"], df["Subtype"]))


def generate(model_name: str, texts: list, log_every: int = 50):
    hb = HoneyBee()
    embeddings = []
    t0 = time.time()
    for i, text in enumerate(texts):
        e = hb.generate_embeddings(text, modality="clinical", model_name=model_name)
        embeddings.append(np.asarray(e).ravel())
        if (i + 1) % log_every == 0 or (i + 1) == len(texts):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(texts) - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"  [{model_name}] {i + 1}/{len(texts)}  "
                  f"({rate:.1f} pts/sec, ETA {eta / 60:.1f} min)")
    return np.vstack(embeddings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", default="results/pathology_json/pathology_extracted.jsonl")
    ap.add_argument("--clinical", default="clinical-data.tsv")
    ap.add_argument("--serialization", required=True, choices=["json", "template"])
    ap.add_argument("--model", required=True, choices=VALID_MODELS)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_extracted(args.extracted)
    subtype_map = load_subtypes(args.clinical)
    print(f"Loaded {len(records)} extracted pathology records.")

    serializer = to_json_text if args.serialization == "json" else to_template_text

    patient_ids, subtypes, texts = [], [], []
    for rec in records:
        pid = rec["Patient ID"]
        patient_ids.append(pid)
        subtypes.append(subtype_map.get(pid, "Unknown"))  # in case of any mismatch
        texts.append(serializer(rec))

    print(f"\nExample ({args.serialization}) text for {patient_ids[0]}:")
    print(f"  {texts[0][:300]}")

    print(f"\nGenerating embeddings with model='{args.model}', "
          f"serialization='{args.serialization}' for {len(texts)} patients...")
    mat = generate(args.model, texts, log_every=args.log_every)
    print(f"Done. Embedding matrix shape: {mat.shape}")

    np.save(outdir / f"embeddings_{args.model}.npy", mat)
    with open(outdir / f"patient_ids_{args.model}.txt", "w") as f:
        f.write("\n".join(patient_ids))
    with open(outdir / f"subtypes_{args.model}.txt", "w") as f:
        f.write("\n".join(subtypes))

    print(f"Saved to {outdir}/")


if __name__ == "__main__":
    main()
