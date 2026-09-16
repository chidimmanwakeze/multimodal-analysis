"""
generate_embeddings.py

Generates clinical embeddings for the FULL patient cohort (patients with a
non-null Subtype label) using a single HoneyBee embedding model.

Design note: this is meant to be run once PER MODEL (see submit_embeddings.slurm),
not looped over all 5 models in one process. Each HoneyBee model instance
gets loaded onto the GPU; running multiple models concurrently in one process
risks GPU memory contention / OOM on a shared node. Running one model per
SLURM array task instead gives each model its own GPU allocation and lets
the scheduler run them concurrently across separate jobs.

Usage:
    python generate_embeddings.py --model sentence-transformers \
        --tsv clinical-data.tsv \
        --outdir results/embeddings

Output (per model):
    results/embeddings/embeddings_<model>.npy       # (N_patients, D) float32
    results/embeddings/patient_ids_<model>.txt       # patient IDs, same row order
    results/embeddings/subtypes_<model>.txt          # subtype labels, same row order
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from honeybee import HoneyBee

# ---------------------------------------------------------------------------
# Leakage-prevention column lists (carried over from generate_embeddings.ipynb)
# ---------------------------------------------------------------------------

TARGET_COLUMNS = [
    "Tumor Type",
    "Subtype",
    "Cancer Type Detailed",
    "Cancer Type",
    "Cancer Type TCGA PanCanAtlas Cancer Type Acronym",
    "Oncotree Code",
    "Disease Free Status",
    "Disease Free (Months)",
    "Overall Survival Status",
    "Overall Survival (Months)",
    "Disease-specific Survival status",
    "Months of disease-specific survival",
    "Progress Free Survival (Months)",
    "Progression Free Status",
    "Person Neoplasm Cancer Status",
    "New Neoplasm Event Post Initial Therapy Indicator",
]

IDENTIFIER_COLUMNS = [
    "Study ID",
    "Patient ID",
    "Sample ID",
    "Other Patient ID",
    "gdc_case_id",
    "Form completion date",
    "Informed consent verified",
]

REDUNDANT_COLUMNS = [
    "ICD-10 Classification",
    "International Classification of Diseases for Oncology, Third Edition ICD-O-3 Histology Code",
    "International Classification of Diseases for Oncology, Third Edition ICD-O-3 Site Code",
]

ALL_EXCLUDED = set(TARGET_COLUMNS + IDENTIFIER_COLUMNS + REDUNDANT_COLUMNS)

VALID_MODELS = ["bioclinicalbert", "pubmedbert", "biobert", "scibert", "sentence-transformers"]


def row_to_lift_text(row: pd.Series, excluded: set) -> str:
    """
    Serialize one patient row to LIFT-format structured text.

    For each column not in `excluded`, produces: "The [column name] is [value]."
    Skips null / NaN / "NA" values.

    Dinh et al. (2022), LIFT, NeurIPS.
    """
    parts = []
    for col, val in row.items():
        if col in excluded:
            continue
        if pd.isna(val) or str(val).strip() in ("NA", "nan", ""):
            continue
        parts.append(f"The {col} is {val}.")
    return " ".join(parts)


def load_cohort(tsv_path: str):
    """Load clinical TSV. Patients with a missing Subtype are kept and
    labeled 'Unknown' rather than dropped, so they form their own cluster
    downstream instead of being excluded from the analysis."""
    df = pd.read_csv(tsv_path, sep="\t")
    n_missing = df["Subtype"].isna().sum()
    df["Subtype"] = df["Subtype"].fillna("Unknown")
    print(f"Loaded {len(df)} patients ({n_missing} had missing Subtype, "
          f"now labeled 'Unknown').")
    print("Subtype counts:")
    print(df["Subtype"].value_counts().to_string())
    return df


def build_narratives(df: pd.DataFrame):
    patient_ids = df["Patient ID"].tolist()
    subtypes = df["Subtype"].tolist()  # kept only for grouping downstream, NOT passed to the model
    narratives = [row_to_lift_text(row, ALL_EXCLUDED) for _, row in df.iterrows()]
    return patient_ids, subtypes, narratives


def generate(model_name: str, narratives: list, log_every: int = 50):
    hb = HoneyBee()
    embeddings = []
    t0 = time.time()
    for i, note in enumerate(narratives):
        e = hb.generate_embeddings(note, modality="clinical", model_name=model_name)
        embeddings.append(np.asarray(e).ravel())
        if (i + 1) % log_every == 0 or (i + 1) == len(narratives):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(narratives) - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"  [{model_name}] {i + 1}/{len(narratives)}  "
                  f"({rate:.1f} pts/sec, ETA {eta / 60:.1f} min)")
    return np.vstack(embeddings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=VALID_MODELS)
    ap.add_argument("--tsv", default="clinical-data.tsv")
    ap.add_argument("--outdir", default="results/embeddings")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_cohort(args.tsv)
    patient_ids, subtypes, narratives = build_narratives(df)

    print(f"\nGenerating embeddings with model='{args.model}' for {len(narratives)} patients...")
    mat = generate(args.model, narratives, log_every=args.log_every)
    print(f"Done. Embedding matrix shape: {mat.shape}")

    np.save(outdir / f"embeddings_{args.model}.npy", mat)
    with open(outdir / f"patient_ids_{args.model}.txt", "w") as f:
        f.write("\n".join(patient_ids))
    with open(outdir / f"subtypes_{args.model}.txt", "w") as f:
        f.write("\n".join(subtypes))

    print(f"Saved to {outdir}/ "
          f"(embeddings_{args.model}.npy, patient_ids_{args.model}.txt, subtypes_{args.model}.txt)")


if __name__ == "__main__":
    main()
