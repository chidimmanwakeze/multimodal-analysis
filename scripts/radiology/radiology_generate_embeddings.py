"""
radiology_generate_embeddings.py

Generates one radiology embedding PER PATIENT by:
  1. Filtering imaging-metadata.csv to the BRCA cohort + successful downloads
  2. Excluding series above --max-filesize (default 300MB) -- large outlier
     series (e.g. 1.36GB, 1500-slice non-uniform volumes) can OOM-kill the
     whole process even with 64GB allocated, and an OOM kill is a SIGKILL
     from the kernel that a Python try/except cannot catch or recover from.
     Pre-filtering by size avoids this rather than hoping to catch it.
  3. Running each remaining series through RadiologyProcessor
     (load_dicom -> preprocess -> generate_embeddings)
  4. Mean-pooling all of a patient's successful series embeddings into a
     single per-patient vector (patients have ~13 series each on average)

Skipped/failed series are logged, not silently dropped, so you can see
exactly what was excluded and why.

Usage:
    python radiology_generate_embeddings.py \
        --imaging-metadata imaging-metadata.csv \
        --clinical clinical-data.tsv \
        --model radimagenet-densenet121 \
        --outdir results/radiology_embeddings \
        --max-filesize 300000000

Output (matches the naming convention cluster_analysis.py expects):
    <outdir>/embeddings_<model>.npy
    <outdir>/patient_ids_<model>.txt
    <outdir>/subtypes_<model>.txt
    <outdir>/skipped_series_<model>.jsonl   (size-excluded or failed series)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from honeybee.processors import RadiologyProcessor


def load_candidates(imaging_metadata_path: str, clinical_path: str) -> pd.DataFrame:
    img = pd.read_csv(imaging_metadata_path)
    clin = pd.read_csv(clinical_path, sep="\t")
    brca_ids = set(clin["Patient ID"])

    candidates = img[
        (img["PatientID"].isin(brca_ids)) & (img["completion_status"] == "success")
    ].reset_index(drop=True)
    return candidates, clin


def load_done_patients(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    ids_path = output_path.parent / output_path.name.replace("embeddings_", "patient_ids_").replace(".npy", ".txt")
    if not ids_path.exists():
        return set()
    with open(ids_path) as f:
        return set(f.read().splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imaging-metadata", default="imaging-metadata.csv")
    ap.add_argument("--clinical", default="clinical-data.tsv")
    ap.add_argument("--model", default="radimagenet-densenet121")
    ap.add_argument("--outdir", default="results/radiology_embeddings")
    ap.add_argument("--max-filesize", type=int, default=300_000_000,
                     help="skip series larger than this many bytes (default 300MB) "
                          "to avoid OOM-killing the whole job on outlier volumes")
    ap.add_argument("--max-series-per-patient", type=int, default=None,
                     help="optional cap on series processed per patient, for speed. "
                          "None = process all of a patient's series.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    candidates, clin = load_candidates(args.imaging_metadata, args.clinical)
    subtype_map = dict(zip(clin["Patient ID"], clin["Subtype"].fillna("Unknown")))

    n_total_series = len(candidates)
    oversized = candidates[candidates["FileSize"] > args.max_filesize]
    candidates = candidates[candidates["FileSize"] <= args.max_filesize].reset_index(drop=True)

    print(f"Loaded {n_total_series} candidate series across "
          f"{candidates['PatientID'].nunique() + oversized['PatientID'].nunique()} patients.")
    print(f"Excluding {len(oversized)} series over {args.max_filesize:,} bytes "
          f"(across {oversized['PatientID'].nunique()} patients, though most of those "
          f"patients likely have other smaller series still included).")
    print(f"{len(candidates)} series remain for processing.\n")

    embeddings_path = outdir / f"embeddings_{args.model}.npy"
    done_patients = load_done_patients(embeddings_path)
    if done_patients:
        print(f"Resuming: {len(done_patients)} patients already done, skipping those.")

    print(f"Initializing RadiologyProcessor(model='{args.model}')...")
    processor = RadiologyProcessor(model=args.model)
    print("Processor ready.\n")

    skipped_log_path = outdir / f"skipped_series_{args.model}.jsonl"
    skipped_f = open(skipped_log_path, "a")

    # Log oversized series up front (these were never attempted)
    for _, row in oversized.iterrows():
        skipped_f.write(json.dumps({
            "PatientID": row["PatientID"], "reason": "oversized",
            "FileSize": int(row["FileSize"]), "path": row["S5cmdManifestPath"],
        }) + "\n")
    skipped_f.flush()

    all_patient_ids, all_subtypes, all_embeddings = [], [], []

    # Load any existing embeddings if resuming
    if done_patients and embeddings_path.exists():
        existing_mat = np.load(embeddings_path)
        existing_ids = (outdir / f"patient_ids_{args.model}.txt").read_text().splitlines()
        existing_subtypes = (outdir / f"subtypes_{args.model}.txt").read_text().splitlines()
        all_patient_ids.extend(existing_ids)
        all_subtypes.extend(existing_subtypes)
        all_embeddings.extend(list(existing_mat))

    patient_groups = candidates.groupby("PatientID")
    remaining_patients = [pid for pid in patient_groups.groups if pid not in done_patients]
    print(f"Processing {len(remaining_patients)} patients this run.\n")

    t0 = time.time()
    for i, pid in enumerate(remaining_patients):
        group = patient_groups.get_group(pid)
        if args.max_series_per_patient:
            group = group.head(args.max_series_per_patient)

        series_embeddings = []
        for _, row in group.iterrows():
            series_path = row["S5cmdManifestPath"]
            try:
                image, metadata = processor.load_dicom(series_path)
                preprocessed = processor.preprocess(image, metadata)
                emb = processor.generate_embeddings(preprocessed)
                series_embeddings.append(np.asarray(emb).ravel())
            except Exception as e:
                skipped_f.write(json.dumps({
                    "PatientID": pid, "reason": f"error: {type(e).__name__}: {e}",
                    "path": series_path,
                }) + "\n")
                skipped_f.flush()

        if series_embeddings:
            patient_embedding = np.mean(np.vstack(series_embeddings), axis=0)
            all_patient_ids.append(pid)
            all_subtypes.append(subtype_map.get(pid, "Unknown"))
            all_embeddings.append(patient_embedding)

            # Save incrementally after every patient so a killed job loses minimal progress
            mat = np.vstack(all_embeddings)
            np.save(embeddings_path, mat)
            (outdir / f"patient_ids_{args.model}.txt").write_text("\n".join(all_patient_ids))
            (outdir / f"subtypes_{args.model}.txt").write_text("\n".join(all_subtypes))
        else:
            skipped_f.write(json.dumps({
                "PatientID": pid, "reason": "all series failed or none available",
            }) + "\n")
            skipped_f.flush()

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(remaining_patients) - (i + 1)) / rate / 60 if rate > 0 else float("nan")
        print(f"  {i + 1}/{len(remaining_patients)} patients  "
              f"({len(series_embeddings)} series used for {pid})  "
              f"({rate:.2f} patients/sec, ETA {eta:.1f} min)")

    skipped_f.close()
    print(f"\nDone. {len(all_patient_ids)} patients with radiology embeddings.")
    print(f"Saved to {outdir}/")
    print(f"Skipped/failed series logged to {skipped_log_path}")


if __name__ == "__main__":
    main()
