"""
build_multimodal_dataset.py

Aligns any SUBSET of the three modality embeddings (clinical, pathology,
radiology) by Patient ID, keeping only patients that have ALL SELECTED
modalities present. Defaults to all three; pass --modalities to use a
subset, e.g. --modalities clinical,pathology to skip radiology entirely.

Usage:
    # All three (original behavior)
    python build_multimodal_dataset.py --out results/multimodal_dataset.npz

    # Clinical + pathology only -- no radiology bottleneck
    python build_multimodal_dataset.py \
        --modalities clinical,pathology \
        --out results/clinical_pathology_dataset.npz

Output: a single .npz file containing the aligned arrays for whichever
modalities were selected, ready for train_classifier.py. Also prints the
patient-count funnel so the sample-size reality is visible up front.
"""

import argparse
from pathlib import Path

import numpy as np

MODALITY_DEFAULTS = {
    "clinical": {"dir": "results/embeddings", "model": "sentence-transformers"},
    "pathology": {"dir": "results/pathology_embeddings/json", "model": "sentence-transformers"},
    "radiology": {"dir": "results/radiology_embeddings", "model": "radimagenet-densenet121"},
}


def load_modality(embeddings_dir: str, model_name: str):
    d = Path(embeddings_dir)
    mat = np.load(d / f"embeddings_{model_name}.npy")
    patient_ids = (d / f"patient_ids_{model_name}.txt").read_text().splitlines()
    subtypes = (d / f"subtypes_{model_name}.txt").read_text().splitlines()
    assert mat.shape[0] == len(patient_ids) == len(subtypes)
    return mat, patient_ids, subtypes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", default="clinical,pathology,radiology",
                     help="comma-separated subset of: clinical, pathology, radiology")
    ap.add_argument("--clinical-dir", default=MODALITY_DEFAULTS["clinical"]["dir"])
    ap.add_argument("--clinical-model", default=MODALITY_DEFAULTS["clinical"]["model"])
    ap.add_argument("--pathology-dir", default=MODALITY_DEFAULTS["pathology"]["dir"])
    ap.add_argument("--pathology-model", default=MODALITY_DEFAULTS["pathology"]["model"])
    ap.add_argument("--radiology-dir", default=MODALITY_DEFAULTS["radiology"]["dir"])
    ap.add_argument("--radiology-model", default=MODALITY_DEFAULTS["radiology"]["model"])
    ap.add_argument("--out", default="results/multimodal_dataset.npz")
    args = ap.parse_args()

    selected = [m.strip() for m in args.modalities.split(",") if m.strip()]
    for m in selected:
        if m not in MODALITY_DEFAULTS:
            raise SystemExit(f"Unknown modality '{m}'. Choose from: clinical, pathology, radiology")
    if len(selected) < 2:
        raise SystemExit("Need at least 2 modalities to build a multimodal dataset.")

    dirs = {"clinical": args.clinical_dir, "pathology": args.pathology_dir, "radiology": args.radiology_dir}
    models = {"clinical": args.clinical_model, "pathology": args.pathology_model, "radiology": args.radiology_model}

    print(f"Building dataset for modalities: {selected}\n")
    loaded = {}
    for m in selected:
        mat, ids, subtypes = load_modality(dirs[m], models[m])
        lookup = {pid: (i, s) for i, (pid, s) in enumerate(zip(ids, subtypes))}
        loaded[m] = {"mat": mat, "lookup": lookup}
        print(f"  {m.capitalize():<10s}: {len(ids)} patients, {mat.shape[1]}-dim")

    print("\nPatient overlap funnel:")
    common = set(loaded[selected[0]]["lookup"])
    print(f"  {selected[0]:<10s}: {len(common)}")
    for m in selected[1:]:
        common &= set(loaded[m]["lookup"])
        print(f"  + {m:<8s}: {len(common)}")
    common_ids = sorted(common)

    ref = selected[0]
    mismatches = [pid for pid in common_ids
                  if not all(loaded[m]["lookup"][pid][1] == loaded[ref]["lookup"][pid][1] for m in selected)]
    if mismatches:
        print(f"\nWARNING: {len(mismatches)} patients have inconsistent Subtype labels "
              f"across modalities (using the {ref} label as source of truth). "
              f"Examples: {mismatches[:5]}")

    labels = [loaded[ref]["lookup"][pid][1] for pid in common_ids]
    label_names = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    y = np.array([label_to_idx[l] for l in labels])

    print(f"\nClass distribution in final dataset:")
    for label in label_names:
        print(f"  {label:<15s} {labels.count(label):>4d}")

    out_dict = {"y": y, "patient_ids": np.array(common_ids), "label_names": np.array(label_names),
                "modalities": np.array(selected)}
    print()
    for m in selected:
        X = np.vstack([loaded[m]["mat"][loaded[m]["lookup"][pid][0]] for pid in common_ids])
        out_dict[f"X_{m}"] = X
        print(f"  X_{m}: {X.shape}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out_dict)
    print(f"\nSaved aligned multimodal dataset to {args.out}")


if __name__ == "__main__":
    main()
