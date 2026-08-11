"""
build_wsi_manifest.py

Scans the shared, READ-ONLY TCGA BRCA WSI directory for diagnostic (DX)
slides, matches them to your BRCA clinical cohort by patient barcode, and
writes a manifest CSV into YOUR OWN project directory.

Safety: this script only ever LISTS filenames in --wsi-dir (Path.glob) --
it never opens those files for writing, never moves them, never deletes
them. The only thing written to disk is the manifest CSV, and that goes
to --out, which defaults to your own results/ directory.

Usage:
    python build_wsi_manifest.py \
        --wsi-dir /lustre/nvwulf/projects/KurcGroup-nvwulf/tcga_all/brca \
        --clinical clinical-data.tsv \
        --out results/wsi/wsi_manifest.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

PATIENT_ID_PATTERN = re.compile(r"^(TCGA-\w{2}-\w{4})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi-dir", required=True,
                     help="READ-ONLY source directory containing .svs slide files. "
                          "This script never writes here.")
    ap.add_argument("--clinical", default="clinical-data.tsv")
    ap.add_argument("--out", default="results/wsi/wsi_manifest.csv")
    ap.add_argument("--slide-type", default="DX",
                     help="filter to slide filenames containing this code "
                          "(default DX = diagnostic slides)")
    args = ap.parse_args()

    wsi_dir = Path(args.wsi_dir)
    if not wsi_dir.exists():
        raise SystemExit(f"WSI directory not found: {wsi_dir}")

    # Read-only: listing filenames only, nothing is opened or written here.
    all_slides = sorted(wsi_dir.glob("*.svs"))
    print(f"Found {len(all_slides)} total .svs files in {wsi_dir} (read-only, not modified)")

    dx_slides = [f for f in all_slides if args.slide_type in f.name]
    print(f"{len(dx_slides)} filenames contain '{args.slide_type}' (diagnostic slides)")

    clin = pd.read_csv(args.clinical, sep="\t")
    cohort_ids = set(clin["Patient ID"])
    clin_subtypes = dict(zip(clin["Patient ID"], clin["Subtype"].fillna("Unknown")))

    rows = []
    unmatched_pattern = 0
    for f in dx_slides:
        m = PATIENT_ID_PATTERN.match(f.name)
        if not m:
            unmatched_pattern += 1
            continue
        patient_id = m.group(1)
        if patient_id not in cohort_ids:
            continue
        rows.append({
            "Patient ID": patient_id,
            "slide_filename": f.name,
            "slide_path": str(f),  # absolute path INTO the read-only source dir
            "Subtype": clin_subtypes[patient_id],
        })

    manifest = pd.DataFrame(rows)
    print(f"\n{len(manifest)} DX slides matched to your BRCA cohort "
          f"({manifest['Patient ID'].nunique() if len(manifest) else 0} unique patients)")
    if unmatched_pattern:
        print(f"({unmatched_pattern} slide filenames did not match the expected "
              f"TCGA-XX-XXXX barcode pattern and were skipped)")

    if len(manifest):
        dupe_counts = manifest["Patient ID"].value_counts()
        multi = int((dupe_counts > 1).sum())
        if multi:
            print(f"{multi} patients have more than one DX slide -- will need "
                  f"patient-level aggregation (e.g. mean-pooling), same pattern used for radiology.")

        print(f"\nSubtype distribution in matched manifest:")
        for subtype, count in manifest.drop_duplicates("Patient ID")["Subtype"].value_counts().items():
            print(f"  {subtype:<15s} {count:>4d}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # only creates directories in YOUR project space
    manifest.to_csv(out_path, index=False)
    print(f"\nSaved manifest to {out_path} (in your own project directory)")
    print(f"Source WSI directory was only read from, never modified: {wsi_dir}")


if __name__ == "__main__":
    main()
