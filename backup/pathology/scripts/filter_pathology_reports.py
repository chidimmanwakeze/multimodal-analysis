"""
filter_pathology_reports.py

pathology-reports.csv contains reports across MULTIPLE TCGA cancer studies
(confirmed: the first row is a renal cell carcinoma report). This script
filters it down to only the patients present in your BRCA clinical cohort
(clinical-data.tsv), matched on the TCGA-XX-XXXX patient barcode prefix
that appears before the UUID suffix in `patient_filename`.

Usage:
    python filter_pathology_reports.py \
        --pathology pathology-reports.csv \
        --clinical clinical-data.tsv \
        --out pathology-reports-brca.csv
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathology", default="pathology-reports.csv")
    ap.add_argument("--clinical", default="clinical-data.tsv")
    ap.add_argument("--out", default="pathology-reports-brca.csv")
    args = ap.parse_args()

    path_df = pd.read_csv(args.pathology)
    clin_df = pd.read_csv(args.clinical, sep="\t")

    print(f"Loaded {len(path_df)} total pathology reports (all cancer types).")
    print(f"Loaded {len(clin_df)} BRCA patients from clinical cohort.")

    # Extract the TCGA-XX-XXXX patient barcode prefix (matches clinical "Patient ID" format)
    path_df["Patient ID"] = path_df["patient_filename"].str.extract(r"^(TCGA-\w{2}-\w{4})")

    brca_ids = set(clin_df["Patient ID"])
    filtered = path_df[path_df["Patient ID"].isin(brca_ids)].reset_index(drop=True)

    missing = brca_ids - set(path_df["Patient ID"])

    print(f"\nMatched {len(filtered)} pathology reports to the BRCA cohort "
          f"({len(path_df) - len(filtered)} non-BRCA reports excluded).")
    print(f"{len(missing)} BRCA patients have no pathology report in this file.")

    filtered.to_csv(args.out, index=False)
    print(f"\nSaved filtered set to {args.out}")

    # Save the list of missing patient IDs for reference
    missing_path = args.out.replace(".csv", "_missing_patient_ids.txt")
    with open(missing_path, "w") as f:
        f.write("\n".join(sorted(missing)))
    print(f"Saved list of {len(missing)} patients with no pathology report to {missing_path}")


if __name__ == "__main__":
    main()
