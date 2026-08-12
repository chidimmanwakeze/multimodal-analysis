"""
Merge clinical, pathology, and imaging metadata on Patient ID.
No columns are dropped from any modality.

Outputs:
    patient_manifest_full.csv     - all clinical patients regardless of modality
    patient_manifest_complete.csv - patients with ALL three modalities

Run on cluster:
    cd ~/multimodal-analysis
    module load python39
    python merge_data.py
"""

import pandas as pd

# ---- Load each modality ----

# Clinical — tab separated, join key: "Patient ID"
clinical = pd.read_csv("clinical-data.tsv", sep="\t")
clinical = clinical.rename(columns={"Patient ID": "patient_id"})
print(f"Clinical:  {len(clinical)} patients, {len(clinical.columns)} columns")

# Pathology — comma separated, join key: patient_filename (extract ID before period)
pathology = pd.read_csv("pathology-reports.csv")
pathology["patient_id"] = pathology["patient_filename"].str.split(".").str[0]
pathology = pathology.drop(columns=["patient_filename"])
pathology = pathology.drop_duplicates(subset="patient_id")
print(f"Pathology: {len(pathology)} unique patients, {len(pathology.columns)} columns")

# Imaging — comma separated, join key: PatientID
# Keep one row per patient but retain all columns
imaging = pd.read_csv("imaging-metadata.csv")
imaging = imaging.rename(columns={"PatientID": "patient_id"})
imaging_patients = imaging.drop_duplicates(subset="patient_id")
print(f"Imaging:   {len(imaging_patients)} unique patients, {len(imaging_patients.columns)} columns")

# ---- Merge ----

# Start with clinical as the base
merged = clinical.copy()

# Add modality indicators
merged["has_pathology"] = merged["patient_id"].isin(pathology["patient_id"])
merged["has_imaging"] = merged["patient_id"].isin(imaging_patients["patient_id"])

# Left join all pathology columns
merged = merged.merge(
    pathology,
    on="patient_id",
    how="left"
)

# Left join all imaging columns
merged = merged.merge(
    imaging_patients,
    on="patient_id",
    how="left"
)

# ---- Summary ----
all_three = merged[merged["has_pathology"] & merged["has_imaging"]]

print(f"\nTotal clinical patients:        {len(merged)}")
print(f"With pathology:                 {merged['has_pathology'].sum()}")
print(f"With imaging:                   {merged['has_imaging'].sum()}")
print(f"With ALL three modalities:      {len(all_three)}")
print(f"Total columns in merged output: {len(merged.columns)}")

# ---- Save outputs ----
merged.to_csv("patient_manifest_full.csv", index=False)
all_three.to_csv("patient_manifest_complete.csv", index=False)

print("\n--- Saved ---")
print(f"  patient_manifest_full.csv     ({len(merged)} patients, {len(merged.columns)} columns)")
print(f"  patient_manifest_complete.csv ({len(all_three)} patients, {len(all_three.columns)} columns — all 3 modalities)")
print("\nDone.")

