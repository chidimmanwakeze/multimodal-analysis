"""
test_radiology_single_series.py

Tests the RadiologyProcessor API (load_dicom -> preprocess -> generate_embeddings)
on ONE real DICOM series before committing to a full batch pipeline across all
1,877 series. Run this interactively on a GPU debug node first.

Usage:
    python test_radiology_single_series.py
"""

import pandas as pd
from honeybee.processors import RadiologyProcessor

img = pd.read_csv("imaging-metadata.csv")
clin = pd.read_csv("clinical-data.tsv", sep="\t")
brca_ids = set(clin["Patient ID"])

# Pick one series from a patient that's actually in our BRCA cohort, with a
# successful download, and a reasonably large file size (avoid tiny scout series
# for this first test so we're looking at a real diagnostic image)
candidates = img[
    (img["PatientID"].isin(brca_ids))
    & (img["completion_status"] == "success")
].sort_values("FileSize")

print("FileSize distribution across candidate series (bytes):")
print(candidates["FileSize"].describe())
print()

median_idx = len(candidates) // 2
row = candidates.iloc[median_idx]
series_path = row["S5cmdManifestPath"]
print(f"Testing on patient {row['PatientID']}, series path:")
print(f"  {series_path}")
print(f"  FileSize: {row['FileSize']:,} bytes (median-ish)")
print()

print("Initializing RadiologyProcessor(model='radimagenet-densenet121')...")
processor = RadiologyProcessor(model="radimagenet-densenet121")
print("Processor initialized.\n")

print("Step 1: load_dicom...")
image, metadata = processor.load_dicom(series_path)
print(f"  image type: {type(image)}")
print(f"  image shape (if array-like): {getattr(image, 'shape', 'n/a')}")
print(f"  metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else type(metadata)}")
print()

print("Step 2: preprocess...")
preprocessed = processor.preprocess(image, metadata)
print(f"  preprocessed type: {type(preprocessed)}")
print(f"  preprocessed shape (if array-like): {getattr(preprocessed, 'shape', 'n/a')}")
print()

print("Step 3: generate_embeddings...")
embedding = processor.generate_embeddings(preprocessed)
print(f"  embedding type: {type(embedding)}")
import numpy as np
arr = np.asarray(embedding)
print(f"  embedding shape: {arr.shape}")
print(f"  embedding dtype: {arr.dtype}")
print(f"  first 5 values: {arr.ravel()[:5]}")
print()
print("SUCCESS -- pipeline works end-to-end on one series.")
