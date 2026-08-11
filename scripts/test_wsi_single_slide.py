"""
test_wsi_single_slide.py

Tests the PathologyProcessor pipeline on ONE real DX slide, using the
GRANULAR methods (not process_slide()) so we can inspect and cap the
patch count before the slow/expensive steps run at full scale --
process_slide() has no obvious max_patches control, and a real gigapixel
slide can yield tens of thousands of patches.

Usage:
    python test_wsi_single_slide.py
    python test_wsi_single_slide.py --max-patches 50   # fast sanity check
"""

import argparse
import time

import pandas as pd
from honeybee.processors import PathologyProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--max-patches", type=int, default=50,
                 help="cap the number of patches actually embedded, for a fast "
                      "sanity check. Set high (e.g. 100000) to effectively disable.")
ap.add_argument("--patch-size", type=int, default=256)
args = ap.parse_args()

manifest = pd.read_csv("results/wsi/wsi_manifest.csv")
row = manifest.iloc[0]
print(f"Testing on patient {row['Patient ID']}, slide: {row['slide_filename']}")
print(f"Slide path (read-only source): {row['slide_path']}")
print()

print("Initializing PathologyProcessor(model='uni')...")
processor = PathologyProcessor(model="uni")
print("Model info:", processor.get_model_info())
print()

print("Step 1: load_wsi...")
t0 = time.time()
wsi = processor.load_wsi(row["slide_path"], tile_size=args.patch_size)
print(f"  Done in {time.time() - t0:.1f}s")

print("Step 2: detect_tissue...")
t0 = time.time()
tissue_mask = processor.detect_tissue(wsi, method="otsu")
print(f"  Done in {time.time() - t0:.1f}s, tissue_mask shape={tissue_mask.shape}")

print("Step 3: extract_patches...")
t0 = time.time()
patches = processor.extract_patches(wsi, tissue_mask=tissue_mask, patch_size=args.patch_size)
n_total = len(patches)
print(f"  Done in {time.time() - t0:.1f}s -- extracted {n_total} tissue patches total")

n_use = min(n_total, args.max_patches)
print(f"\nUsing {n_use} of {n_total} patches for this test "
      f"(--max-patches {args.max_patches})")
patches_subset = patches[:n_use]

print("\nStep 4: generate_embeddings (this loads the UNI model -- requires "
      "HuggingFace access approval + authentication)...")
t0 = time.time()
embeddings = processor.generate_embeddings(patches_subset, progress=True)
print(f"  Done in {time.time() - t0:.1f}s -- embeddings shape={embeddings.shape}")

print("\nStep 5: aggregate_embeddings...")
slide_embedding = processor.aggregate_embeddings(embeddings, method="mean")
print(f"  Slide-level embedding shape: {slide_embedding.shape}")

print(f"\nSUCCESS -- pipeline works end-to-end on {n_use} patches from one slide.")
print(f"Full slide has {n_total} tissue patches total -- worth timing a larger "
      f"--max-patches run before committing to the full batch job.")
