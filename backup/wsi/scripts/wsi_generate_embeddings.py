"""
wsi_generate_embeddings.py

Generates one WSI embedding PER PATIENT from the DX slides listed in the
manifest (built by build_wsi_manifest.py). Uses the same granular 5-step
pipeline validated in test_wsi_single_slide.py (load_wsi -> detect_tissue
-> extract_patches -> generate_embeddings -> aggregate) rather than the
untested/uncapped process_slide() convenience method.

Patients with multiple DX slides get each slide embedded separately and
mean-pooled into one per-patient vector -- same aggregation pattern used
for multi-series radiology patients.

Safety: only ever reads slide files via the paths already recorded in
the manifest (which point into the shared, read-only source directory).
Never writes, moves, or deletes anything there. All output goes to
--outdir, inside your own project directory.

Usage:
    python wsi_generate_embeddings.py \
        --manifest results/wsi/wsi_manifest.csv \
        --model uni \
        --outdir results/wsi_embeddings \
        --max-patches-per-slide 500

Requires HF_TOKEN to be set in the environment (needed to download UNI's
gated weights) -- see submit_wsi_embeddings.slurm.
"""

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from honeybee.processors import PathologyProcessor

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def current_memory_mb() -> float:
    """Resident set size of this process, in MB. Standard library only --
    no extra dependency needed."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_done_patients(embeddings_path: Path, model: str) -> set:
    if not embeddings_path.exists():
        return set()
    ids_path = embeddings_path.parent / f"patient_ids_{model}.txt"
    if not ids_path.exists():
        return set()
    return set(ids_path.read_text().splitlines())


def embed_slide(processor, slide_path: str, patch_size: int, max_patches: int, verbose: bool = False):
    """Runs the validated granular pipeline on one slide. Returns
    (slide_embedding, n_total_patches, n_patches_used)."""
    t0 = time.time()
    wsi = processor.load_wsi(slide_path, tile_size=patch_size)
    t_load = time.time() - t0

    t0 = time.time()
    tissue_mask = processor.detect_tissue(wsi, method="otsu")
    t_tissue = time.time() - t0

    t0 = time.time()
    patches = processor.extract_patches(wsi, tissue_mask=tissue_mask, patch_size=patch_size)
    t_extract = time.time() - t0

    n_total = len(patches)
    if n_total == 0:
        raise RuntimeError("no tissue patches extracted")
    patches_used = patches[:max_patches]

    t0 = time.time()
    embeddings = processor.generate_embeddings(patches_used)
    t_embed = time.time() - t0

    t0 = time.time()
    slide_embedding = np.asarray(processor.aggregate_embeddings(embeddings, method="mean")).ravel()
    t_agg = time.time() - t0

    n_used = len(patches_used)

    if verbose:
        print(f"    [timing] load={t_load:.1f}s tissue={t_tissue:.1f}s extract={t_extract:.1f}s "
              f"embed={t_embed:.1f}s({n_used}patches) agg={t_agg:.1f}s "
              f"total={t_load+t_tissue+t_extract+t_embed+t_agg:.1f}s")

    del wsi, tissue_mask, patches, patches_used, embeddings
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return slide_embedding, n_total, n_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/wsi/wsi_manifest.csv")
    ap.add_argument("--model", default="uni")
    ap.add_argument("--outdir", default="results/wsi_embeddings")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--max-patches-per-slide", type=int, default=500,
                     help="cap patches embedded per slide, to bound time/GPU memory for "
                          "unusually large slides (default 500). The one slide tested so "
                          "far had only 158 total patches -- check a few more before "
                          "assuming this cap rarely triggers.")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--verbose", action="store_true",
                     help="print per-step timing (load/tissue/extract/embed/aggregate) "
                          "for every slide -- essential for diagnosing slow runs.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    print(f"Loaded manifest: {len(manifest)} slides, {manifest['Patient ID'].nunique()} unique patients")

    embeddings_path = outdir / f"embeddings_{args.model}.npy"
    done_patients = load_done_patients(embeddings_path, args.model)
    if done_patients:
        print(f"Resuming: {len(done_patients)} patients already done, skipping those.")

    print(f"Initializing PathologyProcessor(model='{args.model}')...")
    processor = PathologyProcessor(model=args.model)
    model_info = processor.get_model_info()
    print("Model info:", model_info)
    if model_info.get("device") in (None, "unknown", "cpu"):
        print(f"\n*** WARNING: device='{model_info.get('device')}' -- if this stays 'cpu' or "
              f"'unknown' during actual embedding, that would explain drastically slower-than-"
              f"expected runs. Check `nvidia-smi` during the first patient's processing. ***\n")
    print("Processor ready.\n")

    skipped_log_path = outdir / f"skipped_slides_{args.model}.jsonl"
    skipped_f = open(skipped_log_path, "a")

    all_patient_ids, all_subtypes, all_embeddings = [], [], []
    if done_patients and embeddings_path.exists():
        existing_mat = np.load(embeddings_path)
        all_patient_ids = (outdir / f"patient_ids_{args.model}.txt").read_text().splitlines()
        all_subtypes = (outdir / f"subtypes_{args.model}.txt").read_text().splitlines()
        all_embeddings = list(existing_mat)

    patient_groups = manifest.groupby("Patient ID")
    remaining_patients = [pid for pid in patient_groups.groups if pid not in done_patients]
    print(f"Processing {len(remaining_patients)} patients this run.\n")

    t0 = time.time()
    for i, pid in enumerate(remaining_patients):
        group = patient_groups.get_group(pid)
        subtype = group["Subtype"].iloc[0]

        slide_embeddings = []
        for _, row in group.iterrows():
            slide_path = row["slide_path"]
            if args.verbose:
                print(f"  [{pid}] processing slide {row['slide_filename']}...")
            try:
                emb, n_total, n_used = embed_slide(processor, slide_path, args.patch_size,
                                                     args.max_patches_per_slide, verbose=args.verbose)
                slide_embeddings.append(emb)
                if n_used < n_total:
                    skipped_f.write(json.dumps({
                        "Patient ID": pid, "slide": row["slide_filename"],
                        "note": f"used {n_used} of {n_total} patches (capped by --max-patches-per-slide)",
                    }) + "\n")
                    skipped_f.flush()
            except Exception as e:
                skipped_f.write(json.dumps({
                    "Patient ID": pid, "slide": row["slide_filename"],
                    "reason": f"error: {type(e).__name__}: {e}",
                }) + "\n")
                skipped_f.flush()

        if slide_embeddings:
            patient_embedding = np.mean(np.vstack(slide_embeddings), axis=0)
            all_patient_ids.append(pid)
            all_subtypes.append(subtype)
            all_embeddings.append(patient_embedding)

            mat = np.vstack(all_embeddings)
            np.save(embeddings_path, mat)
            (outdir / f"patient_ids_{args.model}.txt").write_text("\n".join(all_patient_ids))
            (outdir / f"subtypes_{args.model}.txt").write_text("\n".join(all_subtypes))
        else:
            skipped_f.write(json.dumps({
                "Patient ID": pid, "reason": "all slides failed for this patient",
            }) + "\n")
            skipped_f.flush()

        if (i + 1) % args.log_every == 0 or (i + 1) == len(remaining_patients):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining_patients) - (i + 1)) / rate / 60 if rate > 0 else float("nan")
            print(f"  {i + 1}/{len(remaining_patients)} patients  "
                  f"({rate:.3f} patients/sec, ETA {eta:.1f} min, "
                  f"RSS={current_memory_mb():.0f}MB)")

    skipped_f.close()
    print(f"\nDone. {len(all_patient_ids)} patients with WSI embeddings.")
    print(f"Saved to {outdir}/")
    print(f"Skipped/capped slides logged to {skipped_log_path}")


if __name__ == "__main__":
    main()
