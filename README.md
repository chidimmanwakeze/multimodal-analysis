# TCGA-BRCA Multimodal Analysis Pipeline

A five-stage pipeline that extracts, embeds, and merges clinical, pathology,
and radiology data for TCGA breast cancer (BRCA) patients, then trains a
classifier on the combined multimodal representation.

## Pipeline stages

```
01_extraction/    raw data --> structured/manifest data
02_embeddings/    structured data --> numeric embeddings (per modality)
03_integration/   merge embeddings across modalities into one dataset
04_modeling/      train + select the classifier
05_analysis/      post-hoc evaluation, clustering, comparisons
```

Each stage folder holds both its Python script(s) and the matching SLURM
launcher(s) together -- e.g. `01_extraction/pathology_extract.py` and
`01_extraction/submit_pathology_extraction.slurm` sit side by side.

## Modalities

- **Clinical**: tabular clinical data, embedded with one of 5 language
  models (bioclinicalbert, pubmedbert, biobert, scibert, sentence-transformers).
- **Pathology**: free-text pathology reports. `pathology_extract.py` uses a
  local LLM (via Ollama) to pull structured fields (tumor size, grade,
  lymph node status, staging, etc.) out of the free text; those structured
  fields are then embedded the same way as clinical data.
- **Radiology**: DICOM imaging series, embedded with a RadImageNet
  DenseNet121 model.
- **Whole-slide imaging (WSI)**: present in the pipeline (`wsi_build_manifest.py`,
  `wsi_generate_embeddings.py`) but **not yet run or verified** as of this
  push. Uses the `UNI` foundation model, which requires separate gated
  access approval on HuggingFace.

`build_multimodal_dataset.py` (03_integration) currently merges only
clinical + pathology + radiology; WSI is deliberately excluded until it's
been run and verified.

## Setup

Raw data is **not included in this repository** -- it lives only on the
cluster filesystem. Before running anything, create this structure yourself
and populate it with your own data:

```
00_raw_data/
  clinical/clinical_data.tsv
  pathology/pathology-reports-brca.csv
  radiology/radiology_metadata.csv, radiology_dicom/
```

A conda environment named `honeybee` with `torch`, `transformers`,
`huggingface_hub`, `pandas`, and `scikit-learn` is expected. Some models
(scibert, pubmedbert, etc.) are pulled from HuggingFace at runtime --
authenticate first with a free HuggingFace account and access token to
avoid anonymous rate limits:

```bash
python -c "from huggingface_hub import login; login()"
```

Pathology extraction additionally requires Ollama (`module load ollama` on
this cluster) and a local `llama3.1:8b` model pull.

## Running a stage

Submit each SLURM script **from within its own stage folder** -- e.g.:

```bash
cd 01_extraction
sbatch submit_pathology_extraction.slurm
```

Each script resolves its own working directory via `$SLURM_SUBMIT_DIR`, so
this works regardless of your absolute path, as long as you submit from the
correct folder.

## Known issues / not yet verified

- `select_best_model.py`, `cluster_analysis.py`, `centroid_summary.py`
  (05_analysis and part of 04_modeling) have not been executed or reviewed
  in depth -- interfaces are documented via `--help` but behavior is
  unconfirmed.
- WSI stage (extraction + embeddings) has never been run.
- `04_modeling/_archive/` holds a superseded classifier version (no early
  stopping) kept for reference; `train_classifier.py` is the current,
  tested version (adds early stopping with best-checkpoint restoration).
- Final merged multimodal dataset (clinical + pathology + radiology) is
  currently small (~135 patients) because radiology is the limiting
  modality; some subtype classes have very few samples (single digits).
