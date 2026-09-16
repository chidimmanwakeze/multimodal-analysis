#!/bin/bash
# ============================================================================
# Fix: make the 5 stages top-level folders in the repo, matching how
# multimodal-analysis-local is actually laid out -- script + slurm files
# together in one folder per stage, not split into separate scripts/ and
# slurm/ trees.
#
# Does NOT commit or push automatically -- review before doing so.
# Run from inside the multimodal-analysis git repo.
# ============================================================================
set -euo pipefail

cd /lustre/nvwulf/projects/KurcGroup-nvwulf/cnwakeze/multimodal-analysis

echo "############################################################"
echo "# 1. Move each stage's scripts + slurm files up to a single"
echo "#    top-level folder per stage"
echo "############################################################"

for stage in 01_extraction 02_embeddings 03_integration 04_modeling 05_analysis; do
    mkdir -p "$stage"

    if [ -d "scripts/$stage" ]; then
        for f in "scripts/$stage"/* ; do
            [ -e "$f" ] || continue
            git mv "$f" "$stage/$(basename "$f")"
        done
    fi

    if [ -d "slurm/$stage" ]; then
        for f in "slurm/$stage"/* ; do
            [ -e "$f" ] || continue
            git mv "$f" "$stage/$(basename "$f")"
        done
    fi
done

echo
echo "############################################################"
echo "# 2. Clean up now-empty scripts/ and slurm/ trees"
echo "############################################################"

find scripts slurm -type d -empty -delete 2>/dev/null || true
rmdir scripts slurm 2>/dev/null || true

echo
echo "############################################################"
echo "# 3. Update README to reflect the new top-level layout"
echo "############################################################"

python3 - << 'PYEOF'
import re

with open("README.md") as f:
    content = f.read()

content = content.replace(
    """```
01_extraction/   raw data --> structured/manifest data
02_embeddings/    structured data --> numeric embeddings (per modality)
03_integration/   merge embeddings across modalities into one dataset
04_modeling/      train + select the classifier
05_analysis/      post-hoc evaluation, clustering, comparisons
```

Each stage's SLURM launcher lives in `slurm/<stage>/`, mirroring
`scripts/<stage>/`.""",
    """```
01_extraction/    raw data --> structured/manifest data
02_embeddings/    structured data --> numeric embeddings (per modality)
03_integration/   merge embeddings across modalities into one dataset
04_modeling/      train + select the classifier
05_analysis/      post-hoc evaluation, clustering, comparisons
```

Each stage folder holds both its Python script(s) and the matching SLURM
launcher(s) together -- e.g. `01_extraction/pathology_extract.py` and
`01_extraction/submit_pathology_extraction.slurm` sit side by side."""
)

content = content.replace(
    """Submit each SLURM script **from within its own stage folder** -- e.g.:

```bash
cd slurm/01_extraction
sbatch submit_pathology_extraction.slurm
```""",
    """Submit each SLURM script **from within its own stage folder** -- e.g.:

```bash
cd 01_extraction
sbatch submit_pathology_extraction.slurm
```"""
)

with open("README.md", "w") as f:
    f.write(content)

print("README.md updated.")
PYEOF

echo
echo "############################################################"
echo "# 4. Stage for review"
echo "############################################################"

git add -A
git status

echo
echo "############################################################"
echo "# 5. Final structure check"
echo "############################################################"

find . -maxdepth 2 -not -path "./.git*" | sort

echo
echo "############################################################"
echo "# DONE -- nothing committed or pushed yet."
echo "# Review with: git status   and   git diff --cached --stat"
echo "# Then:"
echo "#   git commit -m 'Flatten to top-level stage folders (script+slurm together)'"
echo "#   git push"
echo "############################################################"
