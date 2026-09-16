"""
select_best_model.py

Answers "which single model should we standardize on" using a concrete,
data-driven metric: dynamic range of centroid Euclidean distances
(farthest subtype pair / closest subtype pair). A bigger ratio means the
model draws a sharper distinction between subtypes relative to its own
noise floor -- not just "biggest raw distance," since raw distance scale
differs across models with different embedding dimensions.

Works on whatever cluster_analysis_results.json files already exist --
clinical, pathology (json), pathology (template) -- and reports per-source,
plus an overall ranking averaged across all available sources.

Usage:
    python select_best_model.py \
        --results results/analysis/cluster_analysis_results.json:clinical \
        --results results/pathology_analysis/json/cluster_analysis_results.json:pathology_json \
        --results results/pathology_analysis/template/cluster_analysis_results.json:pathology_template
"""

import argparse
import json
from pathlib import Path


def dynamic_range(model_results: dict, exclude_labels: set) -> float:
    euc = model_results["centroid_euclidean_distance"]
    labels = [l for l in model_results["subtypes"] if l not in exclude_labels]
    dists = [euc[a][b] for i, a in enumerate(labels) for b in labels[i + 1:]]
    return max(dists) / min(dists) if min(dists) > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", required=True,
                     help="path:label, e.g. results/analysis/cluster_analysis_results.json:clinical. "
                          "Repeat for each source you want included in the ranking.")
    ap.add_argument("--exclude", action="append", default=["Unknown"],
                     help="subtype label(s) to exclude from the ranking (default: Unknown, "
                          "since it's a missingness artifact, not a real biological group -- "
                          "including it distorts the metric toward whichever model exaggerates "
                          "the artifact most, not whichever model resolves real subtypes best).")
    args = ap.parse_args()
    exclude_labels = set(args.exclude)

    per_source = {}
    for spec in args.results:
        path, label = spec.rsplit(":", 1)
        if not Path(path).exists():
            print(f"[skip] {label}: {path} not found yet")
            continue
        data = json.load(open(path))
        scores = {model: dynamic_range(res, exclude_labels) for model, res in data.items()}
        per_source[label] = scores

    if not per_source:
        print("No result files found yet -- nothing to rank.")
        return

    print(f"{'Model':<22s}" + "".join(f"{label:>18s}" for label in per_source) + f"{'AVERAGE':>12s}")
    all_models = sorted(set(m for scores in per_source.values() for m in scores))
    averages = {}
    for model in all_models:
        row_scores = [per_source[label].get(model) for label in per_source]
        present = [s for s in row_scores if s is not None]
        avg = sum(present) / len(present) if present else float("nan")
        averages[model] = avg
        row = f"{model:<22s}" + "".join(
            f"{per_source[label].get(model, float('nan')):>18.2f}" for label in per_source
        ) + f"{avg:>12.2f}"
        print(row)

    best = max(averages, key=averages.get)
    print(f"\nHighest average dynamic range: {best} ({averages[best]:.2f})")
    print("Recommendation: standardize on this model for future runs, "
          "rather than generating all 5 every time.")


if __name__ == "__main__":
    main()
