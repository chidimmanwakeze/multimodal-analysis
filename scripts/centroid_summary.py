"""
centroid_summary.py

Reads results/analysis/cluster_analysis_results.json (already produced by
cluster_analysis.py) and, per model:
  1. Prints a clean most/least-separated summary using Euclidean centroid
     distance (cosine similarity was too compressed/anisotropic to be useful
     on its own -- see cluster_analysis_results.json's centroid_cosine_similarity
     for comparison).
  2. Builds a dendrogram showing how the 6 subtype centroids hierarchically
     group, based on pairwise Euclidean distance. This directly answers
     "how do the subtype clusters relate to each other in embedding space."

Usage:
    python centroid_summary.py --results results/analysis/cluster_analysis_results.json \
        --outdir results/analysis
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def build_condensed_distance(dist_dict: dict, labels: list) -> np.ndarray:
    """Convert the {label: {label: dist}} dict into scipy's condensed distance
    vector, matching `labels` order."""
    n = len(labels)
    square = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            square[i, j] = dist_dict[li][lj]
    return squareform(square, checks=False)


def summarize_model(model_name: str, model_results: dict, outdir: Path):
    dist_dict = model_results["centroid_euclidean_distance"]
    labels = model_results["subtypes"]

    # --- Most / least separated pair by Euclidean distance -----------------
    pair_dists = {}
    for a, b in combinations(labels, 2):
        pair_dists[f"{a}__vs__{b}"] = dist_dict[a][b]

    most_sep = max(pair_dists, key=pair_dists.get)
    least_sep = min(pair_dists, key=pair_dists.get)

    print(f"\n=== {model_name} (Euclidean centroid distance) ===")
    print(f"  Most separated:  {most_sep:<40s} dist={pair_dists[most_sep]:.4f}")
    print(f"  Least separated: {least_sep:<40s} dist={pair_dists[least_sep]:.4f}")
    print("  Full ranking (closest to farthest):")
    for pair, d in sorted(pair_dists.items(), key=lambda x: x[1]):
        print(f"    {pair:<40s} {d:.4f}")

    # --- Dendrogram ----------------------------------------------------------
    condensed = build_condensed_distance(dist_dict, labels)
    Z = linkage(condensed, method="average")

    plt.figure(figsize=(7, 5))
    dendrogram(Z, labels=labels, leaf_rotation=45)
    plt.title(f"Subtype centroid dendrogram — {model_name}\n(Euclidean distance, average linkage)")
    plt.ylabel("Distance")
    plt.tight_layout()
    fig_path = outdir / f"dendrogram_{model_name}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved {fig_path}")

    return {"most_separated": {"pair": most_sep, "distance": pair_dists[most_sep]},
            "least_separated": {"pair": least_sep, "distance": pair_dists[least_sep]},
            "ranking": sorted(pair_dists.items(), key=lambda x: x[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/analysis/cluster_analysis_results.json")
    ap.add_argument("--outdir", default="results/analysis")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        all_results = json.load(f)

    summary = {}
    for model_name, model_results in all_results.items():
        summary[model_name] = summarize_model(model_name, model_results, outdir)

    with open(outdir / "centroid_euclidean_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {outdir / 'centroid_euclidean_summary.json'}")


if __name__ == "__main__":
    main()
