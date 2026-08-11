"""
cluster_analysis.py

Consumes the embeddings produced by generate_embeddings.py and answers the
question your PI raised: instead of raw pairwise cosine similarity across
ALL patients (which collapses to ~0.99+ due to embedding anisotropy),
group patients by Subtype first, then:

  1. Within-cluster cosine similarity  (how tight is each subtype cluster?)
  2. Between-cluster cosine similarity (mean/min/max per subtype pair)
  3. Centroid-to-centroid distance     (cosine similarity AND Euclidean)
  4. t-SNE projection colored by subtype (visual confirmation)

Runs across all available models in parallel (CPU-only math + t-SNE, safe
to multiprocess — unlike embedding generation, this doesn't touch a GPU).

Usage:
    python cluster_analysis.py --embeddings-dir results/embeddings \
        --outdir results/analysis \
        --workers 5
"""

import argparse
import json
from itertools import combinations
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe for HPC/batch jobs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE

VALID_MODELS = ["bioclinicalbert", "pubmedbert", "biobert", "scibert", "sentence-transformers"]


def load_model_data(model_name: str, embeddings_dir: Path):
    mat = np.load(embeddings_dir / f"embeddings_{model_name}.npy")
    with open(embeddings_dir / f"patient_ids_{model_name}.txt") as f:
        patient_ids = f.read().splitlines()
    with open(embeddings_dir / f"subtypes_{model_name}.txt") as f:
        subtypes = f.read().splitlines()
    assert mat.shape[0] == len(patient_ids) == len(subtypes), \
        f"Row count mismatch for {model_name}: {mat.shape[0]} vs {len(patient_ids)} vs {len(subtypes)}"
    return mat, np.array(patient_ids), np.array(subtypes)


def cosine_sim_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = mat / norms
    return normed @ normed.T


def analyze_model(args_tuple):
    model_name, embeddings_dir, outdir = args_tuple
    embeddings_dir = Path(embeddings_dir)
    outdir = Path(outdir)

    print(f"[{model_name}] loading embeddings...")
    mat, patient_ids, subtypes = load_model_data(model_name, embeddings_dir)
    unique_subtypes = sorted(set(subtypes))
    print(f"[{model_name}] {mat.shape[0]} patients, {len(unique_subtypes)} subtypes: {unique_subtypes}")

    results = {"model": model_name, "n_patients": int(mat.shape[0]), "subtypes": unique_subtypes}

    # --- Within-cluster cosine similarity ---------------------------------
    within = {}
    for s in unique_subtypes:
        m = mat[subtypes == s]
        if len(m) < 2:
            continue
        sim = cosine_sim_matrix(m)
        mask = ~np.eye(len(m), dtype=bool)
        within[s] = {
            "n": int(len(m)),
            "mean": float(sim[mask].mean()),
            "min": float(sim[mask].min()),
            "max": float(sim[mask].max()),
        }
    results["within_cluster_cosine"] = within

    # --- Between-cluster cosine similarity (patient-level, all pairs) -----
    between = {}
    for s1, s2 in combinations(unique_subtypes, 2):
        m1, m2 = mat[subtypes == s1], mat[subtypes == s2]
        n1 = m1 / np.linalg.norm(m1, axis=1, keepdims=True)
        n2 = m2 / np.linalg.norm(m2, axis=1, keepdims=True)
        sim = n1 @ n2.T
        between[f"{s1}__vs__{s2}"] = {
            "mean": float(sim.mean()),
            "min": float(sim.min()),
            "max": float(sim.max()),
        }
    results["between_cluster_cosine"] = between

    # --- Centroid distances -------------------------------------------------
    centroids = np.vstack([mat[subtypes == s].mean(axis=0) for s in unique_subtypes])
    centroid_cos_sim = cosine_sim_matrix(centroids)
    centroid_euclidean = squareform(pdist(centroids, metric="euclidean"))

    results["centroid_cosine_similarity"] = {
        unique_subtypes[i]: {unique_subtypes[j]: float(centroid_cos_sim[i, j])
                              for j in range(len(unique_subtypes))}
        for i in range(len(unique_subtypes))
    }
    results["centroid_euclidean_distance"] = {
        unique_subtypes[i]: {unique_subtypes[j]: float(centroid_euclidean[i, j])
                              for j in range(len(unique_subtypes))}
        for i in range(len(unique_subtypes))
    }

    # Most / least separated subtype pair by centroid cosine similarity
    # (lower cosine similarity = more separated)
    pair_sims = {f"{unique_subtypes[i]}__vs__{unique_subtypes[j]}": float(centroid_cos_sim[i, j])
                 for i in range(len(unique_subtypes)) for j in range(i + 1, len(unique_subtypes))}
    most_separated = min(pair_sims, key=pair_sims.get)
    least_separated = max(pair_sims, key=pair_sims.get)
    results["most_separated_pair"] = {"pair": most_separated, "cosine_similarity": pair_sims[most_separated]}
    results["least_separated_pair"] = {"pair": least_separated, "cosine_similarity": pair_sims[least_separated]}

    # --- t-SNE visualization ------------------------------------------------
    print(f"[{model_name}] running t-SNE...")
    perplexity = max(5, min(30, mat.shape[0] // 20))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
    proj = tsne.fit_transform(mat)

    plt.figure(figsize=(8, 6))
    colors = cm.tab10(np.linspace(0, 1, len(unique_subtypes)))
    for s, c in zip(unique_subtypes, colors):
        idx = subtypes == s
        plt.scatter(proj[idx, 0], proj[idx, 1], label=f"{s} (n={idx.sum()})",
                    alpha=0.6, s=15, color=c)
    plt.legend(fontsize=8, loc="best")
    plt.title(f"t-SNE of clinical embeddings by Subtype — {model_name}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    fig_path = outdir / f"tsne_{model_name}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[{model_name}] saved {fig_path}")

    return model_name, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-dir", default="results/embeddings")
    ap.add_argument("--outdir", default="results/analysis")
    ap.add_argument("--models", nargs="+", default=None,
                     help="subset of models to analyze; default = all found in embeddings-dir")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.models:
        models = args.models
    else:
        # auto-detect which models have saved embeddings
        models = [m for m in VALID_MODELS if (embeddings_dir / f"embeddings_{m}.npy").exists()]
        if not models:
            raise SystemExit(f"No embeddings found in {embeddings_dir}. Run generate_embeddings.py first.")

    print(f"Analyzing models: {models}")

    tasks = [(m, str(embeddings_dir), str(outdir)) for m in models]
    all_results = {}

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            for model_name, results in pool.map(analyze_model, tasks):
                all_results[model_name] = results
    else:
        for t in tasks:
            model_name, results = analyze_model(t)
            all_results[model_name] = results

    out_json = outdir / "cluster_analysis_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined results to {out_json}")

    # --- Cross-model summary table ------------------------------------------
    print("\n=== Summary: most / least separated subtype pair per model (by centroid cosine sim) ===")
    print(f"{'Model':>22s}  {'Most separated pair':<35s} {'sim':>7s}  {'Least separated pair':<35s} {'sim':>7s}")
    for m, r in all_results.items():
        ms, ls = r["most_separated_pair"], r["least_separated_pair"]
        print(f"{m:>22s}  {ms['pair']:<35s} {ms['cosine_similarity']:>7.4f}  "
              f"{ls['pair']:<35s} {ls['cosine_similarity']:>7.4f}")


if __name__ == "__main__":
    main()
