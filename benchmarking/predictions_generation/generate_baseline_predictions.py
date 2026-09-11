# Not functional in public package. Script is just for reference.
"""
Generate non-LLM baseline predictions for the BioGRID gene ranking benchmark.

Baselines implemented:
  - random:                 Random permutation of screen gene library
  - global-hit-freq:        Rank by global hit frequency across training screens
  - screen-type-hit-freq:   Rank by hit frequency stratified by screen_type
  - phenotype-hit-freq:         Rank by hit frequency stratified by screen_rationale (~144 categories)
  - coarse-phenotype-hit-freq:  Rank by hit frequency stratified by coarse phenotype (5 categories)
  - phenotype-knn-hit-freq:     Rank by hit frequency from k-nearest training screens (TF-IDF)
  - bm25:                   Rank by BM25 similarity of gene NCBI summaries to screen question
  - pagerank:               Rank by PageRank on BioGRID PPI network
  - degree:                 Rank by degree centrality on BioGRID PPI network
  - gene-name-overlap:      Rank by token overlap between gene full name and screen question
  - library-size-prior:     Rank by how many training screen libraries contain the gene

Predictions are saved in the same JSON format as LLM predictions so they
can be evaluated with evaluate_model_splits.py via the baseline_models config.

Usage:
  python scripts/generate_baseline_predictions.py
  python scripts/generate_baseline_predictions.py baselines='["bm25", "pagerank"]'
  python scripts/generate_baseline_predictions.py baselines='["global-hit-freq"]' dataset.split_type=year dataset.fold=0
"""

import rootutils
rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import json
import os
import re
import zipfile
import random
import hydra
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, ListConfig
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from datasets import load_dataset, load_from_disk


# ============================================================================
# Helpers
# ============================================================================

def _save_predictions(
    output_dir: Path,
    baseline_name: str,
    split_name: str,
    questions: List[str],
    gene_rankings: List[List[str]],
):
    """Save predictions in the standard format used by evaluate_model_splits.py."""
    model_dir = output_dir / baseline_name
    model_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for question, genes in zip(questions, gene_rankings):
        predictions.append({
            "question": question,
            "predictions": [genes],
            "truncated": [False],
            "any_truncated": False,
        })

    data = {
        "model_name": baseline_name,
        "lm_config": {"provider": "baseline", "method": baseline_name},
        "n_runs": 1,
        "split": split_name,
        "num_examples": len(predictions),
        "num_examples_with_truncation": 0,
        "total_truncated_runs": 0,
        "predictions": predictions,
        "status": "complete",
    }

    out_file = model_dir / f"{split_name}_predictions.json"
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Saved {split_name}: {len(predictions)} examples -> {out_file}")


def _save_harmonized(
    harmonized_dir: Path,
    baseline_name: str,
    all_rankings: Dict[str, List[List[str]]],
    all_examples: Dict[str, List[Dict[str, Any]]],
    split_layout: str = "year",
):
    """Save predictions in the harmonized format used by the figures pipeline."""
    from datetime import datetime, timezone

    harmonized_dir.mkdir(parents=True, exist_ok=True)
    records_by_dataset = {}
    split_name_map = {"val": "val", "validation": "val", "train": "train", "test": "test",
                      "novel_public_dataset": "novel_public_dataset"}

    for split_name, rankings in all_rankings.items():
        examples = all_examples[split_name]
        harm_split = split_name_map.get(split_name, split_name)
        layout = "novel" if "novel" in split_name else split_layout
        for idx, (ex, genes) in enumerate(zip(examples, rankings)):
            ds_name = str(ex["dataset_name"])
            record = {
                "dataset_name": ds_name,
                "split": harm_split,
                "split_layout": layout,
                "predicted_genes": genes,
                "prediction_runs": [genes],
                "n_runs": 1,
                "example_key": f"{harm_split}:{idx}",
            }
            records_by_dataset.setdefault(ds_name, []).append(record)

    n_records = sum(len(v) for v in records_by_dataset.values())
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": f"baseline/{baseline_name}",
        "source_group": "baseline_predictions",
        "n_records": n_records,
        "n_unique_datasets": len(records_by_dataset),
        "records_by_dataset": records_by_dataset,
    }

    filename = f"baseline__{baseline_name}.json"
    out_file = harmonized_dir / filename
    with open(out_file, "w") as f:
        json.dump(payload, f)
    print(f"  Harmonized: {n_records} records -> {out_file}")


def _load_raw_dataset(dataset_path: str):
    """Load the raw HuggingFace dataset from disk."""
    return load_from_disk(dataset_path)


# ============================================================================
# Statistical Baselines
# ============================================================================

def compute_global_hit_frequency(
    train_examples: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute per-gene hit frequency across training screens."""
    gene_hit_count: Dict[str, int] = defaultdict(int)
    gene_total_count: Dict[str, int] = defaultdict(int)

    for ex in train_examples:
        genes = ex["relevance_genes"]
        hits = ex["hit"]
        for gene, is_hit in zip(genes, hits):
            gene_total_count[gene] += 1
            if is_hit:
                gene_hit_count[gene] += 1

    freq = {}
    for gene, total in gene_total_count.items():
        freq[gene] = gene_hit_count[gene] / total if total > 0 else 0.0
    return freq


def compute_screen_type_hit_frequency(
    train_examples: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute per-gene hit frequency stratified by screen_type."""
    counts: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )

    for ex in train_examples:
        stype = ex.get("screen_type", "unknown")
        genes = ex["relevance_genes"]
        hits = ex["hit"]
        for gene, is_hit in zip(genes, hits):
            counts[stype][gene][1] += 1
            if is_hit:
                counts[stype][gene][0] += 1

    freq_by_type: Dict[str, Dict[str, float]] = {}
    for stype, gene_counts in counts.items():
        freq_by_type[stype] = {}
        for gene, (hit_c, total_c) in gene_counts.items():
            freq_by_type[stype][gene] = hit_c / total_c if total_c > 0 else 0.0

    return freq_by_type


def rank_by_scores(
    screen_genes: List[str],
    score_dict: Dict[str, float],
) -> List[str]:
    """Rank genes by score (descending), with random tiebreaking."""
    scored = [(g, score_dict.get(g, 0.0)) for g in screen_genes]
    random.shuffle(scored)
    scored.sort(key=lambda x: x[1], reverse=True)
    return [g for g, _ in scored]


def baseline_random(examples: List[Dict[str, Any]], seed: int = 42) -> List[List[str]]:
    """Random permutation of each screen's gene library."""
    rng = random.Random(seed)
    rankings = []
    for ex in examples:
        genes = list(ex["relevance_genes"])
        rng.shuffle(genes)
        rankings.append(genes)
    return rankings


def baseline_global_hit_freq(
    examples: List[Dict[str, Any]],
    freq: Dict[str, float],
) -> List[List[str]]:
    """Rank genes by global hit frequency."""
    return [rank_by_scores(ex["relevance_genes"], freq) for ex in examples]


def baseline_screen_type_hit_freq(
    examples: List[Dict[str, Any]],
    freq_by_type: Dict[str, Dict[str, float]],
    global_freq: Dict[str, float],
) -> List[List[str]]:
    """Rank genes by screen-type-stratified hit frequency, falling back to global."""
    rankings = []
    for ex in examples:
        stype = ex.get("screen_type", "unknown")
        freq = freq_by_type.get(stype, global_freq)
        rankings.append(rank_by_scores(ex["relevance_genes"], freq))
    return rankings


def compute_phenotype_hit_frequency(
    train_examples: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute per-gene hit frequency stratified by screen_rationale (phenotype)."""
    counts: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )

    for ex in train_examples:
        phenotype = ex.get("screen_rationale", "unknown")
        if not phenotype or phenotype == "Not specified":
            phenotype = "unknown"
        genes = ex["relevance_genes"]
        hits = ex["hit"]
        for gene, is_hit in zip(genes, hits):
            counts[phenotype][gene][1] += 1
            if is_hit:
                counts[phenotype][gene][0] += 1

    freq_by_phenotype: Dict[str, Dict[str, float]] = {}
    for phenotype, gene_counts in counts.items():
        freq_by_phenotype[phenotype] = {}
        for gene, (hit_c, total_c) in gene_counts.items():
            freq_by_phenotype[phenotype][gene] = hit_c / total_c if total_c > 0 else 0.0

    return freq_by_phenotype


def baseline_phenotype_hit_freq(
    examples: List[Dict[str, Any]],
    freq_by_phenotype: Dict[str, Dict[str, float]],
    global_freq: Dict[str, float],
) -> List[List[str]]:
    """Rank genes by phenotype-stratified hit frequency, falling back to global."""
    rankings = []
    for ex in examples:
        phenotype = ex.get("screen_rationale", "unknown")
        if not phenotype or phenotype == "Not specified":
            phenotype = "unknown"
        freq = freq_by_phenotype.get(phenotype, global_freq)
        rankings.append(rank_by_scores(ex["relevance_genes"], freq))
    return rankings


def build_coarse_phenotype_map(
    examples: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Build dataset_name -> coarse phenotype mapping from cleaned_phenotype field."""
    return {ex["dataset_name"]: ex["cleaned_phenotype"] for ex in examples}


def compute_coarse_phenotype_hit_frequency(
    train_examples: List[Dict[str, Any]],
    ds_to_coarse: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-gene hit frequency stratified by coarse phenotype category (5 groups)."""
    counts: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )

    for ex in train_examples:
        coarse = ds_to_coarse.get(ex["dataset_name"], "unknown")
        genes = ex["relevance_genes"]
        hits = ex["hit"]
        for gene, is_hit in zip(genes, hits):
            counts[coarse][gene][1] += 1
            if is_hit:
                counts[coarse][gene][0] += 1

    freq_by_coarse: Dict[str, Dict[str, float]] = {}
    for coarse, gene_counts in counts.items():
        freq_by_coarse[coarse] = {}
        for gene, (hit_c, total_c) in gene_counts.items():
            freq_by_coarse[coarse][gene] = hit_c / total_c if total_c > 0 else 0.0

    return freq_by_coarse


def baseline_coarse_phenotype_hit_freq(
    examples: List[Dict[str, Any]],
    freq_by_coarse: Dict[str, Dict[str, float]],
    global_freq: Dict[str, float],
    ds_to_coarse: Dict[str, str],
) -> List[List[str]]:
    """Rank genes by coarse-phenotype-stratified hit frequency, falling back to global."""
    rankings = []
    for ex in examples:
        coarse = ds_to_coarse.get(ex["dataset_name"], "unknown")
        freq = freq_by_coarse.get(coarse, global_freq)
        rankings.append(rank_by_scores(ex["relevance_genes"], freq))
    return rankings


def baseline_phenotype_knn_hit_freq(
    examples: List[Dict[str, Any]],
    train_examples: List[Dict[str, Any]],
    k: int = 10,
) -> List[List[str]]:
    """
    For each screen, find k most similar training screens by TF-IDF on
    phenotype text, then rank genes by aggregated hit frequency from neighbors.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def _phenotype_text(ex):
        parts = []
        for field in ["phenotype", "condition_clause", "screen_rationale"]:
            val = ex.get(field, "")
            if val and val != "Not specified":
                parts.append(val)
        return " ".join(parts) if parts else ex.get("phenotype", "")

    train_texts = [_phenotype_text(ex) for ex in train_examples]
    eval_texts = [_phenotype_text(ex) for ex in examples]

    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", ngram_range=(1, 2)
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    eval_tfidf = vectorizer.transform(eval_texts)

    sims = cosine_similarity(eval_tfidf, train_tfidf)

    rankings = []
    for i, ex in enumerate(examples):
        top_k_idx = np.argsort(sims[i])[-k:][::-1]

        gene_hit_count: Dict[str, int] = defaultdict(int)
        gene_total_count: Dict[str, int] = defaultdict(int)
        for idx in top_k_idx:
            neighbor = train_examples[idx]
            for gene, is_hit in zip(neighbor["relevance_genes"], neighbor["hit"]):
                gene_total_count[gene] += 1
                if is_hit:
                    gene_hit_count[gene] += 1

        freq = {
            g: gene_hit_count[g] / gene_total_count[g]
            for g in gene_total_count
            if gene_total_count[g] > 0
        }
        rankings.append(rank_by_scores(ex["relevance_genes"], freq))

    return rankings


def baseline_library_size_prior(
    examples: List[Dict[str, Any]],
    train_examples: List[Dict[str, Any]],
) -> List[List[str]]:
    """Rank genes by how many training screen libraries they appear in."""
    library_count: Dict[str, int] = defaultdict(int)
    for ex in train_examples:
        for gene in ex["relevance_genes"]:
            library_count[gene] += 1

    return [rank_by_scores(ex["relevance_genes"], library_count) for ex in examples]


# ============================================================================
# BM25 Baseline
# ============================================================================

def load_gene_summaries(gene_data_dir: str) -> Dict[str, str]:
    """
    Load gene symbol -> summary text mapping from downloaded NCBI data.

    Uses gene_summaries.tsv (from download_gene_summaries.py) and
    Homo_sapiens.gene_info.gz for full_name fallback.
    """
    gene_data_dir = Path(gene_data_dir)
    summaries: Dict[str, str] = {}

    summary_file = gene_data_dir / "gene_summaries.tsv"
    if summary_file.exists():
        df = pd.read_csv(summary_file, sep="\t", dtype=str)
        for _, row in df.iterrows():
            symbol = row.get("symbol")
            summary = row.get("summary")
            full_name = row.get("full_name", "")
            if symbol and isinstance(symbol, str):
                text_parts = []
                if full_name and isinstance(full_name, str) and full_name != "nan":
                    text_parts.append(full_name)
                if summary and isinstance(summary, str) and summary != "nan":
                    text_parts.append(summary)
                if text_parts:
                    summaries[symbol] = " ".join(text_parts)

    gene_info_file = gene_data_dir / "Homo_sapiens.gene_info.gz"
    if gene_info_file.exists():
        info_df = pd.read_csv(gene_info_file, sep="\t", compression="gzip", dtype=str, na_values=["-", ""])
        for _, row in info_df.iterrows():
            symbol = row.get("Symbol")
            if symbol and symbol not in summaries:
                desc = row.get("description", "")
                full_name = row.get("Full_name_from_nomenclature_authority", "")
                text_parts = []
                if full_name and isinstance(full_name, str) and full_name != "nan":
                    text_parts.append(full_name)
                if desc and isinstance(desc, str) and desc != "nan":
                    text_parts.append(desc)
                if text_parts:
                    summaries[symbol] = " ".join(text_parts)

    return summaries


def baseline_bm25(
    examples: List[Dict[str, Any]],
    gene_summaries: Dict[str, str],
) -> List[List[str]]:
    """Rank genes by BM25 similarity of their NCBI summary to the screen question."""
    from rank_bm25 import BM25Okapi

    rankings = []
    for i, ex in enumerate(examples):
        screen_genes = ex["relevance_genes"]
        question = ex.get("question", ex.get("phenotype", ""))

        corpus = []
        gene_order = []
        for gene in screen_genes:
            text = gene_summaries.get(gene, "")
            corpus.append(text.lower().split() if text else [])
            gene_order.append(gene)

        query_tokens = question.lower().split()

        if not any(corpus):
            rankings.append(list(screen_genes))
            continue

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)

        scored = list(zip(gene_order, scores))
        random.shuffle(scored)
        scored.sort(key=lambda x: x[1], reverse=True)
        rankings.append([g for g, _ in scored])

        if (i + 1) % 100 == 0:
            print(f"    BM25: processed {i+1}/{len(examples)} screens")

    return rankings


# ============================================================================
# Gene Name Overlap Baseline
# ============================================================================

def baseline_gene_name_overlap(
    examples: List[Dict[str, Any]],
    gene_summaries: Dict[str, str],
) -> List[List[str]]:
    """
    Rank genes by token overlap between the gene's full name / description
    and the screen question text. A simple bag-of-words text matching approach.
    """
    stop_words = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "and", "but",
        "or", "nor", "not", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "than", "too", "very", "that", "this",
        "these", "those", "it", "its", "which", "who", "whom", "what", "where",
        "when", "why", "how", "protein", "gene", "genes", "protein-coding",
    })

    def tokenize(text: str) -> set:
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        return tokens - stop_words

    rankings = []
    for ex in examples:
        question_tokens = tokenize(ex.get("question", ex.get("phenotype", "")))
        screen_genes = ex["relevance_genes"]

        scored = []
        for gene in screen_genes:
            gene_text = gene_summaries.get(gene, "")
            if gene_text:
                gene_tokens = tokenize(gene_text)
                overlap = len(question_tokens & gene_tokens)
            else:
                overlap = 0
            scored.append((gene, overlap))

        random.shuffle(scored)
        scored.sort(key=lambda x: x[1], reverse=True)
        rankings.append([g for g, _ in scored])

    return rankings


# ============================================================================
# Network Baselines (PageRank, Degree)
# ============================================================================

def download_biogrid_ppi(cache_dir: str) -> Path:
    """
    Download BioGRID PPI data and extract the human interactions file.
    Returns path to the extracted TSV file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    human_files = list(cache_dir.glob("BIOGRID-ORGANISM-Homo_sapiens-*.tab3.txt"))
    if human_files:
        print(f"  Using cached BioGRID file: {human_files[0].name}")
        return human_files[0]

    zip_path = cache_dir / "BIOGRID-ORGANISM-LATEST.tab3.zip"

    if not zip_path.exists():
        url = "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ORGANISM-LATEST.tab3.zip"
        print(f"  Downloading BioGRID PPI data from {url}...")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Downloaded {zip_path.stat().st_size / 1e6:.1f} MB")

    print("  Extracting human interactions file...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if "Homo_sapiens" in name and name.endswith(".tab3.txt"):
                zf.extract(name, cache_dir)
                extracted = cache_dir / name
                print(f"  Extracted: {extracted.name}")
                return extracted

    raise FileNotFoundError("No Homo_sapiens tab3 file found in BioGRID zip")


def build_ppi_graph(biogrid_file: Path):
    """Build a networkx graph from BioGRID tab3 human interactions."""
    import networkx as nx

    print("  Building PPI graph...")
    df = pd.read_csv(biogrid_file, sep="\t", dtype=str, low_memory=False)

    df = df[df["Experimental System Type"] == "physical"]
    df = df[["Official Symbol Interactor A", "Official Symbol Interactor B"]].dropna()
    df = df[df["Official Symbol Interactor A"] != df["Official Symbol Interactor B"]]

    G = nx.from_pandas_edgelist(df, "Official Symbol Interactor A", "Official Symbol Interactor B")

    print(f"  PPI graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_network_scores(cache_dir: str, return_graph: bool = False):
    """
    Download BioGRID, build PPI graph, compute PageRank and degree.
    Caches results to disk.
    """
    import networkx as nx

    cache_dir = Path(cache_dir)
    pagerank_file = cache_dir / "pagerank_scores.json"
    degree_file = cache_dir / "degree_scores.json"

    G = None
    if pagerank_file.exists() and degree_file.exists():
        print("  Loading cached network scores...")
        with open(pagerank_file) as f:
            pagerank_scores = json.load(f)
        with open(degree_file) as f:
            degree_scores = json.load(f)
        print(f"  Loaded PageRank for {len(pagerank_scores)} genes, degree for {len(degree_scores)} genes")
    else:
        biogrid_file = download_biogrid_ppi(cache_dir)
        G = build_ppi_graph(biogrid_file)

        print("  Computing PageRank (this may take a minute)...")
        pagerank_scores = nx.pagerank(G)

        degree_scores = {node: deg for node, deg in G.degree()}

        with open(pagerank_file, "w") as f:
            json.dump(pagerank_scores, f)
        with open(degree_file, "w") as f:
            json.dump(degree_scores, f)

        print(f"  Cached network scores to {cache_dir}")

    if return_graph:
        if G is None:
            biogrid_file = download_biogrid_ppi(cache_dir)
            G = build_ppi_graph(biogrid_file)
        return pagerank_scores, degree_scores, G

    return pagerank_scores, degree_scores


def _precompute_neighbor_sets(G) -> Dict[str, set]:
    """Pre-compute neighbor sets for all nodes (avoids repeated graph lookups)."""
    return {node: set(G.neighbors(node)) for node in G.nodes()}


def _ppi_neighbor_score(
    neighbor_sets: Dict[str, set],
    seed_genes: set,
    candidate_genes: List[str],
) -> Dict[str, float]:
    """Score each candidate gene by the fraction of its PPI neighbors that are seeds."""
    scores = {}
    for gene in candidate_genes:
        neighbors = neighbor_sets.get(gene)
        if not neighbors:
            scores[gene] = 0.0
            continue
        scores[gene] = len(neighbors & seed_genes) / len(neighbors)
    return scores


def baseline_ppi_coarse_phenotype(
    examples: List[Dict[str, Any]],
    train_examples: List[Dict[str, Any]],
    neighbor_sets: Dict[str, set],
    ds_to_coarse: Dict[str, str],
) -> List[List[str]]:
    """Rank genes by PPI neighbor overlap with hits from same coarse phenotype."""
    from collections import defaultdict
    hits_by_coarse = defaultdict(set)
    for ex in train_examples:
        coarse = ds_to_coarse.get(ex["dataset_name"])
        if coarse is None:
            continue
        for gene, is_hit in zip(ex["relevance_genes"], ex["hit"]):
            if is_hit:
                hits_by_coarse[coarse].add(gene)

    rankings = []
    for ex in examples:
        coarse = ds_to_coarse.get(ex["dataset_name"])
        seeds = hits_by_coarse.get(coarse, set())
        scores = _ppi_neighbor_score(neighbor_sets, seeds, ex["relevance_genes"])
        rankings.append(rank_by_scores(ex["relevance_genes"], scores))
    return rankings


def baseline_ppi_knn(
    examples: List[Dict[str, Any]],
    train_examples: List[Dict[str, Any]],
    neighbor_sets: Dict[str, set],
    k: int = 10,
    embedding_cache_path: str = None,
    dotenv_path: str = "/cv/home/debroue1/from_prescient/projects/promptoptbase/.env",
) -> List[List[str]]:
    """Rank genes by PPI neighbor overlap with hits from k nearest training screens (LLM embeddings)."""
    from openai import AzureOpenAI
    from dotenv import dotenv_values

    env = dotenv_values(dotenv_path)

    cache = {}
    if embedding_cache_path and Path(embedding_cache_path).exists():
        import pickle
        with open(embedding_cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"  Loaded {len(cache)} cached embeddings")

    client = AzureOpenAI(
        azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
        api_key=env["API_KEY_ES2"],
        api_version="2025-04-01-preview",
    )

    def _get_embedding(text):
        if text in cache:
            return cache[text]
        response = client.embeddings.create(model="text-embedding-3-small", input=text)
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        cache[text] = embedding
        return embedding

    _SCREEN_CONTEXT_TEMPLATE = (
        "This screen was performed in {cell_line} cells, a {cell_type}. "
        "Researchers used a {library_type} library ({library_methodology}) to systematically perturb gene function. "
        "The experiment followed a {experimental_setup} design and was conducted over {duration}{condition_clause}.\n"
        "The primary objective of this screen was to identify a set of hit genes, each of which {phenotype}\n"
        'A gene is classified as a "hit" if its {library_methodology} significantly {phenotype} '
        "The statistical criterion for significance is: {significance_criteria}.\n"
        "Genes with {ranking_rationale} are ranked most highly.\n"
        "Screen notes: {notes}"
    )

    def _phenotype_text(ex):
        phenotype = ex.get("phenotype", "")
        if phenotype and phenotype[-1] == ".":
            phenotype = phenotype[:-1]
        fields = {k: ex.get(k, "") for k in [
            "cell_line", "cell_type", "library_type", "library_methodology",
            "experimental_setup", "duration", "condition_clause",
            "significance_criteria", "ranking_rationale", "notes",
        ]}
        fields["phenotype"] = phenotype
        return _SCREEN_CONTEXT_TEMPLATE.format(**fields)

    train_texts = [_phenotype_text(ex) for ex in train_examples]
    eval_texts = [_phenotype_text(ex) for ex in examples]

    all_texts = list(set(train_texts + eval_texts))
    uncached = [t for t in all_texts if t not in cache]
    print(f"  {len(all_texts)} unique texts, {len(uncached)} need embedding")

    for i, text in enumerate(uncached):
        _get_embedding(text)
        if (i + 1) % 100 == 0:
            print(f"    Embedded {i + 1}/{len(uncached)}")

    if uncached and embedding_cache_path:
        import pickle
        with open(embedding_cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"  Updated embedding cache at {embedding_cache_path}")

    train_embeddings = np.array([cache[t] for t in train_texts])
    eval_embeddings = np.array([cache[t] for t in eval_texts])

    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(eval_embeddings, train_embeddings)

    rankings = []
    for i, ex in enumerate(examples):
        top_k_idx = np.argsort(sims[i])[-k:][::-1]
        seeds = set()
        for idx in top_k_idx:
            neighbor = train_examples[idx]
            for gene, is_hit in zip(neighbor["relevance_genes"], neighbor["hit"]):
                if is_hit:
                    seeds.add(gene)
        scores = _ppi_neighbor_score(neighbor_sets, seeds, ex["relevance_genes"])
        rankings.append(rank_by_scores(ex["relevance_genes"], scores))
    return rankings


def baseline_ppi_entry_point(
    examples: List[Dict[str, Any]],
    G,
    entry_points_map: Dict[str, Dict[str, Any]],
    seed: int = 42,
) -> Tuple[List[List[str]], List[str]]:
    """Rank genes by shortest-path distance to LLM-extracted entry points in PPI graph."""
    import networkx as nx

    rng = random.Random(seed)
    rankings = []
    flagged = []

    for ex in examples:
        ds_name = str(ex["dataset_name"])
        ep_info = entry_points_map.get(ds_name, {})
        seeds = set(ep_info.get("entry_points", [])) & set(G.nodes())

        if not seeds:
            flagged.append(ds_name)
            genes = list(ex["relevance_genes"])
            rng.shuffle(genes)
            rankings.append(genes)
            continue

        distances = {}
        for seed_gene in seeds:
            lengths = nx.single_source_shortest_path_length(G, seed_gene)
            for gene, dist in lengths.items():
                if gene not in distances or dist < distances[gene]:
                    distances[gene] = dist

        scores = {}
        for gene in ex["relevance_genes"]:
            if gene in distances:
                scores[gene] = 1.0 / (1.0 + distances[gene])
            else:
                scores[gene] = 0.0

        rankings.append(rank_by_scores(ex["relevance_genes"], scores))

    return rankings, flagged


# ============================================================================
# Main
# ============================================================================

ALL_BASELINES = [
    "random",
    "global-hit-freq",
    "screen-type-hit-freq",
    "phenotype-hit-freq",
    "coarse-phenotype-hit-freq",
    "phenotype-knn-hit-freq",
    "bm25",
    "pagerank",
    "degree",
    "gene-name-overlap",
    "library-size-prior",
    "ppi-coarse-phenotype",
    "ppi-knn",
    "ppi-entry-point",
]


@hydra.main(version_base="1.3", config_path="../configs", config_name="generate-baselines")
def main(cfg: DictConfig):
    print("=" * 60)
    print("GENERATE BASELINE PREDICTIONS")
    print("=" * 60)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if isinstance(cfg.baselines, ListConfig):
        baselines = list(cfg.baselines)
    elif cfg.baselines == "all":
        baselines = ALL_BASELINES
    else:
        baselines = [cfg.baselines]

    print(f"\nBaselines to generate: {baselines}")

    # Load dataset from HuggingFace Hub (cached automatically)
    dataset_group = cfg.dataset.get("dataset_group", "Genentech/assaybench")
    dataset_name = cfg.dataset.get("dataset_name", "biogrid")
    print(f"\nLoading dataset: {dataset_group}/{dataset_name}...")
    hf_ds = load_dataset(dataset_group, dataset_name, split="train")
    split_col = f"{cfg.dataset.split_type}fold{cfg.dataset.fold}"
    print(f"  Using split column: {split_col}")

    def _hf_row_to_example(row):
        """Convert a HF dataset row to the dict format expected by baselines."""
        return {k: row[k] for k in row.keys()}

    all_examples = [_hf_row_to_example(hf_ds[i]) for i in range(len(hf_ds))]
    train_raw = [ex for ex in all_examples if ex[split_col] == "train"]
    val_raw = [ex for ex in all_examples if ex[split_col] == "validation"]
    test_raw = [ex for ex in all_examples if ex[split_col] == "test"]

    # Load LaTest (novel) dataset
    novel_dataset_name = cfg.dataset.get("novel_dataset_name", "LaTest")
    print(f"\nLoading novel dataset: {dataset_group}/{novel_dataset_name}...")
    novel_ds = load_dataset(dataset_group, novel_dataset_name, split="train")
    novel_raw = [_hf_row_to_example(novel_ds[i]) for i in range(len(novel_ds))]

    print(f"  Train: {len(train_raw)}, Val: {len(val_raw)}, Test: {len(test_raw)}, LaTest: {len(novel_raw)}")

    # Build a question string per screen (used for saving and text-based baselines)
    train_questions = [ex.get("phenotype", "") for ex in train_raw]
    val_questions = [ex.get("phenotype", "") for ex in val_raw]
    test_questions = [ex.get("phenotype", "") for ex in test_raw]
    novel_questions = [ex.get("phenotype", "") for ex in novel_raw]

    output_dir = Path(cfg.output.save_dir) / "baseline_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Precompute shared resources
    global_freq = None
    freq_by_type = None
    gene_summaries = None
    pagerank_scores = None
    degree_scores = None

    needs_freq = any(
        b in baselines
        for b in ["global-hit-freq", "screen-type-hit-freq", "phenotype-hit-freq",
                   "coarse-phenotype-hit-freq", "phenotype-knn-hit-freq", "library-size-prior"]
    )
    needs_summaries = any(b in baselines for b in ["bm25", "gene-name-overlap"])
    needs_network = any(b in baselines for b in ["pagerank", "degree", "ppi-coarse-phenotype", "ppi-knn", "ppi-entry-point"])

    freq_by_phenotype = None
    freq_by_coarse = None
    ds_to_coarse = None

    if needs_freq:
        print("\nComputing hit frequencies from training data...")
        global_freq = compute_global_hit_frequency(train_raw)
        freq_by_type = compute_screen_type_hit_frequency(train_raw)
        freq_by_phenotype = compute_phenotype_hit_frequency(train_raw)
        print(f"  Global freq computed for {len(global_freq)} genes")
        print(f"  Screen types: {list(freq_by_type.keys())}")
        print(f"  Phenotypes (screen_rationale): {list(freq_by_phenotype.keys())}")

    if "coarse-phenotype-hit-freq" in baselines or "ppi-coarse-phenotype" in baselines:
        print("\nBuilding coarse phenotype mapping (5 categories)...")
        all_raw = train_raw + val_raw + test_raw + novel_raw
        ds_to_coarse = build_coarse_phenotype_map(all_raw)
        if "coarse-phenotype-hit-freq" in baselines:
            freq_by_coarse = compute_coarse_phenotype_hit_frequency(train_raw, ds_to_coarse)
            print(f"  Coarse phenotype categories: {list(freq_by_coarse.keys())}")

    if needs_summaries:
        print(f"\nLoading gene summaries from {cfg.gene_data_dir}...")
        gene_summaries = load_gene_summaries(cfg.gene_data_dir)
        print(f"  Loaded summaries for {len(gene_summaries)} genes")

    ppi_neighbor_sets = None
    ppi_graph = None
    entry_points_map = None
    if needs_network:
        needs_graph = any(b in baselines for b in ["ppi-coarse-phenotype", "ppi-knn", "ppi-entry-point"])
        print(f"\nLoading/computing network scores from {cfg.ppi_data_dir}...")
        if needs_graph:
            pagerank_scores, degree_scores, ppi_graph = compute_network_scores(
                cfg.ppi_data_dir, return_graph=True
            )
            print("  Pre-computing PPI neighbor sets...")
            ppi_neighbor_sets = _precompute_neighbor_sets(ppi_graph)
            if "ppi-entry-point" not in baselines:
                del ppi_graph
                ppi_graph = None
        else:
            pagerank_scores, degree_scores = compute_network_scores(cfg.ppi_data_dir)

    if "ppi-entry-point" in baselines:
        ep_path = Path(cfg.ppi_data_dir) / "entry_points.json"
        if not ep_path.exists():
            raise FileNotFoundError(
                f"Entry points file not found: {ep_path}\n"
                "Run extract_entry_points.py first."
            )
        with open(ep_path) as f:
            entry_points_map = json.load(f)
        n_with = sum(1 for v in entry_points_map.values() if v.get("has_entry_points"))
        print(f"\nLoaded entry points: {n_with}/{len(entry_points_map)} screens have entry points")

    # Harmonized output directory (alongside raw predictions)
    harmonized_dir = Path(cfg.output.save_dir).parent.parent / "predictions" / "baselines"

    # Generate each baseline
    for baseline_name in baselines:
        print(f"\n{'='*50}")
        print(f"Generating: {baseline_name}")
        print(f"{'='*50}")

        splits = {"train": train_raw, "val": val_raw, "test": test_raw, "novel_public_dataset": novel_raw}
        question_lists = {"train": train_questions, "val": val_questions, "test": test_questions, "novel_public_dataset": novel_questions}
        all_rankings_for_harmonized = {}
        all_flagged_screens = []

        for split_name, examples in splits.items():
            questions = question_lists[split_name]

            if baseline_name == "random":
                rankings = baseline_random(examples, seed=cfg.seed)

            elif baseline_name == "global-hit-freq":
                rankings = baseline_global_hit_freq(examples, global_freq)

            elif baseline_name == "screen-type-hit-freq":
                rankings = baseline_screen_type_hit_freq(
                    examples, freq_by_type, global_freq
                )

            elif baseline_name == "phenotype-hit-freq":
                rankings = baseline_phenotype_hit_freq(
                    examples, freq_by_phenotype, global_freq
                )

            elif baseline_name == "coarse-phenotype-hit-freq":
                rankings = baseline_coarse_phenotype_hit_freq(
                    examples, freq_by_coarse, global_freq, ds_to_coarse
                )

            elif baseline_name == "phenotype-knn-hit-freq":
                k = cfg.get("knn_k", 10)
                rankings = baseline_phenotype_knn_hit_freq(
                    examples, train_raw, k=k
                )

            elif baseline_name == "bm25":
                rankings = baseline_bm25(examples, gene_summaries)

            elif baseline_name == "pagerank":
                rankings = [
                    rank_by_scores(ex["relevance_genes"], pagerank_scores)
                    for ex in examples
                ]

            elif baseline_name == "degree":
                float_degree = {k: float(v) for k, v in degree_scores.items()}
                rankings = [
                    rank_by_scores(ex["relevance_genes"], float_degree)
                    for ex in examples
                ]

            elif baseline_name == "gene-name-overlap":
                rankings = baseline_gene_name_overlap(examples, gene_summaries)

            elif baseline_name == "library-size-prior":
                rankings = baseline_library_size_prior(examples, train_raw)

            elif baseline_name == "ppi-coarse-phenotype":
                rankings = baseline_ppi_coarse_phenotype(
                    examples, train_raw, ppi_neighbor_sets, ds_to_coarse
                )

            elif baseline_name == "ppi-knn":
                k = cfg.get("knn_k", 10)
                cache_path = str(Path(cfg.ppi_data_dir) / "phenotype_embedding_cache.pkl")
                rankings = baseline_ppi_knn(
                    examples, train_raw, ppi_neighbor_sets, k=k,
                    embedding_cache_path=cache_path,
                )

            elif baseline_name == "ppi-entry-point":
                rankings, flagged_screens = baseline_ppi_entry_point(
                    examples, ppi_graph, entry_points_map, seed=cfg.seed
                )
                all_flagged_screens.extend(flagged_screens)
                if flagged_screens:
                    print(f"    {len(flagged_screens)}/{len(examples)} screens flagged (no entry points)")

            else:
                print(f"  Unknown baseline: {baseline_name}, skipping")
                continue

            _save_predictions(
                output_dir, baseline_name, split_name, questions, rankings
            )
            all_rankings_for_harmonized[split_name] = rankings

        _save_harmonized(harmonized_dir, baseline_name, all_rankings_for_harmonized, splits)

        if baseline_name == "ppi-entry-point" and all_flagged_screens:
            flagged_path = harmonized_dir / "ppi_entry_point_flagged_screens.json"
            flagged_set = sorted(set(all_flagged_screens))
            with open(flagged_path, "w") as f:
                json.dump({"flagged_screens": flagged_set, "n_flagged": len(flagged_set)}, f, indent=2)
            print(f"  Flagged screens saved to {flagged_path} ({len(flagged_set)} screens)")

    print(f"\n{'='*60}")
    print("ALL BASELINES GENERATED!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Baselines: {baselines}")
    print(f"\nEvaluate with:")
    bl_list = '", "'.join(baselines)
    print(f'  python scripts/evaluate_model_splits.py baseline_models=\'["{bl_list}"]\'')


if __name__ == "__main__":
    main()
