#!/usr/bin/env python3
"""Re-evaluate the best Bayesian ensemble trial with all metrics.

Loads the best-on-validation trial from the Bayesian optimization results,
reconstructs the ensemble function, and evaluates on val / test / novel
splits with the full metric set (adjusted_ndcg@100, precision@100,
inverse_precision@100).

Results are saved to a JSON file that generate_journal_figures.py can
consume.

Usage:
    cd promptoptbase && uv run python scripts/evaluate_best_ensemble.py
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ensemble_learn.shared.data_loading import load_all_model_predictions, load_ground_truth
from ensemble_learn.shared.metrics import RankingMetrics

BAYESIAN_DIR = BASE_DIR / "ensemble_learn" / "bayesian_results"

ENSEMBLE_RUNS = [
    ("multi_model_knn", BAYESIAN_DIR / "multi_model_knn" / "trials.jsonl"),
    ("top3_equal_weights", BAYESIAN_DIR / "top3_equal_weights" / "trials.jsonl"),
    ("top3_optimize_all", BAYESIAN_DIR / "top3_optimize_all" / "trials.jsonl"),
]

TOP3_MODELS = ["gemini-3-pro", "gpt-5.4", "gemini-3-flash"]

MULTI_MODEL_MODELS = [
    "gpt-oss-20b", "gpt-oss-120b", "gpt-5-mini", "gpt-5.2", "gpt-5.4",
    "claude-haiku-4.5", "claude-sonnet-4.5", "claude-opus-4.5",
    "gemini-3-flash", "gemini-3-pro", "gemini-3.1-pro",
    "Kimi-K2.5", "qwen3-4b-2507", "qwen3-30b-a3b-2507",
    "qwen3-235b-a22b-2507", "qwen3.5-397b-a17b",
    "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
    "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-122b-a10b",
    "qwen3-coder-next", "deepseek-v3.2", "deepseek-v3.2-nothink",
    "MiniMax-M2.5", "GLM-5", "olmo-3.1-32b-think",
]

CONFIG_MODELS = {
    "multi_model_knn": MULTI_MODEL_MODELS,
    "top3_equal_weights": TOP3_MODELS,
    "top3_optimize_all": TOP3_MODELS,
}

METRICS_TO_EVAL = ["adjusted_ndcg@100", "precision@100", "inverse_precision@100"]

OUTPUT_PATH = BAYESIAN_DIR / "best_ensemble_all_metrics.json"

NOVEL_SPLIT = "novel_public_dataset"
NOVEL_DATASET_PATH = Path("./data/novel_public_2026_dataset")


def _best_trial_from_jsonl(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    best = None
    best_val = -1.0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            v = d.get("val_score")
            if v is None:
                continue
            if v > best_val:
                best_val = float(v)
                best = d
    return best


def find_best_trial() -> Tuple[str, Dict]:
    best_trial = None
    best_val = -1.0
    best_config = None
    for label, path in ENSEMBLE_RUNS:
        bt = _best_trial_from_jsonl(path)
        if bt is None:
            continue
        v = bt.get("val_score", -1.0)
        if v > best_val:
            best_val = v
            best_trial = bt
            best_config = label
    if best_trial is None:
        raise RuntimeError("No Bayesian trials found")
    return best_config, best_trial


def build_ensemble_fn(params: Dict[str, Any], model_list: List[str]):
    """Reconstruct the ensemble function from trial params."""
    model_weights = {}
    for m in model_list:
        safe = m.replace("-", "_").replace(".", "_")
        model_weights[m] = params.get(f"w_{safe}", 1.0)

    k_rrf = params.get("k_rrf", 60.0)
    n_neighbors = params.get("n_neighbors", 0)
    sim_threshold = params.get("sim_threshold", 0.9)
    knn_boost_factor = params.get("knn_boost_factor", 0.0)
    knn_boost_min_neighbors = params.get("knn_boost_min_neighbors", 1)
    agreement_bonus_2 = params.get("agreement_bonus_2", 1.0)
    agreement_bonus_3 = params.get("agreement_bonus_3", 1.0)
    agreement_threshold = params.get("agreement_threshold", 5)

    def ensemble(
        all_model_predictions: Dict[str, List[List[str]]],
        top_k: int = 100,
    ) -> List[str]:
        gene_scores: Dict[str, float] = defaultdict(float)
        gene_model_count: Dict[str, int] = defaultdict(int)

        for model_name, runs in all_model_predictions.items():
            w = model_weights.get(model_name, 1.0)
            if w < 0.01:
                continue
            for run in runs:
                for rank, gene in enumerate(run):
                    g = gene.strip().upper()
                    if g:
                        gene_scores[g] += w / (k_rrf + rank + 1)
            genes_from_model = set()
            for run in runs:
                for gene in run[:top_k]:
                    genes_from_model.add(gene.strip().upper())
            for g in genes_from_model:
                gene_model_count[g] += 1

        for g, count in gene_model_count.items():
            if count >= agreement_threshold:
                if count >= 3:
                    gene_scores[g] *= agreement_bonus_3
                elif count >= 2:
                    gene_scores[g] *= agreement_bonus_2

        return sorted(gene_scores, key=gene_scores.get, reverse=True)[:top_k]

    return ensemble


def load_novel_ground_truth(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load all examples from the novel dataset (no split filtering)."""
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_path))
    ground_truth = []
    for i in range(len(dataset)):
        item = dataset[i]
        ground_truth.append({
            "genes": [g.upper() for g in item["relevance_genes"]],
            "relevance_scores": item["relevance_scores"],
            "phenotype": item.get("phenotype", ""),
            "screen_rationale": item.get("screen_rationale", ""),
            "cell_line": item.get("cell_line", ""),
            "condition_name": item.get("condition_name", ""),
        })
    return ground_truth


def evaluate_split(
    ensemble_fn,
    all_preds: Dict[str, List[Dict]],
    ground_truth: List[Dict],
    metrics_eval: RankingMetrics,
    split_name: str,
) -> Dict[str, float]:
    """Run the ensemble on every example in a split, return mean metrics."""
    ref_model = max(all_preds, key=lambda m: len(all_preds[m]))
    n_examples = min(len(all_preds[ref_model]), len(ground_truth))

    per_metric: Dict[str, List[float]] = defaultdict(list)
    skipped = 0

    for i in range(n_examples):
        ep: Dict[str, List[List[str]]] = {}
        for model_name, predictions in all_preds.items():
            if i < len(predictions):
                ep[model_name] = predictions[i]["predictions"]

        if not ep:
            skipped += 1
            continue

        gt = ground_truth[i]
        genes = ensemble_fn(ep, top_k=100)
        result = metrics_eval.evaluate(genes, gt["genes"], gt["relevance_scores"])

        for m in METRICS_TO_EVAL:
            v = result.get(m)
            if v is not None and not np.isnan(v):
                per_metric[m].append(v)

    agg = {}
    for m in METRICS_TO_EVAL:
        vals = per_metric.get(m, [])
        agg[m] = float(np.mean(vals)) if vals else None
        print(f"    {m}: {agg[m]:.6f}  (n={len(vals)})" if agg[m] is not None else f"    {m}: N/A")

    if skipped:
        print(f"    (skipped {skipped} examples with no predictions)")

    return agg


def main():
    t0 = time.time()

    config_name, trial = find_best_trial()
    params = trial["params"]
    model_list = CONFIG_MODELS[config_name]

    print(f"Best ensemble config: {config_name}")
    print(f"  val_score (AnDCG@100): {trial['val_score']:.6f}")
    print(f"  test_score (AnDCG@100): {trial['test_score']:.6f}")
    print(f"  models: {model_list}")
    print(f"  params: {json.dumps(params, indent=2)}")
    print()

    ensemble_fn = build_ensemble_fn(params, model_list)

    metrics_eval = RankingMetrics(
        k_values=[100],
        use_gene_mapper=True,
        metric_groups=["ndcg", "adjusted_ndcg", "precision", "inverse_precision"],
    )

    results: Dict[str, Any] = {
        "config": config_name,
        "params": params,
        "model_list": model_list,
        "original_val_score": trial["val_score"],
        "original_test_score": trial["test_score"],
        "splits": {},
    }

    splits_to_eval = [
        ("val", "val", None),
        ("test", "test", None),
        (NOVEL_SPLIT, "novel", NOVEL_DATASET_PATH),
    ]

    for split_name, cohort_name, dataset_path in splits_to_eval:
        print(f"Evaluating on {split_name} (cohort={cohort_name}) ...")
        t1 = time.time()

        print(f"  Loading predictions ...")
        all_preds = load_all_model_predictions(split_name, models=model_list)
        print(f"    {len(all_preds)} models loaded")

        print(f"  Loading ground truth ...")
        if dataset_path is not None:
            gt = load_novel_ground_truth(dataset_path)
        else:
            gt = load_ground_truth(split_name)
        print(f"    {len(gt)} examples")

        agg = evaluate_split(ensemble_fn, all_preds, gt, metrics_eval, split_name)
        results["splits"][cohort_name] = agg
        print(f"  Done in {time.time() - t1:.1f}s")
        print()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
