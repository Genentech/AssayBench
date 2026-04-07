from __future__ import annotations

import csv
import json
import pickle
import random as random_module
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from journal_figures_common import (
    BEST_ENSEMBLE_ALL_METRICS,
    CACHE_SCHEMA_VERSION,
    CLASSIFIER_RANDOM_CSV,
    CLASSIFIER_RECENT_CSV,
    CLASSIFIER_YEAR_CSV,
    DATASET_PATH,
    ENSEMBLE_RUNS,
    EVAL_DIR,
    EXTERNAL_MODELS,
    FEWSHOT_PREFIX,
    GRPO_STAGING_INTERIM,
    KNN_TEST_DIR,
    LEGACY_ORACLE_CACHE_DIR,
    LLM_PRED_DIR,
    MEMORIZATION_DIR,
    METRICS,
    METRICS_CACHE_DIR,
    MissingDataReport,
    NOVEL_DATASET_PATHS,
    NOVEL_JSON,
    NOVEL_SPLIT_NAME,
    ORACLE_BACKBONES,
    ORACLE_CACHE_DIR,
    OUTPUT_DIR,
    QWEN35_SCALING_MAPPED_PNG,
    SCRIPT_DIR,
    STAGED_MODEL_PREDICTIONS,
    _FINAL_ANSWER_RE,
    categorize_model,
    is_random_split_eligible,
    save_figure_cache,
    short_label,
    write_report_and_captions,
)


ORACLE_FILTER_CACHE = ORACLE_CACHE_DIR / "oracle_filter_baselines.json"
LEGACY_ORACLE_FILTER_CACHE = LEGACY_ORACLE_CACHE_DIR / "oracle_filter_baselines.json"
ORACLE_FILTER_N_SHUFFLES = 5
ORACLE_FILTER_SEED = 42
CLASSIFIER_WANDB_DIR = Path("/cv/data/braid/debroue1/promptoptbase/outputs/wandb")
KNN_TRANSFER_MATRIX_DIR = SCRIPT_DIR.parent / "output_latent_biology" / "transfer_matrix"


def log_progress(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[generate_figures_data {timestamp}] {message}", flush=True)


def _mean_metric_from_runs(runs: List[Dict[str, Any]], metric: str) -> Optional[float]:
    values = [run.get(metric) for run in runs if run.get(metric) is not None]
    if not values:
        return None
    return float(np.mean(values))


def load_per_example_from_caches(metrics: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metrics_file in sorted(METRICS_CACHE_DIR.glob("*.json")):
        model_name = metrics_file.stem.replace("__", "/")
        with open(metrics_file) as handle:
            cache = json.load(handle)
        category = categorize_model(model_name)
        for example_key, entry in cache.items():
            runs = entry.get("metrics_per_run", [])
            if not runs:
                continue
            n_runs = len(runs)
            for metric in metrics:
                value = _mean_metric_from_runs(runs, metric)
                if value is None:
                    continue
                rows.append({
                    "example_key": example_key,
                    "model": model_name,
                    "category": category,
                    "metric": metric,
                    "value": value,
                    "n_runs": n_runs,
                })
    return pd.DataFrame(rows)


def _load_grpo_records(pred_dir: Path, split: str, step: str) -> List[Dict[str, Any]]:
    path = pred_dir / f"{step}_{split}_predictions.json"
    if not path.exists():
        candidates = sorted(pred_dir.glob(f"*_{split}_predictions.json"))
        if not candidates:
            return []
        path = candidates[0]
    with open(path) as handle:
        records = json.load(handle)
    for record in records:
        record["genes"] = [_FINAL_ANSWER_RE.sub("", gene).strip() for gene in record["genes"]]
    return records


def load_external_grpo(config: Dict[str, Any], metric: str) -> pd.DataFrame:
    pred_dir = Path(config["path"])
    step = config.get("step", "step130")
    rows: List[Dict[str, Any]] = []
    for split in ["val", "test"]:
        records = _load_grpo_records(pred_dir, split, step)
        for index, record in enumerate(records):
            score = record.get(metric)
            if score is not None:
                rows.append({"example_key": f"{split}:{index}", "metric": metric, "value": float(score)})
    return pd.DataFrame(rows)


def load_external_c2s(config: Dict[str, Any], metric: str) -> pd.DataFrame:
    pred_dir = Path(config["path"])
    rows: List[Dict[str, Any]] = []
    for split in ["val", "test"]:
        pred_file = pred_dir / f"{split}_predictions.json"
        if not pred_file.exists():
            continue
        with open(pred_file) as handle:
            data = json.load(handle)
        for index, prediction in enumerate(data.get("predictions", [])):
            score = prediction.get("metrics", {}).get(metric)
            if score is not None:
                rows.append({"example_key": f"{split}:{index}", "metric": metric, "value": float(score)})
    return pd.DataFrame(rows)


def compute_external_prediction_metrics(
    records: List[Dict[str, Any]],
    split: str,
    metrics: Sequence[str],
    gt_map: Dict[str, Dict[str, Any]],
    eval_k: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, prediction in enumerate(records):
        example_key = f"{split}:{index}"
        ground_truth = gt_map.get(example_key)
        if ground_truth is None:
            continue
        raw_genes = prediction.get("genes", [])
        cleaned_genes = [
            _FINAL_ANSWER_RE.sub("", gene).strip()
            for gene in raw_genes
            if gene and gene.strip()
        ]
        result = eval_k.evaluate(
            predicted_genes=cleaned_genes,
            ground_truth_genes=ground_truth["genes"],
            relevance_scores=ground_truth["relevance_scores"],
        )
        for metric_name in metrics:
            value = result.get(metric_name)
            if value is None:
                continue
            rows.append({
                "example_key": example_key,
                "metric": metric_name,
                "value": float(value),
            })
    return rows


def load_external_grpo_all_metrics(
    config: Dict[str, Any],
    metrics: Sequence[str],
) -> pd.DataFrame:
    from screensqa.benchmark.ranking_metrics import RankingMetrics
    from scripts.shared_utils import load_all_ground_truth

    pred_dir = Path(config["path"])
    step = config.get("step", "step130")
    gt_map, _ = load_all_ground_truth(DATASET_PATH)
    eval_k = RankingMetrics(k_values=[5, 10, 20, 50, 100], use_thresholded_scoring=True)
    rows: List[Dict[str, Any]] = []
    for split in ["val", "test"]:
        rows.extend(
            compute_external_prediction_metrics(
                records=_load_grpo_records(pred_dir, split, step),
                split=split,
                metrics=metrics,
                gt_map=gt_map,
                eval_k=eval_k,
            )
        )
    return pd.DataFrame(rows)


def load_external_c2s_all_metrics(
    config: Dict[str, Any],
    metrics: Sequence[str],
) -> pd.DataFrame:
    from screensqa.benchmark.ranking_metrics import RankingMetrics
    from scripts.shared_utils import load_all_ground_truth

    pred_dir = Path(config["path"])
    gt_map, _ = load_all_ground_truth(DATASET_PATH)
    eval_k = RankingMetrics(k_values=[5, 10, 20, 50, 100], use_thresholded_scoring=True)
    rows: List[Dict[str, Any]] = []
    for split in ["val", "test"]:
        pred_file = pred_dir / f"{split}_predictions.json"
        if not pred_file.exists():
            continue
        with open(pred_file) as handle:
            data = json.load(handle)
        records = []
        for prediction in data.get("predictions", []):
            runs = prediction.get("predictions", [])
            genes = runs[0] if runs else []
            records.append({"genes": genes})
        rows.extend(
            compute_external_prediction_metrics(
                records=records,
                split=split,
                metrics=metrics,
                gt_map=gt_map,
                eval_k=eval_k,
            )
        )
    return pd.DataFrame(rows)


def load_list_prediction_files_all_metrics(
    *,
    val_path: str,
    test_path: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    from screensqa.benchmark.ranking_metrics import RankingMetrics
    from scripts.shared_utils import load_all_ground_truth

    gt_map, _ = load_all_ground_truth(DATASET_PATH)
    eval_k = RankingMetrics(k_values=[5, 10, 20, 50, 100], use_thresholded_scoring=True)
    rows: List[Dict[str, Any]] = []
    split_to_path = {"val": Path(val_path), "test": Path(test_path)}
    for split, path in split_to_path.items():
        if not path.exists():
            continue
        with open(path) as handle:
            data = json.load(handle)
        rows.extend(
            compute_external_prediction_metrics(
                records=data,
                split=split,
                metrics=metrics,
                gt_map=gt_map,
                eval_k=eval_k,
            )
        )
    return pd.DataFrame(rows)


def load_knn_results(run_if_missing: bool = True) -> pd.DataFrame:
    knn_dir = KNN_TEST_DIR / "year_fold0"
    knn_file = knn_dir / "knn_results.json"
    if not knn_file.exists() and run_if_missing:
        import subprocess

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "latent_biology" / "kNN_test.py"),
                    "--split-type",
                    "year",
                    "--fold",
                    "0",
                ],
                check=True,
                cwd=str(SCRIPT_DIR.parent),
            )
        except Exception:
            pass
    if not knn_file.exists():
        return pd.DataFrame()

    with open(knn_file) as handle:
        results = json.load(handle)

    rows: List[Dict[str, Any]] = []
    knn_metrics = results.get("metrics", ["adjusted_ndcg@100"])
    for method, label in [("oracle", "Oracle kNN"), ("embedding", "Embedding kNN")]:
        method_data = results.get(method, {})
        for split in ("val", "test", "novel"):
            split_data = method_data.get(split)
            if split_data is None:
                if split == "test" and "scores" in method_data:
                    for index, score in enumerate(method_data["scores"]):
                        rows.append({
                            "model": label,
                            "example_key": f"test:{index}",
                            "value": float(score),
                            "metric": "adjusted_ndcg@100",
                            "category": "kNN",
                        })
                continue
            for metric in knn_metrics:
                metric_block = split_data.get(metric)
                if metric_block is None:
                    continue
                for index, score in enumerate(metric_block.get("scores", [])):
                    rows.append({
                        "model": label,
                        "example_key": f"{split}:{index}",
                        "value": float(score),
                        "metric": metric,
                        "category": "kNN",
                    })
    return pd.DataFrame(rows)


def load_knn_aggregate_plot_rows(
    knn_file: Path,
    split_layout: str,
    cohorts: Sequence[str] = ("val", "test", "novel"),
) -> List[Dict[str, Any]]:
    if not knn_file.exists():
        return []
    with open(knn_file) as handle:
        results = json.load(handle)

    def compute_missing_knn_metrics() -> Dict[str, Dict[str, Dict[str, float]]]:
        from screensqa.benchmark.ranking_metrics import RankingMetrics

        with open(KNN_TRANSFER_MATRIX_DIR / "screen_genes.pkl", "rb") as handle:
            screen_genes_data = pickle.load(handle)
        display_to_screen = {}
        for label, metadata, screen in zip(
            screen_genes_data["screen_labels"],
            screen_genes_data["screen_metadata"],
            screen_genes_data["screens"],
        ):
            direction = "reverse" if metadata["reverse"] else "forward"
            display_to_screen[f"{label} ({direction})"] = screen

        train_ground_truth = [
            display_to_screen[name]
            for name in results.get("train_screens", [])
            if name in display_to_screen
        ]
        eval_k = RankingMetrics(k_values=[5, 10, 20, 50, 100], use_thresholded_scoring=True)
        computed: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

        for method in ("oracle", "embedding"):
            method_data = results.get(method, {})
            for cohort in cohorts:
                cohort_data = method_data.get(cohort)
                target_names = results.get(f"{cohort}_screens", [])
                if cohort_data is None or not target_names:
                    continue
                target_ground_truth = [
                    display_to_screen[name]
                    for name in target_names
                    if name in display_to_screen
                ]
                matched_train_indices = cohort_data.get("matched_train_indices", [])
                per_metric: Dict[str, List[float]] = defaultdict(list)
                n_examples = min(len(target_ground_truth), len(matched_train_indices))
                for index in range(n_examples):
                    train_idx = matched_train_indices[index]
                    if train_idx >= len(train_ground_truth):
                        continue
                    source = train_ground_truth[train_idx]
                    target = target_ground_truth[index]
                    ranked_source = sorted(
                        zip(source["genes"], source["relevance_scores"]),
                        key=lambda pair: pair[1],
                        reverse=True,
                    )
                    predicted_genes = [gene for gene, _ in ranked_source[:100]]
                    metric_values = eval_k.evaluate(
                        predicted_genes=predicted_genes,
                        ground_truth_genes=target["genes"],
                        relevance_scores=target["relevance_scores"],
                    )
                    for metric in METRICS:
                        value = metric_values.get(metric)
                        if value is not None and not np.isnan(value):
                            per_metric[metric].append(float(value))
                computed[method][cohort] = {
                    metric: {
                        "avg": float(np.mean(values)),
                        "scores": values,
                    }
                    for metric, values in per_metric.items()
                    if values
                }
        return computed

    knn_metrics = results.get("metrics", ["adjusted_ndcg@100"])
    has_all_metrics = True
    for method in ("oracle", "embedding"):
        for cohort in cohorts:
            cohort_data = results.get(method, {}).get(cohort)
            if cohort_data is None:
                continue
            if not set(METRICS).issubset(set(cohort_data.keys())):
                has_all_metrics = False
                break
        if not has_all_metrics:
            break
    computed_metrics = compute_missing_knn_metrics() if not has_all_metrics else {}

    rows: List[Dict[str, Any]] = []
    for method, label in [("oracle", "Oracle kNN"), ("embedding", "Embedding kNN")]:
        method_data = results.get(method, {})
        for cohort in cohorts:
            split_data = method_data.get(cohort)
            if split_data is None and computed_metrics:
                split_data = computed_metrics.get(method, {}).get(cohort)
            if split_data is None:
                continue
            metric_names = list(dict.fromkeys([*knn_metrics, *METRICS]))
            for metric in metric_names:
                metric_block = split_data.get(metric)
                if metric_block is None and computed_metrics:
                    metric_block = computed_metrics.get(method, {}).get(cohort, {}).get(metric)
                if metric_block is None:
                    continue
                avg = metric_block.get("avg")
                if avg is None:
                    continue
                rows.append({
                    "model": label,
                    "category": "kNN",
                    "metric": metric,
                    "value": float(avg),
                    "cohort": cohort,
                    "split_layout": split_layout,
                })
    return rows


def build_split_key_sets() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    from scripts.shared_utils import (
        build_global_to_year_fold0_mapping,
        get_split_assignments,
        load_all_ground_truth,
    )

    gt_map, _ = load_all_ground_truth(DATASET_PATH)
    global_map = build_global_to_year_fold0_mapping(gt_map)
    year_assignments = get_split_assignments(DATASET_PATH, global_map, "year", 0)
    random_assignments = get_split_assignments(DATASET_PATH, global_map, "random", 0)
    year_sets = {split: set(year_assignments[split]) for split in ["train", "val", "test"]}
    random_sets = {split: set(random_assignments[split]) for split in ["train", "val", "test"]}
    return year_sets, random_sets


def oracle_rerank_genes(
    pred_genes: List[str],
    gt_genes: List[str],
    relevance_scores: List[float],
    gene_mapper: Any = None,
) -> List[str]:
    def normalize(gene: str) -> str:
        if gene_mapper is not None:
            mapped = gene_mapper.map_gene(gene.strip().upper())
            if mapped is not None:
                return mapped
        return gene.strip().upper()

    score_map = {normalize(gene): float(score) for gene, score in zip(gt_genes, relevance_scores)}
    cleaned = [_FINAL_ANSWER_RE.sub("", gene).strip() for gene in pred_genes if gene.strip()]
    return sorted(cleaned, key=lambda gene: score_map.get(normalize(gene), 0.0), reverse=True)


def _oracle_cache_path(backbone: str) -> Path:
    return ORACLE_CACHE_DIR / f"{backbone.replace('/', '__')}.json"


def _legacy_oracle_cache_path(backbone: str) -> Path:
    return LEGACY_ORACLE_CACHE_DIR / f"{backbone.replace('/', '__')}.json"


def _oracle_eval_backbone(
    backbone: str,
    gt_map: Dict[str, Dict[str, Any]],
    eval_k: Any,
    metrics: Sequence[str],
    additional_splits: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    from scripts.shared_utils import load_all_model_predictions

    gene_mapper = getattr(eval_k, "gene_mapper", None)
    pred_map = load_all_model_predictions(
        LLM_PRED_DIR.parent,
        backbone,
        predictions_subdir="llm_predictions",
        additional_splits=additional_splits,
    )

    entries: Dict[str, Dict[str, float]] = {}
    for example_key, ground_truth in gt_map.items():
        if example_key not in pred_map:
            continue
        runs = pred_map[example_key].get("predictions", [])
        if not runs or not runs[0]:
            continue
        reranked = oracle_rerank_genes(
            runs[0],
            ground_truth["genes"],
            ground_truth["relevance_scores"],
            gene_mapper=gene_mapper,
        )
        result = eval_k.evaluate(
            predicted_genes=reranked,
            ground_truth_genes=ground_truth["genes"],
            relevance_scores=ground_truth["relevance_scores"],
        )
        entry = {metric: float(result[metric]) for metric in metrics if result.get(metric) is not None}
        if entry:
            entries[example_key] = entry
    return entries


def _oracle_eval_combined(
    backbone_models: List[str],
    gt_map: Dict[str, Dict[str, Any]],
    eval_k: Any,
    metrics: Sequence[str],
    additional_splits: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    from scripts.shared_utils import load_all_model_predictions

    gene_mapper = getattr(eval_k, "gene_mapper", None)

    def normalize(gene: str) -> str:
        if gene_mapper is not None:
            mapped = gene_mapper.map_gene(gene.strip().upper())
            if mapped is not None:
                return mapped
        return gene.strip().upper()

    pred_maps = {
        backbone: load_all_model_predictions(
            LLM_PRED_DIR.parent,
            backbone,
            predictions_subdir="llm_predictions",
            additional_splits=additional_splits,
        )
        for backbone in backbone_models
    }

    entries: Dict[str, Dict[str, float]] = {}
    for example_key, ground_truth in gt_map.items():
        all_genes: List[str] = []
        seen: set[str] = set()
        for backbone in backbone_models:
            runs = pred_maps[backbone].get(example_key, {}).get("predictions", [])
            if not runs or not runs[0]:
                continue
            for gene in runs[0]:
                cleaned = _FINAL_ANSWER_RE.sub("", gene).strip()
                if cleaned and normalize(cleaned) not in seen:
                    seen.add(normalize(cleaned))
                    all_genes.append(cleaned)
        if not all_genes:
            continue
        reranked = oracle_rerank_genes(
            all_genes,
            ground_truth["genes"],
            ground_truth["relevance_scores"],
            gene_mapper=gene_mapper,
        )
        result = eval_k.evaluate(
            predicted_genes=reranked,
            ground_truth_genes=ground_truth["genes"],
            relevance_scores=ground_truth["relevance_scores"],
        )
        entry = {metric: float(result[metric]) for metric in metrics if result.get(metric) is not None}
        if entry:
            entries[example_key] = entry
    return entries


def _oracle_combined_cache_path(backbone_models: List[str]) -> Path:
    safe = "__".join(sorted(backbone.replace("/", "_") for backbone in backbone_models))
    return ORACLE_CACHE_DIR / f"combined__{safe}.json"


def _legacy_oracle_combined_cache_path(backbone_models: List[str]) -> Path:
    safe = "__".join(sorted(backbone.replace("/", "_") for backbone in backbone_models))
    return LEGACY_ORACLE_CACHE_DIR / f"combined__{safe}.json"


def _load_json_dict(path: Path) -> Dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def _load_oracle_cache_with_fallback(
    cache_path: Path,
    legacy_path: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    if cache_path.exists():
        return _load_json_dict(cache_path), cache_path
    if legacy_path is not None and legacy_path.exists():
        log_progress(f"Seeding local oracle cache from legacy file {legacy_path}")
        return _load_json_dict(legacy_path), legacy_path
    return None, None


def _write_local_oracle_cache(cache_path: Path, data: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as handle:
        json.dump(data, handle)


def compute_oracle_filter_baselines(
    gt_map: Dict[str, Dict[str, Any]],
    eval_k: Any,
    metrics: Sequence[str],
    n_shuffles: int = ORACLE_FILTER_N_SHUFFLES,
    seed: int = ORACLE_FILTER_SEED,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    rng = random_module.Random(seed)
    entries: Dict[str, Dict[str, Dict[str, float]]] = {}

    for example_key, ground_truth in gt_map.items():
        genes = ground_truth["genes"]
        scores = ground_truth["relevance_scores"]
        hit_genes = [gene for gene, score in zip(genes, scores) if score > 0]
        neg_genes = [gene for gene, score in zip(genes, scores) if score < 0]
        non_neg_genes = [gene for gene, score in zip(genes, scores) if score >= 0]

        entry: Dict[str, Dict[str, float]] = {}
        for label, gene_groups in [("hit_filter", [hit_genes]), ("neg_filter", [neg_genes, non_neg_genes])]:
            flat = [gene for group in gene_groups for gene in group]
            if not flat:
                continue
            accum: Dict[str, List[float]] = defaultdict(list)
            for _ in range(n_shuffles):
                shuffled_genes: List[str] = []
                for group in gene_groups:
                    shuffled = list(group)
                    rng.shuffle(shuffled)
                    shuffled_genes.extend(shuffled)
                result = eval_k.evaluate(
                    predicted_genes=shuffled_genes,
                    ground_truth_genes=genes,
                    relevance_scores=scores,
                )
                for metric in metrics:
                    value = result.get(metric)
                    if value is not None:
                        accum[metric].append(float(value))
            if accum:
                entry[label] = {metric: float(np.mean(values)) for metric, values in accum.items()}
        if entry:
            entries[example_key] = entry

    return entries


def _oracle_cache_has_metrics(
    cached: Dict[str, Dict[str, float]],
    metrics: Sequence[str],
) -> bool:
    if not cached:
        return False
    for metric_values in cached.values():
        if not metric_values:
            continue
        if not set(metrics).issubset(set(metric_values.keys())):
            return False
    return True


def _oracle_filter_cache_has_metrics(
    cached: Dict[str, Dict[str, Dict[str, float]]],
    metrics: Sequence[str],
) -> bool:
    if not cached:
        return False
    for by_label in cached.values():
        for metric_values in by_label.values():
            if not set(metrics).issubset(set(metric_values.keys())):
                return False
    return True


def compute_oracle_rerank_metrics(
    backbone_models: List[str],
    metrics: Sequence[str],
    report: MissingDataReport,
    allowed_keys: Optional[Set[str]] = None,
) -> pd.DataFrame:
    from screensqa.benchmark.ranking_metrics import RankingMetrics
    from scripts.shared_utils import load_additional_ground_truth, load_all_ground_truth

    ORACLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    gt_map, _ = load_all_ground_truth(DATASET_PATH)
    novel_gt = load_additional_ground_truth(
        split_name=NOVEL_SPLIT_NAME,
        dataset_paths=NOVEL_DATASET_PATHS,
        display_library_genes=True,
    )
    combined_gt = {**gt_map, **novel_gt}
    eval_k = RankingMetrics(k_values=[5, 10, 20, 50, 100], use_thresholded_scoring=True)
    rows: List[Dict[str, Any]] = []

    def append_cached_rows(cached: Dict[str, Dict[str, float]], label: str) -> None:
        for example_key, metric_values in cached.items():
            if allowed_keys is not None and example_key not in allowed_keys:
                continue
            for metric in metrics:
                value = metric_values.get(metric)
                if value is None:
                    continue
                rows.append({
                    "example_key": example_key,
                    "model": label,
                    "category": "Oracle rerank",
                    "metric": metric,
                    "value": float(value),
                    "n_runs": 1,
                })

    for backbone in backbone_models:
        label = f"Oracle rerank ({short_label(backbone)})"
        cache_path = _oracle_cache_path(backbone)
        legacy_cache_path = _legacy_oracle_cache_path(backbone)
        cached, cache_source = _load_oracle_cache_with_fallback(cache_path, legacy_cache_path)
        if cached is not None:
            has_novel = any(key.startswith(f"{NOVEL_SPLIT_NAME}:") for key in cached)
            has_all_metrics = _oracle_cache_has_metrics(cached, metrics)
            if not has_novel or not has_all_metrics:
                refreshed = _oracle_eval_backbone(
                    backbone,
                    combined_gt,
                    eval_k,
                    metrics,
                    additional_splits=[NOVEL_SPLIT_NAME],
                )
                cached.update(refreshed)
            if cache_source != cache_path or not has_novel or not has_all_metrics:
                _write_local_oracle_cache(cache_path, cached)
            append_cached_rows(cached, label)
            continue

        cache_entries = _oracle_eval_backbone(
            backbone,
            combined_gt,
            eval_k,
            metrics,
            additional_splits=[NOVEL_SPLIT_NAME],
        )
        _write_local_oracle_cache(cache_path, cache_entries)
        append_cached_rows(cache_entries, label)

    combined_label = "Oracle rerank (combined)"
    combined_cache = _oracle_combined_cache_path(backbone_models)
    legacy_combined_cache = _legacy_oracle_combined_cache_path(backbone_models)
    cached_combined, combined_source = _load_oracle_cache_with_fallback(
        combined_cache,
        legacy_combined_cache,
    )
    if cached_combined is not None:
        has_novel = any(key.startswith(f"{NOVEL_SPLIT_NAME}:") for key in cached_combined)
        has_all_metrics = _oracle_cache_has_metrics(cached_combined, metrics)
        if not has_novel or not has_all_metrics:
            refreshed = _oracle_eval_combined(
                backbone_models,
                combined_gt,
                eval_k,
                metrics,
                additional_splits=[NOVEL_SPLIT_NAME],
            )
            cached_combined.update(refreshed)
        if combined_source != combined_cache or not has_novel or not has_all_metrics:
            _write_local_oracle_cache(combined_cache, cached_combined)
        append_cached_rows(cached_combined, combined_label)
    else:
        combined_entries = _oracle_eval_combined(
            backbone_models,
            combined_gt,
            eval_k,
            metrics,
            additional_splits=[NOVEL_SPLIT_NAME],
        )
        _write_local_oracle_cache(combined_cache, combined_entries)
        append_cached_rows(combined_entries, combined_label)

    filter_labels = {
        "hit_filter": "Oracle hit filter",
        "neg_filter": "Oracle negative filter",
    }
    cached_filters, filter_source = _load_oracle_cache_with_fallback(
        ORACLE_FILTER_CACHE,
        LEGACY_ORACLE_FILTER_CACHE,
    )
    if cached_filters is not None:
        has_novel = any(key.startswith(f"{NOVEL_SPLIT_NAME}:") for key in cached_filters)
        has_all_metrics = _oracle_filter_cache_has_metrics(cached_filters, metrics)
        if not has_novel or not has_all_metrics:
            refreshed = compute_oracle_filter_baselines(combined_gt, eval_k, metrics)
            cached_filters.update(refreshed)
        if filter_source != ORACLE_FILTER_CACHE or not has_novel or not has_all_metrics:
            _write_local_oracle_cache(ORACLE_FILTER_CACHE, cached_filters)
        filter_data = cached_filters
    else:
        filter_data = compute_oracle_filter_baselines(combined_gt, eval_k, metrics)
        _write_local_oracle_cache(ORACLE_FILTER_CACHE, filter_data)

    for example_key, by_label in filter_data.items():
        if allowed_keys is not None and example_key not in allowed_keys:
            continue
        for short_key, metric_values in by_label.items():
            label = filter_labels.get(short_key)
            if label is None:
                continue
            for metric in metrics:
                value = metric_values.get(metric)
                if value is None:
                    continue
                rows.append({
                    "example_key": example_key,
                    "model": label,
                    "category": "Oracle rerank",
                    "metric": metric,
                    "value": float(value),
                    "n_runs": 1,
                })

    return pd.DataFrame(rows)


def _ensure_csv_field_limit() -> None:
    limit = min(2**31 - 1, sys.maxsize)
    while limit > 1024 * 1024:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def _extract_csv_run_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    _ensure_csv_field_limit()
    run_ids: List[str] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_id = (row.get("run_id") or "").strip()
            if run_id:
                run_ids.append(run_id)
    return sorted(set(run_ids))


def _find_classifier_run_dir(run_id: str) -> Optional[Path]:
    matches = sorted(CLASSIFIER_WANDB_DIR.glob(f"run-*-{run_id}"))
    return matches[0] if matches else None


def _load_classifier_split_examples(
    *,
    dataset_path: str,
    split_type: str,
    fold: int,
    cache: Dict[Tuple[str, str, int], Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, List[Dict[str, Any]]]:
    from screensqa.dataset.dataset import BioGRIDDSPY

    cache_key = (dataset_path, split_type, int(fold))
    if cache_key not in cache:
        dataset = BioGRIDDSPY(
            dataset_path=dataset_path,
            split_type=split_type,
            fold=int(fold),
        )
        _, val_examples, test_examples = dataset.get_train_test_split()
        cache[cache_key] = {"val": val_examples, "test": test_examples}
    return cache[cache_key]


def _load_classifier_predictions(path: Path) -> List[List[str]]:
    with open(path) as handle:
        data = json.load(handle)
    predictions = data.get("predictions", []) if isinstance(data, dict) else data
    gene_lists: List[List[str]] = []
    for prediction in predictions:
        runs = prediction.get("predictions", []) if isinstance(prediction, dict) else []
        genes = runs[0] if runs else []
        gene_lists.append([
            _FINAL_ANSWER_RE.sub("", gene).strip()
            for gene in genes
            if gene and gene.strip()
        ])
    return gene_lists


def _compute_classifier_metrics_from_runs(
    run_ids: Sequence[str],
    report: MissingDataReport,
    split_layout: str,
    metrics: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    from screensqa.benchmark.ranking_metrics import RankingMetrics
    import yaml

    needed_metrics = [metric for metric in metrics if metric in METRICS]
    if not needed_metrics:
        return {}

    eval_k = RankingMetrics(k_values=[5, 10, 20, 50, 100], use_thresholded_scoring=True)
    examples_cache: Dict[Tuple[str, str, int], Dict[str, List[Dict[str, Any]]]] = {}
    accum: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for run_id in run_ids:
        run_dir = _find_classifier_run_dir(run_id)
        if run_dir is None:
            report.add(
                model="Classifier",
                split_layout=split_layout,
                cohort="*",
                metric="*",
                status="missing",
                reason=f"wandb run directory missing for {run_id}",
            )
            continue

        config_path = run_dir / "files" / "config.yaml"
        if not config_path.exists():
            report.add(
                model="Classifier",
                split_layout=split_layout,
                cohort="*",
                metric="*",
                status="missing",
                reason=f"classifier config missing for {run_id}",
            )
            continue

        config = yaml.safe_load(config_path.read_text())
        data_cfg = config.get("data", {}).get("value", {})
        dataset_path = data_cfg.get("dataset_path", DATASET_PATH)
        run_split_type = data_cfg.get("split_type", split_layout)
        run_fold = int(data_cfg.get("fold", 0))
        split_examples = _load_classifier_split_examples(
            dataset_path=dataset_path,
            split_type=run_split_type,
            fold=run_fold,
            cache=examples_cache,
        )

        for split in ("val", "test"):
            pred_path = run_dir / "files" / "predictions" / f"{split}_predictions.json"
            if not pred_path.exists():
                report.add(
                    model="Classifier",
                    split_layout=split_layout,
                    cohort=split,
                    metric="*",
                    status="missing",
                    reason=f"classifier predictions missing for {run_id}",
                )
                continue

            gene_lists = _load_classifier_predictions(pred_path)
            examples = split_examples.get(split, [])
            n_examples = min(len(gene_lists), len(examples))
            for index in range(n_examples):
                result = eval_k.evaluate(
                    predicted_genes=gene_lists[index],
                    ground_truth_genes=examples[index]["relevance_genes"],
                    relevance_scores=examples[index]["relevance_scores"],
                )
                for metric in needed_metrics:
                    value = result.get(metric)
                    if value is not None and not np.isnan(value):
                        accum[metric][split].append(float(value))

    return {
        metric: {
            split: float(np.mean(values))
            for split, values in split_dict.items()
            if values
        }
        for metric, split_dict in accum.items()
    }


def load_classifier_series(
    path: Path,
    report: MissingDataReport,
    cohort_label: str,
) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        report.add(
            model="Classifier",
            split_layout="any",
            cohort=cohort_label,
            metric="*",
            status="missing",
            reason=f"file missing: {path}",
        )
        return {}
    _ensure_csv_field_limit()
    accum: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            split = (row.get("split") or "").strip().lower()
            if split not in ("val", "test"):
                continue
            for metric in METRICS:
                if metric not in fieldnames:
                    continue
                try:
                    accum[metric][split].append(float(row[metric]))
                except (TypeError, ValueError):
                    pass
    missing_metrics = [metric for metric in METRICS if metric not in fieldnames]
    if missing_metrics:
        run_ids = _extract_csv_run_ids(path)
        raw_metric_means = _compute_classifier_metrics_from_runs(
            run_ids,
            report,
            cohort_label,
            missing_metrics,
        )
        for metric, split_dict in raw_metric_means.items():
            for split, value in split_dict.items():
                accum[metric][split].append(value)
        for metric in missing_metrics:
            if metric in raw_metric_means:
                continue
            for split in ("val", "test"):
                report.add(
                    model="Classifier",
                    split_layout=cohort_label,
                    cohort=split,
                    metric=metric,
                    status="missing",
                    reason="metric absent from CSV and raw predictions unavailable",
                )

    return {
        metric: {split: float(np.mean(values)) for split, values in by_split.items() if values}
        for metric, by_split in accum.items()
    }


def load_classifier_novel_recent(path: Path, report: MissingDataReport) -> Dict[str, float]:
    if not path.exists():
        report.add(
            model="Classifier",
            split_layout="novel",
            cohort="novel",
            metric="*",
            status="missing",
            reason=f"file missing: {path}",
        )
        return {}
    _ensure_csv_field_limit()
    metrics_out: Dict[str, List[float]] = defaultdict(list)
    fieldnames: List[str] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            for metric in METRICS:
                if metric not in fieldnames:
                    continue
                try:
                    metrics_out[metric].append(float(row[metric]))
                except (TypeError, ValueError):
                    pass
    for metric in METRICS:
        if metric not in fieldnames:
            report.add(
                model="Classifier",
                split_layout="novel",
                cohort="novel",
                metric=metric,
                status="missing",
                reason="column absent",
            )
    return {
        metric: float(np.mean(values)) if values else float("nan")
        for metric, values in metrics_out.items()
    }


def load_novel_public_llm_ndcg(path: Path, report: MissingDataReport) -> Dict[str, float]:
    if not path.exists():
        report.add(
            model="*",
            split_layout="novel",
            cohort="novel",
            metric="adjusted_ndcg@100",
            status="missing",
            reason=f"novel JSON missing: {path}",
        )
        return {}
    with open(path) as handle:
        data = json.load(handle)
    return {
        metric: value["mean"]
        for metric, value in data.get("aggregate", {}).items()
        if isinstance(value, dict) and "mean" in value
    }


def merge_classifier_into_frame(
    plot_df: pd.DataFrame,
    cls_year: Dict[str, Dict[str, float]],
    cls_random: Dict[str, Dict[str, float]],
    cls_novel: Dict[str, float],
) -> pd.DataFrame:
    if plot_df.empty:
        plot_df = pd.DataFrame(columns=["model", "category", "metric", "value", "cohort", "split_layout"])
    rows: List[Dict[str, Any]] = []
    for metric, splits in cls_year.items():
        for split, value in splits.items():
            rows.append({
                "model": "Classifier",
                "category": "Classifier",
                "metric": metric,
                "value": value,
                "cohort": split,
                "split_layout": "year",
            })
    for metric, splits in cls_random.items():
        for split, value in splits.items():
            rows.append({
                "model": "Classifier",
                "category": "Classifier",
                "metric": metric,
                "value": value,
                "cohort": split,
                "split_layout": "random",
            })
    for metric, value in cls_novel.items():
        if np.isnan(value):
            continue
        for split_layout in ("year", "random"):
            rows.append({
                "model": "Classifier",
                "category": "Classifier",
                "metric": metric,
                "value": value,
                "cohort": "novel",
                "split_layout": split_layout,
            })
    if not rows:
        return plot_df
    return pd.concat([plot_df, pd.DataFrame(rows)], ignore_index=True)


def _best_trial_from_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    best = None
    best_val = -1.0
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            trial = json.loads(line)
            val_score = trial.get("val_score")
            if val_score is None:
                continue
            if float(val_score) > best_val:
                best_val = float(val_score)
                best = trial
    return best


def _ensemble_file_has_metrics(
    data: Dict[str, Any],
    metrics: Sequence[str],
    cohorts: Sequence[str] = ("val", "test", "novel"),
) -> bool:
    splits = data.get("splits", {})
    for cohort in cohorts:
        cohort_metrics = splits.get(cohort, {})
        if not set(metrics).issubset(set(cohort_metrics.keys())):
            return False
    return True


def _build_best_ensemble_fn(params: Dict[str, Any], model_list: Sequence[str]):
    def ensemble_predictions(
        all_model_predictions: Dict[str, List[List[str]]],
        top_k: int = 100,
    ) -> List[str]:
        gene_scores: Dict[str, float] = defaultdict(float)
        gene_model_count: Dict[str, int] = defaultdict(int)

        k_rrf = params.get("k_rrf", 60.0)
        agreement_bonus_2 = params.get("agreement_bonus_2", 1.0)
        agreement_bonus_3 = params.get("agreement_bonus_3", 1.0)
        agreement_threshold = params.get("agreement_threshold", 5)

        model_weights = {}
        for model_name in model_list:
            safe = model_name.replace("-", "_").replace(".", "_")
            model_weights[model_name] = params.get(f"w_{safe}", 1.0)

        for model_name, runs in all_model_predictions.items():
            weight = model_weights.get(model_name, 1.0)
            if weight < 0.01:
                continue
            for run in runs:
                for rank, gene in enumerate(run):
                    cleaned = gene.strip().upper()
                    if cleaned:
                        gene_scores[cleaned] += weight / (k_rrf + rank + 1)
            genes_from_model = set()
            for run in runs:
                for gene in run[:top_k]:
                    cleaned = gene.strip().upper()
                    if cleaned:
                        genes_from_model.add(cleaned)
            for gene in genes_from_model:
                gene_model_count[gene] += 1

        for gene, count in gene_model_count.items():
            if count >= agreement_threshold:
                if count >= 3:
                    gene_scores[gene] *= agreement_bonus_3
                elif count >= 2:
                    gene_scores[gene] *= agreement_bonus_2

        return sorted(gene_scores, key=gene_scores.get, reverse=True)[:top_k]

    return ensemble_predictions


def _compute_best_ensemble_all_metrics(
    report: MissingDataReport,
) -> Dict[str, Any]:
    from screensqa.benchmark.ranking_metrics import RankingMetrics
    from scripts.shared_utils import load_additional_ground_truth, load_all_ground_truth, load_all_model_predictions

    existing: Dict[str, Any] = {}
    if BEST_ENSEMBLE_ALL_METRICS.exists():
        with open(BEST_ENSEMBLE_ALL_METRICS) as handle:
            existing = json.load(handle)
        if _ensemble_file_has_metrics(existing, METRICS):
            return existing

    params = existing.get("params")
    model_list = existing.get("model_list")
    if params is None or model_list is None:
        best_trial = None
        best_val = -1.0
        for _, path in ENSEMBLE_RUNS:
            trial = _best_trial_from_jsonl(path)
            if trial is None:
                continue
            val_score = float(trial.get("val_score", -1.0))
            if val_score > best_val:
                best_val = val_score
                best_trial = trial
        if best_trial is None:
            report.add(
                model="LLM RRF Ensemble",
                split_layout="year",
                cohort="*",
                metric="*",
                status="missing",
                reason="no Bayesian trials found for all-metrics recomputation",
            )
            return existing
        params = best_trial["params"]
        model_list = ["gemini-3-pro", "gpt-5.4", "gemini-3-flash"]

    log_progress("Recomputing all ensemble metrics, including normalized precision variants")
    ensemble_fn = _build_best_ensemble_fn(params, model_list)
    eval_k = RankingMetrics(k_values=[100], use_thresholded_scoring=True)

    gt_map, _ = load_all_ground_truth(DATASET_PATH)
    novel_gt = load_additional_ground_truth(
        split_name=NOVEL_SPLIT_NAME,
        dataset_paths=NOVEL_DATASET_PATHS,
        display_library_genes=True,
    )
    combined_gt = {**gt_map, **novel_gt}
    pred_maps = {
        model_name: load_all_model_predictions(
            LLM_PRED_DIR.parent,
            model_name,
            predictions_subdir="llm_predictions",
            additional_splits=[NOVEL_SPLIT_NAME],
        )
        for model_name in model_list
    }

    split_keys = {
        "val": [key for key in combined_gt if key.startswith("val:")],
        "test": [key for key in combined_gt if key.startswith("test:")],
        "novel": [key for key in combined_gt if key.startswith(f"{NOVEL_SPLIT_NAME}:")],
    }

    output = {
        "config": existing.get("config", "top3_optimize_all"),
        "params": params,
        "model_list": model_list,
        "original_val_score": existing.get("original_val_score"),
        "original_test_score": existing.get("original_test_score"),
        "splits": {},
    }
    for cohort, keys in split_keys.items():
        per_metric: Dict[str, List[float]] = defaultdict(list)
        for example_key in keys:
            example_predictions = {}
            for model_name, pred_map in pred_maps.items():
                runs = pred_map.get(example_key, {}).get("predictions", [])
                if runs:
                    example_predictions[model_name] = runs
            if not example_predictions:
                continue
            gt = combined_gt[example_key]
            predicted_genes = ensemble_fn(example_predictions, top_k=100)
            result = eval_k.evaluate(
                predicted_genes=predicted_genes,
                ground_truth_genes=gt["genes"],
                relevance_scores=gt["relevance_scores"],
            )
            for metric in METRICS:
                value = result.get(metric)
                if value is not None and not np.isnan(value):
                    per_metric[metric].append(float(value))
        output["splits"][cohort] = {
            metric: float(np.mean(values)) if values else None
            for metric, values in per_metric.items()
        }

    try:
        with open(BEST_ENSEMBLE_ALL_METRICS, "w") as handle:
            json.dump(output, handle, indent=2)
    except OSError:
        pass
    return output


def load_best_ensemble_rows(report: MissingDataReport) -> List[Dict[str, Any]]:
    data = _compute_best_ensemble_all_metrics(report)
    if data:
        rows: List[Dict[str, Any]] = []
        for cohort, metrics in data.get("splits", {}).items():
            for metric, value in metrics.items():
                if value is None:
                    continue
                rows.append({
                    "model": "LLM RRF Ensemble",
                    "category": "Ensemble",
                    "metric": metric,
                    "value": float(value),
                    "cohort": cohort,
                    "split_layout": "year",
                })
        if rows:
            return rows
        report.add(
            model="LLM RRF Ensemble",
            split_layout="year",
            cohort="*",
            metric="*",
            status="missing",
            reason=f"all-metrics JSON empty: {BEST_ENSEMBLE_ALL_METRICS}",
        )

    best_trial = None
    best_val = -1.0
    for _, path in ENSEMBLE_RUNS:
        trial = _best_trial_from_jsonl(path)
        if trial is None:
            continue
        val_score = float(trial.get("val_score", -1.0))
        if val_score > best_val:
            best_val = val_score
            best_trial = trial
    if best_trial is None:
        report.add(
            model="LLM RRF Ensemble",
            split_layout="year",
            cohort="*",
            metric="adjusted_ndcg@100",
            status="missing",
            reason="no Bayesian trials found",
        )
        return []
    rows = []
    for cohort, key in [("val", "val_score"), ("test", "test_score")]:
        value = best_trial.get(key)
        if value is None:
            continue
        rows.append({
            "model": "LLM RRF Ensemble",
            "category": "Ensemble",
            "metric": "adjusted_ndcg@100",
            "value": float(value),
            "cohort": cohort,
            "split_layout": "year",
        })
    return rows


def load_plot3_trial_rows(report: MissingDataReport) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, path in ENSEMBLE_RUNS:
        best_trial = _best_trial_from_jsonl(path)
        if best_trial is None:
            report.add(
                model=label,
                split_layout="ensemble",
                cohort="*",
                metric="adjusted_ndcg@100",
                status="missing",
                reason=f"no trials: {path}",
            )
            continue
        rows.append({
            "label": label,
            "val_score": float(best_trial.get("val_score", float("nan"))),
            "test_score": float(best_trial.get("test_score", float("nan"))),
        })
    return rows


def load_metadata() -> pd.DataFrame:
    from datasets import load_from_disk
    from screensqa.utils.biogrid_maps import stratify_metrics_by_dataset_name

    dataset = load_from_disk(DATASET_PATH)
    rows: List[Dict[str, Any]] = []
    split_counters: Dict[str, int] = defaultdict(int)
    key_map = {"train": "train", "validation": "val", "test": "test"}
    for row in dataset:
        split = row["yearfold0"]
        index = split_counters[split]
        split_counters[split] += 1
        rows.append({
            "example_key": f"{key_map[split]}:{index}",
            "dataset_name": row["dataset_name"],
            "split": key_map[split],
            "screen_type": row["screen_type"],
            "library_methodology": row["library_methodology"],
            "num_genes": len(row["relevance_genes"]),
        })
    meta_df = pd.DataFrame(rows)
    phenotype_df = stratify_metrics_by_dataset_name({row["dataset_name"]: 0 for row in rows})
    phenotype_df = phenotype_df.rename(columns={
        "phenotype": "biogrid_phenotype",
        "screen_rationale": "biogrid_screen_rationale",
    })
    phenotype_df.index.name = "dataset_name"
    phenotype_df = phenotype_df[["biogrid_phenotype", "biogrid_screen_rationale"]].reset_index()
    return meta_df.merge(phenotype_df, on="dataset_name", how="left")


def _select_models_phenotype(
    summary: pd.DataFrame,
    per_ex: pd.DataFrame,
    layout: str,
) -> List[str]:
    if layout == "year":
        selected: List[str] = []
        llm = summary[summary["category"] == "LLM"].nlargest(3, "mean")
        selected.extend(llm["model"].tolist())
        biomni = summary[summary["category"] == "Biomni (agent)"]
        if not biomni.empty:
            selected.append(biomni.nlargest(1, "mean")["model"].iloc[0])
        for category in ["GEPA", "LLM (trained)", "kNN", "Baseline (statistical)"]:
            subset = summary[summary["category"] == category]
            if not subset.empty:
                selected.append(subset.nlargest(1, "mean")["model"].iloc[0])
        oracle = summary[summary["category"] == "Oracle rerank"]
        if not oracle.empty:
            selected.append(oracle.nlargest(1, "mean")["model"].iloc[0])
        if "Classifier" in per_ex["model"].values:
            selected.append("Classifier")
    else:
        selected = [
            model
            for model in per_ex["model"].unique()
            if categorize_model(model) in ("LLM", "Biomni (agent)", "Classifier", "Oracle rerank")
        ][:12]
    seen = set()
    output = []
    for model in selected:
        if model not in seen:
            seen.add(model)
            output.append(model)
    return output


def resolve_grpo_rows(
    grpo_staging_path: Optional[str],
    report: MissingDataReport,
) -> List[Dict[str, Any]]:
    rows = list(GRPO_STAGING_INTERIM)
    if not grpo_staging_path:
        return rows
    path = Path(grpo_staging_path)
    if not path.exists():
        report.add(
            model="GRPO table",
            split_layout="config",
            cohort="*",
            metric="*",
            status="missing",
            reason=f"--grpo-staging not found: {path}",
        )
        return rows
    report.grpo_path = str(path)
    if path.suffix.lower() == ".json":
        with open(path) as handle:
            rows = json.load(handle)
        report.used_grpo_interim = False
        return rows
    report.add(
        model="GRPO table",
        split_layout="config",
        cohort="*",
        metric="*",
        status="partial",
        reason=f"non-JSON grpo path; using interim table: {path}",
    )
    return rows


def load_plot4_panels(
    report: MissingDataReport,
    scaling_png: Optional[str] = None,
) -> List[Dict[str, Any]]:
    scale_path = Path(scaling_png) if scaling_png else QWEN35_SCALING_MAPPED_PNG
    panel_specs = [
        (scale_path, "Qwen3.5 scaling (mapped nDCG@100)"),
        (MEMORIZATION_DIR / "test6_phenotype_stratified.png", "Phenotype-stratified (memorization analysis)"),
        (MEMORIZATION_DIR / "test7_citation_analysis.png", "Citations vs performance"),
        (MEMORIZATION_DIR / "test8_classifier_comparison.png", "Classifier vs LLM"),
    ]
    panels: List[Dict[str, Any]] = []
    for path, title in panel_specs:
        if not path.exists():
            report.add(
                model="plot4",
                split_layout="plot4",
                cohort="*",
                metric="*",
                status="missing",
                reason=f"missing {path}",
            )
            continue
        panels.append({"title": title, "image": mpimg.imread(path), "source_path": str(path)})
    return panels


def compute_duplicate_transfer_df(report: MissingDataReport) -> pd.DataFrame:
    from screensqa.benchmark.ranking_metrics import RankingMetrics
    from screensqa.dataset.dataset import BioGRIDDSPY
    from screensqa.utils.biogrid_maps import stratify_metrics_by_dataset_name

    duplicate_singletons_path = "/cv/data/braid/gnesys/datasets/screensQA/biogrid_v0.5"

    combined_ds = BioGRIDDSPY(
        dataset_path=DATASET_PATH,
        split_type=None,
        fold=0,
    )
    log_progress("Loading duplicate-screen source examples from BioGRID v0.4 combined")
    _, _, combined_examples = combined_ds.get_train_test_split()
    duplicate_pairs = [
        (example["dataset_name"], example["screen_ids"])
        for example in combined_examples
        if len(example.get("screen_ids", [])) > 1
    ]
    if not duplicate_pairs:
        report.add(
            model="plot5",
            split_layout="plot5",
            cohort="*",
            metric="adjusted_ndcg@100",
            status="missing",
            reason="no duplicate merged screens found in v0.4 combined",
        )
        return pd.DataFrame()

    phenotype_df = stratify_metrics_by_dataset_name(
        {dataset_name: 0 for dataset_name, _ in duplicate_pairs}
    )
    phenotype_map = phenotype_df.set_index("dataset_name")["phenotype"].to_dict()

    singleton_ds = BioGRIDDSPY(
        dataset_path=duplicate_singletons_path,
        split_type=None,
        fold=0,
    )
    log_progress("Loading singleton duplicate members from BioGRID v0.5")
    _, _, singleton_examples = singleton_ds.get_train_test_split()
    singleton_lookup = {
        tuple(example["screen_ids"]): example
        for example in singleton_examples
        if len(example.get("screen_ids", [])) == 1
    }

    metrics_evaluator = RankingMetrics(
        k_values=[5, 10, 20, 50, 100],
        use_thresholded_scoring=True,
    )

    rows: List[Dict[str, Any]] = []
    log_progress(f"Computing duplicate-transfer scores for {len(duplicate_pairs)} duplicate merged screens")
    for dataset_name, duplicate_ids in duplicate_pairs:
        if len(rows) > 0 and len(rows) % 25 == 0:
            log_progress(f"Processed {len(rows)}/{len(duplicate_pairs)} duplicate merged screens")
        if len(duplicate_ids) != 2:
            report.add(
                model="plot5",
                split_layout="plot5",
                cohort="*",
                metric="adjusted_ndcg@100",
                status="partial",
                reason=f"expected duplicate pair of size 2 for {dataset_name}",
            )
            continue

        example1 = singleton_lookup.get((duplicate_ids[0],))
        example2 = singleton_lookup.get((duplicate_ids[1],))
        if example1 is None or example2 is None:
            report.add(
                model="plot5",
                split_layout="plot5",
                cohort="*",
                metric="adjusted_ndcg@100",
                status="missing",
                reason=f"missing singleton duplicate members for {dataset_name}",
            )
            continue

        idx1 = np.argsort(example1["relevance_scores"])[::-1]
        idx2 = np.argsort(example2["relevance_scores"])[::-1]
        relevance_genes1 = [example1["relevance_genes"][i] for i in idx1[:100]]
        relevance_genes2 = [example2["relevance_genes"][i] for i in idx2[:100]]

        results1 = metrics_evaluator.evaluate(
            predicted_genes=relevance_genes2,
            ground_truth_genes=example1["relevance_genes"],
            relevance_scores=example1["relevance_scores"],
        )
        results2 = metrics_evaluator.evaluate(
            predicted_genes=relevance_genes1,
            ground_truth_genes=example2["relevance_genes"],
            relevance_scores=example2["relevance_scores"],
        )

        rows.append({
            "dataset_name": dataset_name,
            "duplicate_id": list(duplicate_ids),
            "id1": duplicate_ids[0],
            "id2": duplicate_ids[1],
            "adjusted_ndcg@100_1": float(results1["adjusted_ndcg@100"]),
            "adjusted_ndcg@100_2": float(results2["adjusted_ndcg@100"]),
            "hit_scaled_adjusted_ndcg@100_1": float(results1["hit_scaled_adjusted_ndcg@100"]),
            "hit_scaled_adjusted_ndcg@100_2": float(results2["hit_scaled_adjusted_ndcg@100"]),
            "duplicate_avg_adjusted_ndcg@100": float(
                np.mean([results1["adjusted_ndcg@100"], results2["adjusted_ndcg@100"]])
            ),
            "phenotype": phenotype_map.get(dataset_name),
        })

    log_progress(f"Finished duplicate-transfer scoring for {len(rows)} duplicate merged screens")
    return pd.DataFrame(rows)


def build_plot5_model_scores_df(
    per_ex: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    model_scores = per_ex[per_ex["metric"] == "adjusted_ndcg@100"].copy()
    model_scores = model_scores.merge(
        meta_df[["example_key", "dataset_name", "biogrid_phenotype"]],
        on="example_key",
        how="left",
    )
    if "split" not in model_scores.columns:
        model_scores["split"] = model_scores["example_key"].str.split(":").str[0]
    model_scores = model_scores.dropna(subset=["dataset_name"]).copy()
    model_scores = (
        model_scores.groupby(
            ["example_key", "dataset_name", "split", "biogrid_phenotype", "model", "category", "metric"],
            as_index=False,
        )["value"]
        .mean()
    )
    return model_scores


def build_figure_cache_payload(
    *,
    no_knn: bool = False,
    no_oracle: bool = False,
    grpo_staging: Optional[str] = None,
    plot4_scaling_png: Optional[str] = None,
) -> Dict[str, Any]:
    report = MissingDataReport()
    log_progress("Resolving GRPO staging rows")
    grpo_rows = resolve_grpo_rows(grpo_staging, report)

    log_progress("Loading per-example metric caches")
    per_ex = load_per_example_from_caches(METRICS)
    log_progress(f"Loaded {len(per_ex):,} per-example metric rows from cache files")

    external_frames: List[pd.DataFrame] = []
    for name, config in EXTERNAL_MODELS.items():
        log_progress(f"Loading external metrics for {name}")
        if config["source"] == "grpo":
            df = load_external_grpo_all_metrics(config, METRICS)
        elif config["source"] == "c2s":
            df = load_external_c2s_all_metrics(config, METRICS)
        else:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["model"] = name
        df["category"] = config["category"]
        df["n_runs"] = 1
        external_frames.append(df[["example_key", "model", "category", "metric", "value", "n_runs"]])
    for name, config in STAGED_MODEL_PREDICTIONS.items():
        if config.get("format") != "list_predictions":
            continue
        log_progress(f"Recomputing metrics from staged prediction files for {name}")
        df = load_list_prediction_files_all_metrics(
            val_path=config["val_path"],
            test_path=config["test_path"],
            metrics=METRICS,
        )
        if df.empty:
            continue
        df = df.copy()
        df["model"] = name
        df["category"] = config["category"]
        df["n_runs"] = 1
        external_frames.append(df[["example_key", "model", "category", "metric", "value", "n_runs"]])
    if external_frames:
        log_progress(f"Appending {len(external_frames)} external/staged metric tables")
        per_ex = pd.concat([per_ex, *external_frames], ignore_index=True)

    if not no_knn:
        log_progress("Loading kNN evaluation results")
        knn_df = load_knn_results()
        if not knn_df.empty:
            per_ex = pd.concat(
                [per_ex, knn_df[["example_key", "model", "category", "metric", "value"]].assign(n_runs=1)],
                ignore_index=True,
            )

    log_progress("Building year/random split key sets")
    year_sets, random_sets = build_split_key_sets()
    if no_oracle:
        report.add(
            model="Oracle rerank",
            split_layout="*",
            cohort="*",
            metric="*",
            status="missing",
            reason="skipped (--no-oracle)",
        )
    else:
        log_progress("Computing/loading oracle rerank metrics")
        oracle_df = compute_oracle_rerank_metrics(ORACLE_BACKBONES, METRICS, report)
        if not oracle_df.empty:
            per_ex = pd.concat([per_ex, oracle_df], ignore_index=True)

    per_ex = per_ex.copy()
    per_ex["split"] = per_ex["example_key"].str.split(":").str[0]

    plot_rows: List[Dict[str, Any]] = []
    log_progress("Loading novel public LLM aggregate results")
    novel_ndcg = load_novel_public_llm_ndcg(NOVEL_JSON, report)

    def add_agg(split_layout: str, cohort: str, keys: Set[str], model_filter=None) -> None:
        subset = per_ex[per_ex["example_key"].isin(keys)]
        for metric in METRICS:
            grouped = subset[subset["metric"] == metric].groupby(["model", "category"], as_index=False)["value"].mean()
            for _, row in grouped.iterrows():
                if model_filter and not model_filter(row["model"], row["category"]):
                    continue
                plot_rows.append({
                    "model": row["model"],
                    "category": row["category"],
                    "metric": metric,
                    "value": row["value"],
                    "cohort": cohort,
                    "split_layout": split_layout,
                })

    add_agg("year", "val", year_sets["val"])
    add_agg("year", "test", year_sets["test"])

    for model_name, mean in novel_ndcg.items():
        plot_rows.append({
            "model": model_name,
            "category": categorize_model(model_name),
            "metric": "adjusted_ndcg@100",
            "value": float(mean),
            "cohort": "novel",
            "split_layout": "year",
        })
    for metric in METRICS:
        if metric != "adjusted_ndcg@100":
            report.add(
                model="LLM zoo",
                split_layout="year",
                cohort="novel",
                metric=metric,
                status="missing",
                reason="novel cohort JSON is nDCG-only from evaluate_model_splits STEP 4",
            )
    novel_keys = {
        example_key
        for example_key in per_ex["example_key"].unique()
        if example_key.startswith(f"{NOVEL_SPLIT_NAME}:")
    }
    if novel_keys:
        add_agg("year", "novel", novel_keys, lambda model, category: category == "Oracle rerank")

    def random_filter(model: str, category: str) -> bool:
        if is_random_split_eligible(model, category):
            return True
        if category == "Oracle rerank" and not model.startswith(FEWSHOT_PREFIX):
            return True
        return category == "Classifier"

    add_agg("random", "val", random_sets["val"], random_filter)
    add_agg("random", "test", random_sets["test"], random_filter)

    if not no_knn:
        plot_rows.extend(load_knn_aggregate_plot_rows(KNN_TEST_DIR / "year_fold0" / "knn_results.json", "year", cohorts=("novel",)))
        plot_rows.extend(load_knn_aggregate_plot_rows(KNN_TEST_DIR / "random_fold0" / "knn_results.json", "random"))

    plot_rows.extend(load_best_ensemble_rows(report))

    for row in grpo_rows:
        if row["stage"] == "Base (zero-shot)":
            continue
        model_name = f"{row['stage']} ({row['backbone']})"
        if model_name in STAGED_MODEL_PREDICTIONS:
            continue
        for cohort in ("val", "test"):
            value = row.get(cohort)
            if value is None:
                continue
            plot_rows.append({
                "model": model_name,
                "category": "LLM (trained)",
                "metric": "adjusted_ndcg@100",
                "value": float(value),
                "cohort": cohort,
                "split_layout": "year",
            })

    plot_df = pd.DataFrame(
        plot_rows,
        columns=["model", "category", "metric", "value", "cohort", "split_layout"],
    )
    plot_df = merge_classifier_into_frame(
        plot_df,
        load_classifier_series(CLASSIFIER_YEAR_CSV, report, "year"),
        load_classifier_series(CLASSIFIER_RANDOM_CSV, report, "random"),
        load_classifier_novel_recent(CLASSIFIER_RECENT_CSV, report),
    )

    for metric in METRICS:
        random_subset = plot_df[
            (plot_df["split_layout"] == "random")
            & (plot_df["metric"] == metric)
        ]
        if random_subset.empty:
            report.add(
                model="*",
                split_layout="random",
                cohort="*",
                metric=metric,
                status="partial",
                reason="no random rows",
            )

    log_progress("Loading metadata and building plot-specific cached tables")
    meta_df = load_metadata()
    year_test = per_ex[per_ex["example_key"].isin(year_sets["test"])].copy()
    random_test = per_ex[per_ex["example_key"].isin(random_sets["test"])].copy()
    plot5_duplicate_transfer_df = compute_duplicate_transfer_df(report)
    plot5_model_scores_df = build_plot5_model_scores_df(per_ex, meta_df)

    year_summary = (
        year_test[year_test["metric"] == "adjusted_ndcg@100"]
        .groupby(["model", "category"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "mean"})
    )
    random_summary = (
        random_test[random_test["metric"] == "adjusted_ndcg@100"]
        .groupby(["model", "category"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "mean"})
    )

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "inputs": {
            "no_knn": no_knn,
            "no_oracle": no_oracle,
            "grpo_staging": grpo_staging,
            "plot4_scaling_png": plot4_scaling_png,
        },
        "meta_df": meta_df,
        "plot1_df": plot_df,
        "plot2_year_df": year_test,
        "plot2_rand_df": random_test,
        "plot2_models_year": _select_models_phenotype(year_summary, year_test, "year"),
        "plot2_models_rand": _select_models_phenotype(random_summary, random_test, "random"),
        "plot2_metric": "adjusted_ndcg@100",
        "plot3_trials": load_plot3_trial_rows(report),
        "plot4_panels": load_plot4_panels(report, plot4_scaling_png),
        "plot5_duplicate_transfer_df": plot5_duplicate_transfer_df,
        "plot5_model_scores_df": plot5_model_scores_df,
        "grpo_rows": grpo_rows,
        "report": report.to_dict(),
    }
    log_progress("Finished assembling figure cache payload")
    return payload


def generate_and_save_figure_cache(
    *,
    cache_path: Path,
    no_knn: bool = False,
    no_oracle: bool = False,
    grpo_staging: Optional[str] = None,
    plot4_scaling_png: Optional[str] = None,
    captions_out: Optional[Path] = None,
    missing_report_out: Optional[Path] = None,
) -> Dict[str, Any]:
    started = time.time()
    log_progress("Starting figure cache generation")
    payload = build_figure_cache_payload(
        no_knn=no_knn,
        no_oracle=no_oracle,
        grpo_staging=grpo_staging,
        plot4_scaling_png=plot4_scaling_png,
    )
    log_progress(f"Saving cache payload to {cache_path}")
    save_figure_cache(payload, cache_path)
    report = MissingDataReport.from_dict(payload.get("report"))
    if captions_out is not None and missing_report_out is not None:
        log_progress("Writing captions and missing-data report")
        write_report_and_captions(report, captions_out, missing_report_out)
    elapsed = time.time() - started
    log_progress(f"Done in {elapsed:.1f}s")
    return payload
