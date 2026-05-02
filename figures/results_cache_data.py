from __future__ import annotations

import json
import sys
import time
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from assaybench import AssayBenchDataset, RankingMetrics

from journal_figures_common import (
    DEFAULT_RESULTS_CACHE_PATH,
    FEWSHOT_PREFIX,
    METRICS,
    PREDICTIONS_DIR,
    RESULTS_CACHE_SCHEMA_VERSION,
    MissingDataReport,
    categorize_model,
    load_results_cache,
    save_results_cache,
)

NOVEL_SPLIT_NAME = "novel_public_dataset"
_SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def log_progress(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[generate_results_cache {timestamp}] {message}", flush=True)


def _elapsed(started: float) -> str:
    return f"{time.perf_counter() - started:.1f}s"


_WORKER_YEAR_GT_BY_DATASET: Optional[Dict[str, Dict[str, Any]]] = None
_WORKER_NOVEL_GT_BY_DATASET: Optional[Dict[str, Dict[str, Any]]] = None
_WORKER_EVALUATOR: Any = None


def _load_ground_truth_by_dataset() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    ds = AssayBenchDataset(
        dataset_name="biogrid",
        split_type="year",
        fold=0,
        novel_dataset_name="2026Q1",
    )
    train, val, test, novel = ds.get_train_test_split()

    year_by_dataset: Dict[str, Dict[str, Any]] = {}
    for split_name, examples in [("train", train), ("val", val), ("test", test)]:
        for idx, ex in enumerate(examples):
            year_by_dataset[str(ex["dataset_name"])] = {
                "dataset_name": str(ex["dataset_name"]),
                "example_key": f"{split_name}:{idx}",
                "split": split_name,
                "genes": ex["relevance_genes"],
                "relevance_scores": ex["relevance_scores"],
            }

    novel_by_dataset: Dict[str, Dict[str, Any]] = {}
    for idx, ex in enumerate(novel):
        novel_by_dataset[str(ex["dataset_name"])] = {
            "dataset_name": str(ex["dataset_name"]),
            "example_key": f"novel:{idx}",
            "split": NOVEL_SPLIT_NAME,
            "genes": ex["relevance_genes"],
            "relevance_scores": ex["relevance_scores"],
        }
    return year_by_dataset, novel_by_dataset


def _load_metadata() -> pd.DataFrame:
    ds = AssayBenchDataset(dataset_name="biogrid", split_type="year", fold=0)
    examples = ds.get_list_examples()
    split_counters: Dict[str, int] = defaultdict(int)
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        split = _SPLIT_MAP.get(ex["split"], ex["split"])
        idx = split_counters[split]
        split_counters[split] += 1
        rows.append({
            "example_key": f"{split}:{idx}",
            "dataset_name": ex["dataset_name"],
            "split": split,
            "screen_type": ex.get("screen_type", "Not specified"),
            "library_methodology": ex.get("library_methodology", "Not specified"),
            "num_genes": ex.get("num_genes", 0),
            "biogrid_phenotype": ex.get("cleaned_phenotype", "Not specified"),
            "biogrid_screen_rationale": ex.get("screen_rationale", "Not specified"),
        })
    return pd.DataFrame(rows)


def _build_split_key_sets() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    year_sets: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    random_sets: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    for split_type, sets in [("year", year_sets), ("random", random_sets)]:
        ds = AssayBenchDataset(dataset_name="biogrid", split_type=split_type, fold=0)
        examples = ds.get_list_examples()
        counters: Dict[str, int] = defaultdict(int)
        for ex in examples:
            split = _SPLIT_MAP.get(ex["split"], ex["split"])
            idx = counters[split]
            counters[split] += 1
            sets[split].add(f"{split}:{idx}")
    return year_sets, random_sets


def _load_results_manifest(
    results_dir: Path,
    selected_models: Optional[Set[str]] = None,
) -> List[Path]:
    paths = sorted(
        p for p in results_dir.rglob("*.json")
        if p.name != "harmonized_predictions_manifest.json"
    )
    if selected_models is None:
        return paths

    selected_paths: List[Path] = []
    for path in paths:
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except Exception:
            continue
        model_name = str(payload.get("model_name", path.stem))
        if model_name in selected_models or _normalized_model_name(model_name) in selected_models:
            selected_paths.append(path)
    return selected_paths


def _normalized_model_name(model_name: str) -> str:
    if model_name.startswith(FEWSHOT_PREFIX):
        return model_name[len(FEWSHOT_PREFIX):]
    return model_name


def _extract_prediction_runs(record: Dict[str, Any]) -> List[List[str]]:
    runs = record.get("prediction_runs")
    if isinstance(runs, list) and runs:
        return [
            [str(gene).strip() for gene in run if str(gene).strip()]
            for run in runs
            if isinstance(run, list)
        ]
    predicted = record.get("predicted_genes", [])
    if isinstance(predicted, list):
        return [[str(gene).strip() for gene in predicted if str(gene).strip()]]
    return []


def _evaluate_prediction_runs(
    predicted_runs: List[List[str]],
    ground_truth: Dict[str, Any],
    evaluator: RankingMetrics,
) -> Dict[str, float]:
    if not predicted_runs:
        return {}
    per_metric: Dict[str, List[float]] = {metric: [] for metric in METRICS}
    for run in predicted_runs:
        result = evaluator.evaluate(
            predicted_genes=list(run),
            ground_truth_genes=ground_truth["genes"],
            relevance_scores=ground_truth["relevance_scores"],
        )
        for metric in METRICS:
            value = result.get(metric)
            if value is not None and not np.isnan(value):
                per_metric[metric].append(float(value))
    return {
        metric: float(np.mean(values))
        for metric, values in per_metric.items()
        if values
    }


def _normalize_cohort(split: str) -> str:
    if split == NOVEL_SPLIT_NAME:
        return "novel"
    return split


def _get_worker_ground_truth() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    global _WORKER_YEAR_GT_BY_DATASET
    global _WORKER_NOVEL_GT_BY_DATASET
    if _WORKER_YEAR_GT_BY_DATASET is None or _WORKER_NOVEL_GT_BY_DATASET is None:
        _WORKER_YEAR_GT_BY_DATASET, _WORKER_NOVEL_GT_BY_DATASET = _load_ground_truth_by_dataset()
    return _WORKER_YEAR_GT_BY_DATASET, _WORKER_NOVEL_GT_BY_DATASET


def _get_worker_evaluator() -> RankingMetrics:
    global _WORKER_EVALUATOR
    if _WORKER_EVALUATOR is None:
        _WORKER_EVALUATOR = RankingMetrics(k_values=[5, 10, 20, 50, 100])
    return _WORKER_EVALUATOR


def _score_single_results_file(path: Path) -> Dict[str, Any]:
    year_gt_by_dataset, novel_gt_by_dataset = _get_worker_ground_truth()
    evaluator = _get_worker_evaluator()

    file_started = time.perf_counter()
    with open(path) as handle:
        payload = json.load(handle)

    model_name = _normalized_model_name(str(payload.get("model_name", path.stem)))
    category = categorize_model(model_name)
    source_group = payload.get("source_group")
    records_by_dataset = payload.get("records_by_dataset", {})
    total_datasets = len(records_by_dataset)
    total_records = sum(len(records) for records in records_by_dataset.values())
    log_progress(
        f"Scoring {model_name} from {path.name}: "
        f"{total_datasets:,} datasets, {total_records:,} records"
    )

    processed_records = 0
    scoring_started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    report_entries: List[Dict[str, Any]] = []

    for dataset_name, records in records_by_dataset.items():
        for record in records:
            processed_records += 1
            split_layout = str(record.get("split_layout", "")).strip().lower()
            split = str(record.get("split", "")).strip()
            cohort = _normalize_cohort(split)

            if split_layout == "novel" or split == NOVEL_SPLIT_NAME:
                ground_truth = novel_gt_by_dataset.get(str(dataset_name))
            else:
                ground_truth = year_gt_by_dataset.get(str(dataset_name))

            if ground_truth is None:
                report_entries.append({
                    "model": model_name,
                    "split_layout": split_layout or "*",
                    "cohort": cohort or "*",
                    "metric": "*",
                    "status": "missing",
                    "reason": f"ground truth not found for dataset_name={dataset_name}",
                })
                continue

            predicted_runs = _extract_prediction_runs(record)
            metric_values = _evaluate_prediction_runs(predicted_runs, ground_truth, evaluator)
            if not metric_values:
                report_entries.append({
                    "model": model_name,
                    "split_layout": split_layout or "*",
                    "cohort": cohort or "*",
                    "metric": "*",
                    "status": "missing",
                    "reason": f"no prediction runs for dataset_name={dataset_name}",
                })
                continue

            example_key = record.get("example_key")
            if not example_key and split_layout == "year":
                example_key = ground_truth.get("example_key")

            for metric, value in metric_values.items():
                rows.append({
                    "model": model_name,
                    "category": category,
                    "source_group": source_group,
                    "dataset_name": str(dataset_name),
                    "example_key": example_key,
                    "split": split,
                    "cohort": cohort,
                    "split_layout": split_layout,
                    "metric": metric,
                    "value": float(value),
                    "n_runs": len(predicted_runs),
                })
            if processed_records % 250 == 0:
                log_progress(
                    f"{model_name}: processed {processed_records:,}/{total_records:,} records "
                    f"in {_elapsed(scoring_started)}"
                )

    return {
        "path": str(path),
        "model_name": model_name,
        "rows": rows,
        "report_entries": report_entries,
        "processed_records": processed_records,
        "total_records": total_records,
        "elapsed_seconds": time.perf_counter() - file_started,
    }


def build_results_metric_table(
    results_dir: Path,
    report: MissingDataReport,
    workers: int = 1,
    selected_models: Optional[Set[str]] = None,
) -> pd.DataFrame:
    started = time.perf_counter()
    gt_started = time.perf_counter()
    year_gt_by_dataset, novel_gt_by_dataset = _load_ground_truth_by_dataset()
    log_progress(
        "Loaded ground-truth maps: "
        f"{len(year_gt_by_dataset):,} main + {len(novel_gt_by_dataset):,} novel "
        f"in {_elapsed(gt_started)}"
    )

    manifest_started = time.perf_counter()
    result_files = _load_results_manifest(results_dir, selected_models=selected_models)
    log_progress(
        f"Resolved {len(result_files):,} prediction files from {results_dir} "
        f"in {_elapsed(manifest_started)}"
    )
    if not result_files:
        report.add(
            model="*",
            split_layout="results",
            cohort="*",
            metric="*",
            status="missing",
            reason=f"no prediction files found in {results_dir}",
        )
        return pd.DataFrame()

    global _WORKER_YEAR_GT_BY_DATASET
    global _WORKER_NOVEL_GT_BY_DATASET
    global _WORKER_EVALUATOR
    _WORKER_YEAR_GT_BY_DATASET = year_gt_by_dataset
    _WORKER_NOVEL_GT_BY_DATASET = novel_gt_by_dataset
    _WORKER_EVALUATOR = None

    evaluator_started = time.perf_counter()
    _get_worker_evaluator()
    log_progress(f"Initialized RankingMetrics evaluator in {_elapsed(evaluator_started)}")
    rows: List[Dict[str, Any]] = []
    if workers <= 1:
        for file_index, path in enumerate(result_files, start=1):
            log_progress(f"[{file_index}/{len(result_files)}] Starting {path.name}")
            result = _score_single_results_file(path)
            rows.extend(result["rows"])
            report.entries.extend(result["report_entries"])
            log_progress(
                f"Finished {result['model_name']} ({Path(result['path']).name}) in "
                f"{result['elapsed_seconds']:.1f}s; "
                f"processed {result['processed_records']:,}/{result['total_records']:,} records; "
                f"cumulative metric rows: {len(rows):,}"
            )
    else:
        import os
        worker_count = min(max(1, workers), len(result_files))
        log_progress(f"Scoring {len(result_files):,} files in parallel with {worker_count} workers")
        mp_context = mp.get_context("fork") if os.name != "nt" else None
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
            future_map = {
                executor.submit(_score_single_results_file, path): path
                for path in result_files
            }
            completed = 0
            for future in as_completed(future_map):
                result = future.result()
                completed += 1
                rows.extend(result["rows"])
                report.entries.extend(result["report_entries"])
                log_progress(
                    f"[{completed}/{len(result_files)}] Finished {result['model_name']} "
                    f"({Path(result['path']).name}) in {result['elapsed_seconds']:.1f}s; "
                    f"processed {result['processed_records']:,}/{result['total_records']:,} records; "
                    f"cumulative metric rows: {len(rows):,}"
                )

    log_progress(f"Finished scoring all predictions in {_elapsed(started)}")
    return pd.DataFrame(rows)


def _dataset_name_sets_by_split(meta_df: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    year_sets, random_sets = _build_split_key_sets()
    dataset_lookup = (
        meta_df[["example_key", "dataset_name"]]
        .drop_duplicates()
        .set_index("example_key")["dataset_name"]
        .to_dict()
    )

    def translate(keys: Iterable[str]) -> Set[str]:
        return {
            str(dataset_lookup[key])
            for key in keys
            if key in dataset_lookup and pd.notna(dataset_lookup[key])
        }

    year_dataset_sets = {split: translate(keys) for split, keys in year_sets.items()}
    random_dataset_sets = {split: translate(keys) for split, keys in random_sets.items()}
    return year_dataset_sets, random_dataset_sets


def build_plot1_df_from_results(
    metrics_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    report: MissingDataReport,
) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["model", "category", "metric", "value", "cohort", "split_layout"])

    year_dataset_sets, random_dataset_sets = _dataset_name_sets_by_split(meta_df)
    plot_rows: List[Dict[str, Any]] = []

    all_models = sorted(metrics_df["model"].dropna().unique().tolist())

    def append_mean_rows(subset: pd.DataFrame, split_layout: str, cohort: str) -> None:
        if subset.empty:
            return
        grouped = subset.groupby(["model", "category", "metric"], as_index=False)["value"].mean()
        for _, row in grouped.iterrows():
            plot_rows.append({
                "model": row["model"],
                "category": row["category"],
                "metric": row["metric"],
                "value": float(row["value"]),
                "cohort": cohort,
                "split_layout": split_layout,
            })

    year_subset = metrics_df[metrics_df["split_layout"] == "year"].copy()
    for cohort in ("val", "test"):
        append_mean_rows(year_subset[year_subset["cohort"] == cohort], "year", cohort)

    novel_subset = metrics_df[
        (metrics_df["cohort"] == "novel")
        & (metrics_df["split_layout"].isin(["year", "novel"]))
    ].copy()
    append_mean_rows(novel_subset, "year", "novel")

    explicit_random_models = set(
        metrics_df.loc[metrics_df["split_layout"] == "random", "model"].dropna().tolist()
    )
    for cohort in ("val", "test"):
        explicit_random = metrics_df[
            (metrics_df["split_layout"] == "random")
            & (metrics_df["cohort"] == cohort)
        ].copy()
        append_mean_rows(explicit_random, "random", cohort)

        remapped_random = metrics_df[
            (metrics_df["split_layout"] == "year")
            & (metrics_df["dataset_name"].isin(random_dataset_sets.get(cohort, set())))
            & (~metrics_df["model"].isin(explicit_random_models))
        ].copy()
        append_mean_rows(remapped_random, "random", cohort)

    plot_df = pd.DataFrame(
        plot_rows,
        columns=["model", "category", "metric", "value", "cohort", "split_layout"],
    )

    for metric in METRICS:
        for split_layout, cohorts in [("year", ("val", "test", "novel")), ("random", ("val", "test"))]:
            for cohort in cohorts:
                subset = plot_df[
                    (plot_df["metric"] == metric)
                    & (plot_df["split_layout"] == split_layout)
                    & (plot_df["cohort"] == cohort)
                ]
                if subset.empty:
                    report.add(
                        model="*",
                        split_layout=split_layout,
                        cohort=cohort,
                        metric=metric,
                        status="partial",
                        reason="no rows in results cache",
                    )

    missing_models = set(all_models) - set(plot_df["model"].dropna().unique().tolist())
    for model_name in sorted(missing_models):
        report.add(
            model=model_name,
            split_layout="plot1",
            cohort="*",
            metric="*",
            status="missing",
            reason="model had scored rows but no plot1 aggregate rows",
        )

    return plot_df


def build_model_scores_df_from_results(
    metrics_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "example_key", "dataset_name", "split", "biogrid_phenotype",
                "model", "category", "metric", "value",
            ]
        )

    year_lookup = (
        meta_df[["dataset_name", "example_key", "split", "biogrid_phenotype"]]
        .drop_duplicates(subset=["dataset_name"])
        .rename(columns={"split": "year_split"})
    )
    model_scores = metrics_df[
        (metrics_df["split_layout"] == "year")
        & (metrics_df["metric"] == "adjusted_ndcg@100")
        & (metrics_df["dataset_name"].notna())
    ].copy()
    model_scores = model_scores.merge(
        year_lookup,
        on="dataset_name",
        how="left",
        suffixes=("_record", "_year"),
    )
    model_scores["example_key"] = model_scores["example_key_record"].fillna(model_scores["example_key_year"])
    model_scores["split"] = model_scores["cohort"].replace({"novel": NOVEL_SPLIT_NAME})
    model_scores["split"] = model_scores["split"].fillna(model_scores["year_split"])
    model_scores["biogrid_phenotype"] = model_scores["biogrid_phenotype"].fillna("Not specified")
    model_scores = model_scores[
        ["example_key", "dataset_name", "split", "biogrid_phenotype", "model", "category", "metric", "value"]
    ].copy()
    return model_scores


def build_results_cache_payload(
    *,
    results_dir: Path = PREDICTIONS_DIR,
    workers: int = 1,
    selected_models: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    report = MissingDataReport()
    log_progress(f"Scoring prediction files from {results_dir}")
    metrics_df = build_results_metric_table(
        results_dir,
        report,
        workers=workers,
        selected_models=selected_models,
    )
    log_progress(f"Scored {len(metrics_df):,} per-record metric rows")

    log_progress("Loading metadata and split assignments")
    meta_df = _load_metadata()

    log_progress("Building Plot 1 aggregate table from scored results")
    plot1_df = build_plot1_df_from_results(metrics_df, meta_df, report)

    log_progress("Building model score table from scored results")
    model_scores_df = build_model_scores_df_from_results(metrics_df, meta_df)

    payload = {
        "schema_version": RESULTS_CACHE_SCHEMA_VERSION,
        "inputs": {
            "results_dir": str(results_dir),
            "workers": workers,
            "selected_models": sorted(selected_models) if selected_models is not None else None,
        },
        "plot1_df": plot1_df,
        "plot5_model_scores_df": model_scores_df,
        "report": report.to_dict(),
    }
    log_progress("Finished assembling results cache payload")
    return payload


def _merge_selected_models_into_existing_cache(
    existing_payload: Dict[str, Any],
    updated_payload: Dict[str, Any],
    selected_models: Set[str],
) -> Dict[str, Any]:
    merged = dict(existing_payload)

    existing_plot1 = existing_payload.get("plot1_df", pd.DataFrame())
    updated_plot1 = updated_payload.get("plot1_df", pd.DataFrame())
    if isinstance(existing_plot1, pd.DataFrame):
        existing_plot1 = existing_plot1.loc[~existing_plot1["model"].isin(selected_models)].copy()
    else:
        existing_plot1 = pd.DataFrame()
    merged["plot1_df"] = pd.concat([existing_plot1, updated_plot1], ignore_index=True)

    existing_scores = existing_payload.get("plot5_model_scores_df", pd.DataFrame())
    updated_scores = updated_payload.get("plot5_model_scores_df", pd.DataFrame())
    if isinstance(existing_scores, pd.DataFrame):
        existing_scores = existing_scores.loc[~existing_scores["model"].isin(selected_models)].copy()
    else:
        existing_scores = pd.DataFrame()
    merged["plot5_model_scores_df"] = pd.concat([existing_scores, updated_scores], ignore_index=True)

    merged["schema_version"] = RESULTS_CACHE_SCHEMA_VERSION
    merged["inputs"] = dict(existing_payload.get("inputs", {}))
    merged["inputs"].update({
        "results_dir": updated_payload.get("inputs", {}).get("results_dir"),
        "workers": updated_payload.get("inputs", {}).get("workers"),
        "last_partial_update_models": sorted(selected_models),
    })
    merged["report"] = existing_payload.get("report", {})
    return merged


def generate_and_save_results_cache(
    *,
    cache_path: Path = DEFAULT_RESULTS_CACHE_PATH,
    results_dir: Path = PREDICTIONS_DIR,
    missing_report_out: Optional[Path] = None,
    workers: int = 1,
    selected_models: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    started = time.time()
    had_existing_cache = cache_path.exists()
    log_progress("Starting results cache generation")
    payload = build_results_cache_payload(
        results_dir=results_dir,
        workers=workers,
        selected_models=selected_models,
    )
    if selected_models:
        if had_existing_cache:
            log_progress(
                f"Merging updated cache entries for {', '.join(sorted(selected_models))} "
                f"into existing cache at {cache_path}"
            )
            existing_payload = load_results_cache(cache_path)
            payload = _merge_selected_models_into_existing_cache(
                existing_payload,
                payload,
                selected_models,
            )
        else:
            log_progress(
                "No existing results cache found; building a partial cache containing only "
                f"{', '.join(sorted(selected_models))}"
            )
    save_results_cache(payload, cache_path)
    log_progress(f"Saved results cache to {cache_path}")

    if missing_report_out is not None:
        if selected_models and had_existing_cache:
            log_progress(
                "Keeping the existing missing-data report unchanged for partial model updates"
            )
        else:
            report = MissingDataReport.from_dict(payload.get("report"))
            missing_report_out.parent.mkdir(parents=True, exist_ok=True)
            missing_report_out.write_text(report.to_markdown())
            log_progress(f"Updated missing-data report at {missing_report_out}")

    elapsed = time.time() - started
    log_progress(f"Results cache generation finished in {elapsed:.1f}s")
    return payload
