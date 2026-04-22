"""
Shared utilities for LLM prediction evaluation scripts.

This module extracts common functionality used by both evaluate_model_splits.py
and analyze_multi_model_ensemble.py:
  - Data loading (predictions, ground truth)
  - Metrics cache management (per-model files)
  - Global ID mapping for cross-split evaluation
  - JSON serialization helpers
"""

import json
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from collections import defaultdict
from datasets import load_from_disk

from screensqa.benchmark.ranking_metrics import RankingMetrics
from screensqa.dataset.dataset import BioGRIDDSPY

try:
    from run_ensemble_baseline import create_dspy_examples
except ImportError:
    from scripts.run_ensemble_baseline import create_dspy_examples


# ============================================================================
# Constants
# ============================================================================

METRICS_CACHE_DIR = "metrics_cache"
METRICS_CACHE_MAPPED_DIR = "metrics_cache_mapped"

_LEGACY_CACHE_FILENAME = "metrics_cache.json"
_LEGACY_CACHE_MAPPED_FILENAME = "metrics_cache_mapped.json"

# Non-scalar fields that must NOT be stored in the metrics cache.
# raw_ground_truth_genes was previously missing from this set, causing
# ~166KB of gene lists to be stored per example per run (48GB total bloat).
SKIP_METRIC_FIELDS = frozenset({
    'num_predicted', 'num_genes', 'predicted_genes', 'raw_predicted_genes',
    'ground_truth_genes', 'raw_ground_truth_genes',
    'predicted_values', 'num_genes_normalized',
})


# ============================================================================
# Utility Helpers
# ============================================================================

def deterministic_hash(text: str) -> str:
    """
    Create a deterministic hash of a string that's consistent across Python sessions.
    Uses SHA256 truncated to first 16 characters for compactness.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def convert_for_json(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj


def remove_final_answer_tags(prediction: List[str]) -> List[str]:
    """Remove <Final Answer> tags from gene names."""
    cleaned = []
    for gene in prediction:
        gene = gene.replace('<Final Answer>', '').replace('</Final Answer>', '').strip()
        if gene:
            cleaned.append(gene)
    return cleaned


# ============================================================================
# Data Loading
# ============================================================================

def load_all_model_predictions(
    base_dir: Path,
    model_name: str,
    predictions_subdir: str = "llm_predictions",
    additional_splits: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load ALL predictions from a model (train, val, test, plus any additional
    splits) and create a mapping using index-based keys.

    Args:
        base_dir: Root output directory (e.g. output/multi_model_ensemble/)
        model_name: Name of the model subdirectory
        predictions_subdir: Subdirectory under base_dir containing model dirs.
            Default "llm_predictions" for LLM models; use "baseline_predictions"
            for baseline models.
        additional_splits: Extra split names to load (e.g. ["novel_public_dataset"]).

    Returns:
        Dict mapping "split:index" -> {'predictions': [...], 'question': ..., 'n_runs': ...}

    Keys are formatted as "train:0", "train:1", ..., "val:0", "test:0", etc.
    This matches the order in which examples appear in the dataset.
    """
    predictions_map = {}

    splits_to_load = ['train', 'val', 'test'] + (additional_splits or [])
    for split in splits_to_load:
        pred_file = base_dir / predictions_subdir / model_name / f"{split}_predictions.json"

        if not pred_file.exists():
            continue

        with open(pred_file, 'r') as f:
            data = json.load(f)

        n_runs = data.get('n_runs', 1)

        for idx, pred in enumerate(data['predictions']):
            question = pred['question']

            cleaned_predictions = []
            for p in pred['predictions']:
                cleaned_predictions.append(remove_final_answer_tags(p))

            example_key = f"{split}:{idx}"

            predictions_map[example_key] = {
                'predictions': cleaned_predictions,
                'question': question,
                'n_runs': n_runs,
                'original_split': split,
                'original_index': idx
            }

    return predictions_map


def load_all_ground_truth(
    dataset_path: str
) -> Tuple[Dict[str, Dict[str, Any]], Any]:
    """
    Load ALL ground truth from BioGRID dataset using index-based keys.

    Uses year fold 0 as the canonical ordering (matching prediction collection).

    Returns:
        Tuple of ("split:index" -> ground_truth_dict, raw_dataset)
    """
    raw_dataset = load_from_disk(dataset_path)

    dataset = BioGRIDDSPY(
        dataset_path=dataset_path,
        split_type='year',
        fold=0,
    )

    train_examples, val_examples, test_examples = dataset.get_train_test_split()

    train_dspy = create_dspy_examples(train_examples)
    val_dspy = create_dspy_examples(val_examples)
    test_dspy = create_dspy_examples(test_examples)

    ground_truth_map = {}

    for split_name, raw_exs, dspy_exs in [
        ('train', train_examples, train_dspy),
        ('val', val_examples, val_dspy),
        ('test', test_examples, test_dspy),
    ]:
        for idx, (raw_ex, dspy_ex) in enumerate(zip(raw_exs, dspy_exs)):
            example_key = f"{split_name}:{idx}"
            ground_truth_map[example_key] = {
                'genes': dspy_ex.genes,
                'relevance_scores': dspy_ex.relevance_scores,
                'question': raw_ex['question'],
                'question_hash': deterministic_hash(raw_ex['question']),
                'dataset_name': raw_ex.get('dataset_name', ''),
            }

    return ground_truth_map, raw_dataset


def load_additional_ground_truth(
    split_name: str,
    dataset_paths: List[str],
    use_existing_prompt: bool = False,
    display_library_genes: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Load ground truth from additional HuggingFace datasets.

    Returns a dict mapping ``"{split_name}:{index}"`` to ground truth dicts
    with the same format as :func:`load_all_ground_truth`.
    """
    from screensqa.utils.prompt_loaders import load_objective_prompt

    prompt_template = None
    if not use_existing_prompt:
        prompt_template = load_objective_prompt("biogrid_ranking_prompt")

    ground_truth_map: Dict[str, Dict[str, Any]] = {}
    global_idx = 0

    for ds_path in dataset_paths:
        ds = load_from_disk(ds_path)
        for item in ds:
            if use_existing_prompt:
                question = item['prompt']
            else:
                phenotype = item.get('phenotype', '')
                if phenotype and phenotype[-1] == '.':
                    item = dict(item)
                    item['phenotype'] = phenotype[:-1]
                question = prompt_template.format(**item)

            if display_library_genes and len(item['relevance_genes']) < 2000:
                question += (
                    "\n\nOnly the following genes were considered in this screen. "
                    "Only output genes from this list: "
                    + ', '.join(item['relevance_genes'])
                )

            example_key = f"{split_name}:{global_idx}"
            gt_entry = {
                'genes': item['relevance_genes'],
                'relevance_scores': item['relevance_scores'],
                'question': question,
                'question_hash': deterministic_hash(question),
                'dataset_name': item.get('dataset_name', 'unknown'),
            }
            if 'raw_scores' in item:
                gt_entry['raw_scores'] = item['raw_scores']
            ground_truth_map[example_key] = gt_entry
            global_idx += 1

    return ground_truth_map


# ============================================================================
# Global ID Mapping (for cross-split evaluation)
# ============================================================================

def create_example_global_id(question: str, genes: List[str], dataset_name: str = "") -> str:
    """
    Create a globally unique identifier for an example.

    Uses dataset_name (screen ID) + question + all ground-truth genes to
    create a unique key that's consistent across different split configurations.

    NOTE: dataset_name alone is NOT unique -- e.g. U_1241_inc and U_1508_inc each
    appear twice (inc/dec phenotype directions sharing the same name due to a
    screensqa bug).  The hash includes question text and gene list to ensure
    disambiguation even when dataset_names collide.
    """
    gene_str = "|".join(genes) if genes else ""
    combined = f"{dataset_name}||{question}||{gene_str}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:20]


def build_global_to_year_fold0_mapping(
    ground_truth_map: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """
    Build a mapping from global_id to year fold 0 key ("split:index").

    This allows looking up metrics computed on year fold 0 when evaluating
    on different split configurations.
    """
    global_to_key = {}
    collisions = 0

    for example_key, gt_data in ground_truth_map.items():
        global_id = create_example_global_id(
            gt_data['question'], gt_data['genes'], gt_data.get('dataset_name', ''))
        if global_id in global_to_key:
            collisions += 1
            print(f"  Warning: global_id collision between {global_to_key[global_id]} and {example_key}")
        global_to_key[global_id] = example_key

    if collisions:
        print(f"  Warning: {collisions} global_id collision(s) — some examples will be lost in cross-split lookups")

    return global_to_key


def get_split_assignments(
    dataset_path: str,
    global_to_year_fold0_key: Dict[str, str],
    split_type: str,
    fold: int
) -> Dict[str, List[str]]:
    """
    Get which examples belong to train/val/test for a given split configuration.

    Maps examples from the requested split configuration back to their
    year fold 0 keys (used for metrics cache lookup).

    Returns:
        Dict mapping split_name -> list of year_fold_0 keys
    """
    dataset = BioGRIDDSPY(
        dataset_path=dataset_path,
        split_type=split_type,
        fold=fold,
    )

    train_examples, val_examples, test_examples = dataset.get_train_test_split()

    train_dspy = create_dspy_examples(train_examples)
    val_dspy = create_dspy_examples(val_examples)
    test_dspy = create_dspy_examples(test_examples)

    split_assignments = {
        'train': [],
        'val': [],
        'test': []
    }

    for split_name, raw_exs, dspy_exs in [
        ('train', train_examples, train_dspy),
        ('val', val_examples, val_dspy),
        ('test', test_examples, test_dspy),
    ]:
        for raw_ex, dspy_ex in zip(raw_exs, dspy_exs):
            global_id = create_example_global_id(
                raw_ex['question'], dspy_ex.genes, raw_ex.get('dataset_name', ''))
            if global_id in global_to_year_fold0_key:
                split_assignments[split_name].append(global_to_year_fold0_key[global_id])

    return split_assignments


# ============================================================================
# Metrics Cache Management
# ============================================================================

def _get_cache_subdir(cache_dir: Path, use_mapped: bool = False) -> Path:
    """Get the per-model metrics cache subdirectory."""
    dirname = METRICS_CACHE_MAPPED_DIR if use_mapped else METRICS_CACHE_DIR
    return cache_dir / dirname


def _model_cache_path(cache_subdir: Path, model_name: str) -> Path:
    """Get the cache file path for a specific model (sanitizes slashes)."""
    safe_name = model_name.replace("/", "__")
    return cache_subdir / f"{safe_name}.json"


def _migrate_legacy_cache(cache_dir: Path, use_mapped: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    One-time migration: split a legacy monolithic cache file into per-model files.

    Returns the loaded cache dict, or empty dict if no migration was needed.
    """
    legacy_filename = _LEGACY_CACHE_MAPPED_FILENAME if use_mapped else _LEGACY_CACHE_FILENAME
    legacy_file = cache_dir / legacy_filename
    cache_type = "mapped" if use_mapped else "unmapped"

    if not legacy_file.exists():
        return {}

    cache_subdir = _get_cache_subdir(cache_dir, use_mapped)

    if cache_subdir.exists() and any(cache_subdir.glob("*.json")):
        return {}

    size_gb = legacy_file.stat().st_size / 1e9
    print(f"Migrating legacy {cache_type} cache to per-model directory format...")
    print(f"  Legacy file: {legacy_file} ({size_gb:.1f} GB)")

    try:
        with open(legacy_file, 'r') as f:
            cache_data = json.load(f)

        version = cache_data.get('version', '1.0')
        if version in ['1.0', '2.0']:
            print(f"  Legacy cache is v{version} (incompatible) — skipping migration")
            return {}

        models_data = cache_data.get('models', {})
        if not models_data:
            return {}

        cache_subdir.mkdir(parents=True, exist_ok=True)

        for model_name, model_cache in models_data.items():
            model_file = _model_cache_path(cache_subdir, model_name)
            with open(model_file, 'w') as f:
                json.dump(model_cache, f)
            print(f"  Migrated {model_name} ({len(model_cache)} examples)")

        backup_file = legacy_file.with_suffix('.json.bak')
        legacy_file.rename(backup_file)
        print(f"  Renamed legacy file to {backup_file.name}")
        print(f"  Migration complete: {len(models_data)} models")

        return models_data

    except Exception as e:
        print(f"  Warning: Migration failed: {e}")
        print(f"  Will recompute metrics as needed")
        return {}


def _compact_single_cache(model_cache: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Strip non-scalar fields from a loaded model cache dict (in-memory).

    Returns:
        Tuple of (compacted_cache, was_modified)
    """
    modified = False
    for example_key, entry in model_cache.items():
        for run_metrics in entry.get('metrics_per_run', []):
            for field in SKIP_METRIC_FIELDS:
                if field in run_metrics:
                    del run_metrics[field]
                    modified = True
    return model_cache, modified


def load_metrics_cache(cache_dir: Path, use_mapped: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Load existing metrics cache from per-model files in cache directory.

    Falls back to migrating a legacy monolithic cache file if no per-model
    directory exists yet.  Auto-compacts caches that still contain bloated
    non-scalar fields (e.g. raw_ground_truth_genes).

    Args:
        cache_dir: Directory containing the cache subdirectory
        use_mapped: If True, load the gene-mapped metrics cache

    Returns:
        Dict mapping model_name -> {example_key (str) -> metrics_data}
    """
    cache_subdir = _get_cache_subdir(cache_dir, use_mapped)
    cache_type = "mapped" if use_mapped else "unmapped"

    if cache_subdir.exists():
        model_files = sorted(cache_subdir.glob("*.json"))
        if model_files:
            result = {}
            for model_file in model_files:
                model_name = model_file.stem.replace("__", "/")
                try:
                    with open(model_file, 'r') as f:
                        model_cache = json.load(f)
                    model_cache, was_compacted = _compact_single_cache(model_cache)
                    if was_compacted:
                        with open(model_file, 'w') as f:
                            json.dump(model_cache, f)
                        new_size_mb = model_file.stat().st_size / 1e6
                        print(f"  Auto-compacted {model_name} cache -> {new_size_mb:.1f} MB")
                    result[model_name] = model_cache
                except (json.JSONDecodeError, OSError) as e:
                    print(f"  Warning: Could not load cache for {model_name}: {e}")
            print(f"Loaded {cache_type} metrics cache: {len(result)} models from per-model files")
            return result

    migrated = _migrate_legacy_cache(cache_dir, use_mapped)
    if migrated:
        return migrated

    return {}


def save_model_metrics_cache(
    cache_dir: Path,
    model_name: str,
    model_cache: Dict[str, Dict[str, Any]],
    use_mapped: bool = False
):
    """
    Save metrics cache for a single model to its own file.

    Only the specified model's file is written — other models are untouched.
    """
    cache_subdir = _get_cache_subdir(cache_dir, use_mapped)
    cache_subdir.mkdir(parents=True, exist_ok=True)
    cache_type = "mapped" if use_mapped else "unmapped"

    model_file = _model_cache_path(cache_subdir, model_name)
    with open(model_file, 'w') as f:
        json.dump(convert_for_json(model_cache), f)

    print(f"Saved {cache_type} cache for {model_name}: {len(model_cache)} examples -> {model_file.name}")


def compact_metrics_cache(cache_dir_path: str, use_mapped: bool = False):
    """
    One-time utility to compact existing metrics cache files on disk.

    Strips non-scalar fields (especially raw_ground_truth_genes) from all
    per-model cache files, dramatically reducing disk usage.

    Can be run standalone:
        python -c "from scripts.shared_utils import compact_metrics_cache; compact_metrics_cache('./output/multi_model_ensemble/model_split_evaluation')"
    """
    cache_dir = Path(cache_dir_path)
    for mapped in ([use_mapped] if isinstance(use_mapped, bool) else [False, True]):
        cache_subdir = _get_cache_subdir(cache_dir, mapped)
        cache_type = "mapped" if mapped else "unmapped"

        if not cache_subdir.exists():
            print(f"No {cache_type} cache directory at {cache_subdir}")
            continue

        model_files = sorted(cache_subdir.glob("*.json"))
        if not model_files:
            print(f"No cache files in {cache_subdir}")
            continue

        print(f"\nCompacting {len(model_files)} {cache_type} cache files in {cache_subdir}...")
        total_saved = 0

        for model_file in model_files:
            old_size = model_file.stat().st_size
            with open(model_file, 'r') as f:
                model_cache = json.load(f)

            model_cache, was_modified = _compact_single_cache(model_cache)

            if was_modified:
                with open(model_file, 'w') as f:
                    json.dump(model_cache, f)
                new_size = model_file.stat().st_size
                saved = old_size - new_size
                total_saved += saved
                print(f"  {model_file.name}: {old_size/1e6:.1f} MB -> {new_size/1e6:.1f} MB (saved {saved/1e6:.1f} MB)")
            else:
                print(f"  {model_file.name}: already compact")

        print(f"\nTotal saved: {total_saved/1e9:.2f} GB")


# ============================================================================
# Metrics Computation
# ============================================================================

# Module-level evaluator shared with forked worker processes (avoids pickling
# large objects on Linux where fork is the default start method).
_mp_evaluator: Optional['RankingMetrics'] = None


def _eval_example_shared(args: Tuple) -> Tuple[str, Dict[str, Any]]:
    """Worker function: evaluate all runs for a single example."""
    example_key, predictions_list, gt_genes, gt_relevance = args
    evaluator = _mp_evaluator
    metrics_per_run = []
    for predicted_genes in predictions_list:
        results = evaluator.evaluate(
            predicted_genes=predicted_genes,
            ground_truth_genes=gt_genes,
            relevance_scores=gt_relevance,
        )
        metrics_per_run.append({
            k: v for k, v in results.items() if k not in SKIP_METRIC_FIELDS
        })
    return example_key, {
        'metrics_per_run': metrics_per_run,
        'n_runs': len(metrics_per_run),
    }


def compute_all_metrics(
    predictions_map: Dict[str, Dict[str, Any]],
    ground_truth_map: Dict[str, Dict[str, Any]],
    metrics_evaluator: 'RankingMetrics',
    n_workers: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute metrics for ALL examples ONCE using index-based keys.

    When *n_workers* > 1 the per-example evaluations are distributed across
    processes via ``ProcessPoolExecutor``.  On Linux (fork start-method) the
    ``RankingMetrics`` evaluator is inherited by workers without pickling.

    Returns:
        Dict mapping "split:index" -> {
            'metrics_per_run': [run0_metrics, run1_metrics, ...],
            'n_runs': int
        }
    """
    global _mp_evaluator

    common_keys = set(predictions_map.keys()) & set(ground_truth_map.keys())

    def _normalize_ws(s: str) -> str:
        """Collapse all whitespace runs to single spaces for comparison."""
        return " ".join(s.split())

    mismatches_ws = 0
    mismatched_keys = set()
    for example_key in sorted(common_keys):
        pred_question = predictions_map[example_key].get('question', '')
        gt_question = ground_truth_map[example_key].get('question', '')
        if pred_question and gt_question and pred_question != gt_question:
            if _normalize_ws(pred_question) == _normalize_ws(gt_question):
                mismatches_ws += 1
            else:
                mismatched_keys.add(example_key)
                if len(mismatched_keys) <= 3:
                    print(f"  MISMATCH at {example_key}:")
                    print(f"    prediction question: {pred_question[:80]}...")
                    print(f"    ground truth question: {gt_question[:80]}...")

    if mismatches_ws:
        print(f"  Note: {mismatches_ws} examples have whitespace-only question differences (OK)")

    mismatch_ratio = len(mismatched_keys) / len(common_keys) if common_keys else 0
    if mismatched_keys and mismatch_ratio > 0.01:
        print(f"\n  ERROR: {len(mismatched_keys)}/{len(common_keys)} examples ({mismatch_ratio:.1%}) have mismatched questions!")
        print(f"  This usually means the dataset_path used for evaluation differs from the one")
        print(f"  used during prediction collection (different ordering or different version).")
        raise ValueError(
            f"Question mismatch: {len(mismatched_keys)} examples ({mismatch_ratio:.1%}) have different questions. "
            f"The dataset_path likely differs from the one used during prediction collection."
        )
    elif mismatched_keys:
        print(f"  Warning: {len(mismatched_keys)}/{len(common_keys)} examples have mismatched question text "
              f"(likely dataset_name update for duplicate screens \u2014 index alignment preserved, proceeding)")

    print(f"  Computing metrics for {len(common_keys)} examples...")

    tasks = []
    for example_key in common_keys:
        pred_data = predictions_map[example_key]
        gt_data = ground_truth_map[example_key]
        preds = [
            pred_data['predictions'][i]
            for i in range(pred_data['n_runs'])
            if i < len(pred_data['predictions'])
        ]
        tasks.append((example_key, preds, gt_data['genes'], gt_data['relevance_scores']))

    effective_workers = min(max(n_workers, 1), len(tasks)) if tasks else 1

    if effective_workers > 1:
        _mp_evaluator = metrics_evaluator
        print(f"  Using {effective_workers} parallel workers...")
        metrics_cache: Dict[str, Dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            for key, value in executor.map(_eval_example_shared, tasks, chunksize=32):
                metrics_cache[key] = value
        _mp_evaluator = None
    else:
        _mp_evaluator = metrics_evaluator
        metrics_cache = {}
        for t in tasks:
            key, value = _eval_example_shared(t)
            metrics_cache[key] = value
        _mp_evaluator = None

    return metrics_cache


# ============================================================================
# Metrics Aggregation
# ============================================================================

def aggregate_metrics_for_split(
    metrics_cache: Dict[str, Dict[str, Any]],
    split_keys: List[str],
    metric_name: str
) -> Optional[Dict[str, Any]]:
    """
    Aggregate pre-computed metrics for a specific split.

    Args:
        metrics_cache: Dict mapping example_key -> metrics data
        split_keys: List of example keys to aggregate (year fold 0 keys)
        metric_name: Name of metric to aggregate (used for filtering)

    Returns:
        Dict with 'per_run_means', 'all_values', 'n_runs', 'n_examples'
    """
    if not split_keys:
        return None

    valid_keys = [k for k in split_keys if k in metrics_cache]

    if not valid_keys:
        return None

    n_runs = metrics_cache[valid_keys[0]]['n_runs']

    all_values = defaultdict(list)
    per_run_means = defaultdict(list)

    for run_idx in range(n_runs):
        run_values = defaultdict(list)

        for example_key in valid_keys:
            cached = metrics_cache[example_key]
            if run_idx < len(cached['metrics_per_run']):
                run_metrics = cached['metrics_per_run'][run_idx]
                for key, value in run_metrics.items():
                    if key in SKIP_METRIC_FIELDS:
                        continue
                    if isinstance(value, (list, tuple, np.ndarray)):
                        continue
                    all_values[key].append(value)
                    run_values[key].append(value)

        for key, values in run_values.items():
            if values:
                per_run_means[key].append(np.nanmean(values))

    if not per_run_means:
        return None

    return {
        'all_values': dict(all_values),
        'per_run_means': dict(per_run_means),
        'n_runs': n_runs,
        'n_examples': len(valid_keys)
    }
