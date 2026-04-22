from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from journal_figures_common import (
    BEST_ENSEMBLE_ALL_METRICS,
    DATASET_PATH,
    LLM_PRED_DIR,
    NOVEL_DATASET_PATHS,
    NOVEL_SPLIT_NAME,
    SCRIPT_DIR,
)


DEFAULT_OUTPUT_DIR = Path("/cv/data/braid/gnesys/datasets/screensQA/results")
SCHEMA_VERSION = 1
PROMPTOPTBASE_ROOT = SCRIPT_DIR.parent.parent
SCREENSQA_SRC = Path("/cv/home/debroue1/from_prescient/projects/screensQA/src")
ENSEMBLE_MODEL_NAME = "LLM RRF Ensemble"

for extra_path in [PROMPTOPTBASE_ROOT, SCREENSQA_SRC]:
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from promptoptbase.scripts.shared_utils import (  # noqa: E402
    load_additional_ground_truth,
    load_all_ground_truth,
    load_all_model_predictions,
)


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[export_harmonized_ensemble {timestamp} UTC] {message}", flush=True)


def safe_model_filename(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", model_name.strip())
    slug = slug.strip("._")
    return f"{slug or 'model'}.json"


def split_sort_key(split_name: str) -> tuple[int, str]:
    order = {"train": 0, "val": 1, "test": 2, NOVEL_SPLIT_NAME: 3, "novel": 3}
    return (order.get(split_name, 99), split_name)


def clean_gene_list(genes: Sequence[str]) -> List[str]:
    return [str(gene).strip() for gene in genes if str(gene).strip()]


def make_record(
    *,
    dataset_name: str,
    split: str,
    split_layout: str,
    predicted_runs: Sequence[Sequence[str]],
    example_key: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runs = [clean_gene_list(run) for run in predicted_runs]
    record: Dict[str, Any] = {
        "dataset_name": str(dataset_name),
        "split": split,
        "split_layout": split_layout,
        "predicted_genes": runs[0] if runs else [],
        "prediction_runs": runs,
        "n_runs": len(runs),
    }
    if example_key is not None:
        record["example_key"] = example_key
    if extra:
        record.update(extra)
    return record


def group_records_by_dataset(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["dataset_name"])].append(record)

    for dataset_name, dataset_records in grouped.items():
        grouped[dataset_name] = sorted(
            dataset_records,
            key=lambda item: (
                split_sort_key(str(item.get("split", ""))),
                str(item.get("split_layout", "")),
                str(item.get("example_key", "")),
            ),
        )
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def build_payload(
    *,
    model_name: str,
    source_files: Sequence[str],
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    grouped = group_records_by_dataset(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "source_group": "ensemble_predictions",
        "source_files": list(source_files),
        "n_records": len(records),
        "n_unique_datasets": len(grouped),
        "records_by_dataset": grouped,
    }


def write_payload(payload: Dict[str, Any], output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_model_filename(payload["model_name"])
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def build_canonical_lookup() -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    year_gt, _ = load_all_ground_truth(DATASET_PATH)
    for example_key, entry in year_gt.items():
        split = example_key.split(":", 1)[0]
        lookup[example_key] = {
            "dataset_name": str(entry["dataset_name"]),
            "split": split,
            "split_layout": "year",
        }

    novel_gt = load_additional_ground_truth(
        split_name=NOVEL_SPLIT_NAME,
        dataset_paths=NOVEL_DATASET_PATHS,
        display_library_genes=True,
    )
    for example_key, entry in novel_gt.items():
        lookup[example_key] = {
            "dataset_name": str(entry["dataset_name"]),
            "split": NOVEL_SPLIT_NAME,
            "split_layout": "year",
        }
    return lookup


def build_ensemble_fn(params: Dict[str, Any], model_list: Sequence[str]):
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


def export_ensemble_payload() -> Dict[str, Any]:
    with open(BEST_ENSEMBLE_ALL_METRICS) as handle:
        best_ensemble = json.load(handle)

    params = best_ensemble["params"]
    model_list = best_ensemble["model_list"]
    canonical_lookup = build_canonical_lookup()
    ensemble_fn = build_ensemble_fn(params, model_list)

    pred_maps = {
        model_name: load_all_model_predictions(
            LLM_PRED_DIR.parent,
            model_name,
            predictions_subdir="llm_predictions",
            additional_splits=[NOVEL_SPLIT_NAME],
        )
        for model_name in model_list
    }

    records: List[Dict[str, Any]] = []
    for example_key, meta in sorted(canonical_lookup.items()):
        split = meta["split"]
        if split not in {"val", "test", NOVEL_SPLIT_NAME}:
            continue
        example_predictions = {}
        max_runs = 0
        for model_name, pred_map in pred_maps.items():
            runs = pred_map.get(example_key, {}).get("predictions", [])
            if runs:
                example_predictions[model_name] = runs
                max_runs = max(max_runs, len(runs))
        if not example_predictions:
            continue

        predicted_runs: List[List[str]] = []
        for run_index in range(max_runs):
            run_predictions = {}
            for model_name, runs in example_predictions.items():
                if run_index < len(runs):
                    run_predictions[model_name] = [runs[run_index]]
            if not run_predictions:
                continue
            predicted_runs.append(ensemble_fn(run_predictions, top_k=100))

        if not predicted_runs:
            continue

        records.append(
            make_record(
                dataset_name=meta["dataset_name"],
                split=split,
                split_layout=meta["split_layout"],
                predicted_runs=predicted_runs,
                example_key=example_key,
                extra={
                    "component_models": list(model_list),
                    "ensemble_config": best_ensemble.get("config"),
                },
            )
        )

    source_files = [str(BEST_ENSEMBLE_ALL_METRICS)]
    for model_name in model_list:
        model_dir = LLM_PRED_DIR / model_name
        source_files.extend(sorted(str(path) for path in model_dir.glob("*_predictions.json")))

    return build_payload(
        model_name=ENSEMBLE_MODEL_NAME,
        source_files=source_files,
        records=records,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export harmonized prediction file for the learned LLM RRF ensemble.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = export_ensemble_payload()
    log(
        f"Prepared {payload['model_name']} with "
        f"{payload['n_records']} records across {payload['n_unique_datasets']} datasets"
    )
    if args.dry_run:
        return
    output_path = write_payload(payload, args.output_dir, overwrite=args.overwrite)
    log(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
