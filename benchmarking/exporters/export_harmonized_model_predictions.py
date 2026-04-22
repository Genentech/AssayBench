from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from journal_figures_common import (
    CLASSIFIER_RANDOM_CSV,
    CLASSIFIER_RECENT_CSV,
    CLASSIFIER_YEAR_CSV,
    DATASET_PATH,
    EXTERNAL_MODELS,
    LLM_PRED_DIR,
    NOVEL_DATASET_PATHS,
    NOVEL_SPLIT_NAME,
    SCRIPT_DIR,
    STAGED_MODEL_PREDICTIONS,
    _FINAL_ANSWER_RE,
)
from export_harmonized_ensemble_predictions import (
    ENSEMBLE_MODEL_NAME,
    export_ensemble_payload,
)
from export_harmonized_knn_predictions import export_knn_payloads


DEFAULT_OUTPUT_DIR = Path("/cv/data/braid/gnesys/datasets/screensQA/results")
SCHEMA_VERSION = 1
PROMPTOPTBASE_ROOT = SCRIPT_DIR.parent.parent
SCREENSQA_SRC = Path("/cv/home/debroue1/from_prescient/projects/screensQA/src")
GEPA_DIR = SCRIPT_DIR.parent / "output" / "gepa"
CLASSIFIER_OUTPUT_ROOT = Path("/cv/data/braid/debroue1/promptoptbase/outputs")
CLASSIFIER_RUN_IDS = {
    "year": "r3ph5o3v",
    "random": "rp8xcp30",
    "novel": "r3ph5o3v",
}

for extra_path in [PROMPTOPTBASE_ROOT, SCREENSQA_SRC]:
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[export_harmonized_predictions {timestamp} UTC] {message}", flush=True)


def get_shared_utils() -> tuple[Any, Any, Any]:
    from promptoptbase.scripts.shared_utils import (
        load_additional_ground_truth,
        load_all_ground_truth,
        load_all_model_predictions,
    )

    return load_all_ground_truth, load_additional_ground_truth, load_all_model_predictions


def _ensure_csv_field_limit() -> None:
    csv.field_size_limit(sys.maxsize)


def clean_gene_list(genes: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for gene in genes:
        text = _FINAL_ANSWER_RE.sub("", str(gene)).strip()
        if text:
            cleaned.append(text)
    return cleaned


def safe_model_filename(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", model_name.strip())
    slug = slug.strip("._")
    return f"{slug or 'model'}.json"


def split_sort_key(split_name: str) -> tuple[int, str]:
    order = {"train": 0, "val": 1, "test": 2, NOVEL_SPLIT_NAME: 3, "novel": 3}
    return (order.get(split_name, 99), split_name)


def normalized_export_model_name(model_name: str) -> str:
    if model_name.startswith("fewshot/"):
        return model_name.removeprefix("fewshot/")
    return model_name


def model_name_is_selected(model_name: str, selected_models: Optional[set[str]]) -> bool:
    if not selected_models:
        return True
    normalized = normalized_export_model_name(model_name)
    legacy_fewshot = f"fewshot/{normalized}"
    return (
        model_name in selected_models
        or normalized in selected_models
        or legacy_fewshot in selected_models
    )


def build_canonical_lookup() -> Dict[str, Dict[str, Any]]:
    load_all_ground_truth, load_additional_ground_truth, _ = get_shared_utils()
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
            "split_layout": "novel",
        }

    return lookup


def make_record(
    *,
    dataset_name: str,
    split: str,
    split_layout: str,
    predicted_runs: Sequence[Sequence[str]],
    example_key: Optional[str] = None,
    source_files: Optional[Sequence[str]] = None,
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
    if source_files:
        record["source_files"] = list(source_files)
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
                str(item.get("run_id", "")),
            ),
        )
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def build_payload(
    *,
    model_name: str,
    source_group: str,
    source_files: Sequence[str],
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    grouped = group_records_by_dataset(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "source_group": source_group,
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


def export_llm_like_model(
    *,
    model_name: str,
    source_group: str,
    predictions_subdir: str,
    additional_splits: Optional[Sequence[str]],
    canonical_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    _, _, load_all_model_predictions = get_shared_utils()
    pred_map = load_all_model_predictions(
        LLM_PRED_DIR.parent,
        model_name,
        predictions_subdir=predictions_subdir,
        additional_splits=list(additional_splits or []),
    )
    records: List[Dict[str, Any]] = []
    for example_key, entry in sorted(pred_map.items()):
        meta = canonical_lookup.get(example_key)
        if meta is None:
            continue
        records.append(
            make_record(
                dataset_name=meta["dataset_name"],
                split=meta["split"],
                split_layout=meta["split_layout"],
                predicted_runs=entry.get("predictions", []),
                example_key=example_key,
            )
        )
    source_root = LLM_PRED_DIR.parent / predictions_subdir / model_name
    return build_payload(
        model_name=model_name,
        source_group=source_group,
        source_files=sorted(str(path) for path in source_root.glob("*_predictions.json")),
        records=records,
    )


def export_gepa_model(
    *,
    model_name: str,
    canonical_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    _, _, load_all_model_predictions = get_shared_utils()
    gepa_model_dir = GEPA_DIR / model_name
    pred_map = load_all_model_predictions(
        gepa_model_dir,
        model_name,
        predictions_subdir="llm_predictions",
        additional_splits=[],
    )
    records: List[Dict[str, Any]] = []
    for example_key, entry in sorted(pred_map.items()):
        meta = canonical_lookup.get(example_key)
        if meta is None:
            continue
        records.append(
            make_record(
                dataset_name=meta["dataset_name"],
                split=meta["split"],
                split_layout=meta["split_layout"],
                predicted_runs=entry.get("predictions", []),
                example_key=example_key,
            )
        )
    source_root = gepa_model_dir / "llm_predictions" / model_name
    source_files = sorted(str(path) for path in source_root.glob("*_predictions.json"))
    return build_payload(
        model_name=f"gepa/{model_name}",
        source_group="gepa_predictions",
        source_files=source_files,
        records=records,
    )


def _load_json(path: Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def export_external_grpo_model(
    *,
    model_name: str,
    config: Dict[str, Any],
    canonical_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    pred_dir = Path(config["path"])
    step = config.get("step", "step130")
    records: List[Dict[str, Any]] = []
    source_files: List[str] = []
    for split in ("val", "test"):
        path = pred_dir / f"{step}_{split}_predictions.json"
        if not path.exists():
            candidates = sorted(pred_dir.glob(f"*_{split}_predictions.json"))
            if not candidates:
                continue
            path = candidates[0]
        source_files.append(str(path))
        data = _load_json(path)
        for index, prediction in enumerate(data):
            example_key = f"{split}:{index}"
            meta = canonical_lookup.get(example_key)
            if meta is None:
                continue
            records.append(
                make_record(
                    dataset_name=meta["dataset_name"],
                    split=split,
                    split_layout=meta["split_layout"],
                    predicted_runs=[prediction.get("genes", [])],
                    example_key=example_key,
                    extra={"uid": prediction.get("uid")},
                )
            )
    return build_payload(
        model_name=model_name,
        source_group="trained_model",
        source_files=source_files,
        records=records,
    )


def export_external_c2s_model(
    *,
    model_name: str,
    config: Dict[str, Any],
    canonical_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    pred_dir = Path(config["path"])
    records: List[Dict[str, Any]] = []
    source_files: List[str] = []
    for split in ("val", "test"):
        path = pred_dir / f"{split}_predictions.json"
        if not path.exists():
            continue
        source_files.append(str(path))
        data = _load_json(path)
        for index, prediction in enumerate(data.get("predictions", [])):
            example_key = f"{split}:{index}"
            meta = canonical_lookup.get(example_key)
            if meta is None:
                continue
            records.append(
                make_record(
                    dataset_name=meta["dataset_name"],
                    split=split,
                    split_layout=meta["split_layout"],
                    predicted_runs=prediction.get("predictions", []),
                    example_key=example_key,
                )
            )
    return build_payload(
        model_name=model_name,
        source_group="trained_model",
        source_files=source_files,
        records=records,
    )


def export_list_prediction_model(
    *,
    model_name: str,
    config: Dict[str, Any],
    canonical_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    source_files: List[str] = []
    for split, key in (("val", "val_path"), ("test", "test_path")):
        path = Path(config[key])
        if not path.exists():
            continue
        source_files.append(str(path))
        data = _load_json(path)
        for index, prediction in enumerate(data):
            example_key = f"{split}:{index}"
            meta = canonical_lookup.get(example_key)
            if meta is None:
                continue
            records.append(
                make_record(
                    dataset_name=meta["dataset_name"],
                    split=split,
                    split_layout=meta["split_layout"],
                    predicted_runs=[prediction.get("genes", [])],
                    example_key=example_key,
                    extra={"uid": prediction.get("uid")},
                )
            )
    return build_payload(
        model_name=model_name,
        source_group="trained_model",
        source_files=source_files,
        records=records,
    )


def parse_gene_list_column(value: str) -> List[str]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return clean_gene_list(parsed)


def _find_classifier_output_run_dir(run_id: str) -> Optional[Path]:
    wandb_dir = CLASSIFIER_OUTPUT_ROOT / "wandb"
    matches = sorted(wandb_dir.glob(f"run-*-{run_id}"))
    return matches[0] if matches else None


def _rank_genes_from_classifier_scores(score_map: Dict[str, Any]) -> List[str]:
    cleaned_scores: List[tuple[str, float]] = []
    for gene, score in score_map.items():
        gene_name = str(gene).strip()
        if not gene_name:
            continue
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            continue
        cleaned_scores.append((gene_name, score_value))
    cleaned_scores.sort(key=lambda item: (-item[1], item[0]))
    return [gene for gene, _ in cleaned_scores]


def _load_classifier_raw_predictions(
    *,
    run_id: str,
    split_layout: str,
) -> tuple[List[Dict[str, Any]], List[str]]:
    run_dir = _find_classifier_output_run_dir(run_id)
    if run_dir is None:
        raise FileNotFoundError(f"No classifier output directory found for run_id={run_id}")

    files_dir = run_dir / "files"
    source_files: List[str] = []
    records: List[Dict[str, Any]] = []
    for split in ("val", "test"):
        path = files_dir / "predictions" / f"{split}_predictions.json"
        if not path.exists():
            continue
        source_files.append(str(path))
        data = _load_json(path)
        for prediction in data.get("predictions", []):
            dataset_name = str(prediction.get("dataset_name", "")).strip()
            if not dataset_name:
                continue
            predicted_scores = prediction.get("predicted_scores", [])
            score_map = predicted_scores[0] if predicted_scores and isinstance(predicted_scores[0], dict) else {}
            if not score_map:
                continue
            ranked_genes = _rank_genes_from_classifier_scores(score_map)
            records.append(
                make_record(
                    dataset_name=dataset_name,
                    split=split,
                    split_layout=split_layout,
                    predicted_runs=[ranked_genes],
                    source_files=[str(path)],
                    extra={"run_id": run_id},
                )
            )
    return records, source_files


def _load_classifier_novel_records(path: Path, run_id: str) -> List[Dict[str, Any]]:
    _ensure_csv_field_limit()
    records: List[Dict[str, Any]] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            dataset_name = str(row.get("dataset_name", "")).strip()
            if not dataset_name:
                continue
            predicted_genes = parse_gene_list_column(row.get("predicted_genes", ""))
            if not predicted_genes:
                continue
            records.append(
                make_record(
                    dataset_name=dataset_name,
                    split=NOVEL_SPLIT_NAME,
                    split_layout="novel",
                    predicted_runs=[predicted_genes],
                    source_files=[str(path)],
                    extra={
                        "run_id": run_id,
                        "row_index": row_index,
                    },
                )
            )
    return records


def export_classifier_predictions() -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    source_files: List[str] = []

    year_records, year_sources = _load_classifier_raw_predictions(
        run_id=CLASSIFIER_RUN_IDS["year"],
        split_layout="year",
    )
    random_records, random_sources = _load_classifier_raw_predictions(
        run_id=CLASSIFIER_RUN_IDS["random"],
        split_layout="random",
    )
    records.extend(year_records)
    records.extend(random_records)
    source_files.extend(year_sources)
    source_files.extend(random_sources)

    if CLASSIFIER_RECENT_CSV.exists():
        records.extend(
            _load_classifier_novel_records(
                CLASSIFIER_RECENT_CSV,
                run_id=CLASSIFIER_RUN_IDS["novel"],
            )
        )
        source_files.append(str(CLASSIFIER_RECENT_CSV))

    return build_payload(
        model_name="Classifier",
        source_group="classifier",
        source_files=sorted(set(source_files)),
        records=records,
    )


def discover_llm_models() -> List[str]:
    models: List[str] = []
    for path in sorted(LLM_PRED_DIR.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("_"):
            continue
        if not any(path.glob("*_predictions.json")):
            continue
        models.append(path.name)
    return models


def discover_baseline_models() -> List[str]:
    baseline_dir = LLM_PRED_DIR.parent / "baseline_predictions"
    models: List[str] = []
    if not baseline_dir.exists():
        return models
    for path in sorted(baseline_dir.iterdir()):
        if not path.is_dir():
            continue
        if not any(path.glob("*_predictions.json")):
            continue
        models.append(f"baseline/{path.name}")
    return models


def discover_fewshot_models() -> List[str]:
    fewshot_dir = LLM_PRED_DIR.parent / "fewshot_predictions"
    models: List[str] = []
    if not fewshot_dir.exists():
        return models
    for path in sorted(fewshot_dir.iterdir()):
        if not path.is_dir():
            continue
        if not any(path.glob("*_predictions.json")):
            continue
        models.append(path.name)
    return models


def discover_gepa_models() -> List[str]:
    models: List[str] = []
    if not GEPA_DIR.exists():
        return models
    for path in sorted(GEPA_DIR.iterdir()):
        if not path.is_dir():
            continue
        pred_dir = path / "llm_predictions" / path.name
        has_prediction_files = any(pred_dir.glob("*_predictions.json"))
        has_val_fallback = (path / "generated_best_outputs_valset").exists()
        if not has_prediction_files and not has_val_fallback:
            continue
        models.append(f"gepa/{path.name}")
    return models


def build_manifest_entry(payload: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    return {
        "model_name": payload["model_name"],
        "source_group": payload["source_group"],
        "output_path": str(output_path),
        "n_records": payload["n_records"],
        "n_unique_datasets": payload["n_unique_datasets"],
        "source_files": payload["source_files"],
    }


def export_model_payloads(
    *,
    include_llms: bool,
    include_baselines: bool,
    include_trained: bool,
    include_classifier: bool,
    selected_models: Optional[set[str]],
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    canonical_lookup: Optional[Dict[str, Dict[str, Any]]] = None

    def get_canonical_lookup() -> Dict[str, Dict[str, Any]]:
        nonlocal canonical_lookup
        if canonical_lookup is None:
            canonical_lookup = build_canonical_lookup()
        return canonical_lookup

    if include_llms:
        for model_name in discover_llm_models():
            if not model_name_is_selected(model_name, selected_models):
                continue
            log(f"Loading canonical LLM predictions for {model_name}")
            payloads.append(
                export_llm_like_model(
                    model_name=model_name,
                    source_group="llm_predictions",
                    predictions_subdir="llm_predictions",
                    additional_splits=[NOVEL_SPLIT_NAME],
                    canonical_lookup=get_canonical_lookup(),
                )
            )

    if include_baselines:
        for model_name in discover_baseline_models():
            if not model_name_is_selected(model_name, selected_models):
                continue
            log(f"Loading baseline predictions for {model_name}")
            payloads.append(
                export_llm_like_model(
                    model_name=model_name.removeprefix("baseline/"),
                    source_group="baseline_predictions",
                    predictions_subdir="baseline_predictions",
                    additional_splits=[NOVEL_SPLIT_NAME],
                    canonical_lookup=get_canonical_lookup(),
                )
            )
            payloads[-1]["model_name"] = model_name

    if include_llms:
        for model_name in discover_fewshot_models():
            if not model_name_is_selected(model_name, selected_models):
                continue
            log(f"Loading few-shot predictions for {model_name}")
            payloads.append(
                export_llm_like_model(
                    model_name=model_name,
                    source_group="fewshot_predictions",
                    predictions_subdir="fewshot_predictions",
                    additional_splits=[NOVEL_SPLIT_NAME],
                    canonical_lookup=get_canonical_lookup(),
                )
            )

    if include_llms:
        for model_name in discover_gepa_models():
            if not model_name_is_selected(model_name, selected_models):
                continue
            raw_model_name = model_name.removeprefix("gepa/")
            log(f"Loading GEPA predictions for {model_name}")
            payloads.append(
                export_gepa_model(
                    model_name=raw_model_name,
                    canonical_lookup=get_canonical_lookup(),
                )
            )

    if include_trained:
        for model_name, config in EXTERNAL_MODELS.items():
            if not model_name_is_selected(model_name, selected_models):
                continue
            log(f"Loading trained-model predictions for {model_name}")
            if config.get("source") == "grpo":
                payloads.append(
                    export_external_grpo_model(
                        model_name=model_name,
                        config=config,
                        canonical_lookup=get_canonical_lookup(),
                    )
                )
            elif config.get("source") == "c2s":
                payloads.append(
                    export_external_c2s_model(
                        model_name=model_name,
                        config=config,
                        canonical_lookup=get_canonical_lookup(),
                    )
                )

        for model_name, config in STAGED_MODEL_PREDICTIONS.items():
            if not model_name_is_selected(model_name, selected_models):
                continue
            log(f"Loading staged-model predictions for {model_name}")
            payloads.append(
                export_list_prediction_model(
                    model_name=model_name,
                    config=config,
                    canonical_lookup=get_canonical_lookup(),
                )
            )

    if include_classifier and model_name_is_selected("Classifier", selected_models):
        log("Loading classifier predictions from result CSVs")
        payloads.append(export_classifier_predictions())

    if model_name_is_selected(ENSEMBLE_MODEL_NAME, selected_models):
        ensemble_payload = export_ensemble_payload()
        log(f"Loading reconstructed ensemble predictions for {ensemble_payload['model_name']}")
        payloads.append(ensemble_payload)

    knn_payloads = export_knn_payloads(selected_models=selected_models)
    for payload in knn_payloads:
        log(f"Loading reconstructed kNN predictions for {payload['model_name']}")
    payloads.extend(knn_payloads)

    return payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export harmonized prediction files that map dataset_name to ranked genes "
            "for each model or baseline source."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where harmonized JSON files will be written (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Limit export to a specific model name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--skip-llms",
        action="store_true",
        help="Skip canonical LLM prediction exports.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline prediction exports.",
    )
    parser.add_argument(
        "--skip-trained",
        action="store_true",
        help="Skip trained and staged model exports.",
    )
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        help="Skip classifier prediction export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing harmonized files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build payloads and print a summary without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_models = set(args.model) if args.model else None

    payloads = export_model_payloads(
        include_llms=not args.skip_llms,
        include_baselines=not args.skip_baselines,
        include_trained=not args.skip_trained,
        include_classifier=not args.skip_classifier,
        selected_models=selected_models,
    )

    manifest_entries: List[Dict[str, Any]] = []
    for payload in payloads:
        log(
            f"Prepared {payload['model_name']} with "
            f"{payload['n_records']} records across {payload['n_unique_datasets']} datasets"
        )
        if args.dry_run:
            continue
        output_path = write_payload(payload, args.output_dir, overwrite=args.overwrite)
        manifest_entries.append(build_manifest_entry(payload, output_path))

    if args.dry_run:
        log(f"Dry run completed for {len(payloads)} models")
        return

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir),
        "n_models": len(manifest_entries),
        "models": manifest_entries,
    }
    manifest_path = args.output_dir / "harmonized_predictions_manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    log(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
