from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from journal_figures_common import NOVEL_DATASET_PATHS, NOVEL_SPLIT_NAME, RESULTS_DIR
from screensqa.dataset.dataset import BioGRIDDSPY


MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "base": {
        "model_name": "gpt-oss-120b",
        "results_path": RESULTS_DIR / "gpt-oss-120b.json",
        "predictions_path": Path("/cv/data/braid/lix361/screensqa_gpt_120B_pred/base_novel_predictions.json"),
        "source_group": "llm_predictions",
        "include_uid": False,
    },
    "sft": {
        "model_name": "SFT (gpt-oss-120B)",
        "results_path": RESULTS_DIR / "SFT__gpt-oss-120B.json",
        "predictions_path": Path("/cv/data/braid/lix361/screensqa_gpt_120B_pred/sft_novel_predictions.json"),
        "source_group": "trained_model",
        "include_uid": True,
    },
    "sft_grpo": {
        "model_name": "SFT + GRPO best (gpt-oss-120B)",
        "results_path": RESULTS_DIR / "SFT__GRPO__best__gpt-oss-120B.json",
        "predictions_path": Path("/cv/data/braid/lix361/screensqa_gpt_120B_pred/step50_novel_predictions.json"),
        "source_group": "trained_model",
        "include_uid": True,
    },
}


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[update_gpt_oss_120b_novel_public {timestamp} UTC] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge novel_public_2026_dataset predictions for the gpt-oss-120B "
            "base, SFT, and SFT+GRPO models into the harmonized results JSONs."
        )
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(MODEL_SPECS),
        default=[],
        help="Only update the selected variant(s). Defaults to all three.",
    )
    parser.add_argument(
        "--novel-dataset-path",
        type=Path,
        default=Path(NOVEL_DATASET_PATHS[0]),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split-name", type=str, default=NOVEL_SPLIT_NAME)
    parser.add_argument("--split-layout", type=str, default="novel")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_gene_list(genes: Sequence[str]) -> List[str]:
    return [str(gene).strip() for gene in genes if str(gene).strip()]


def split_sort_key(split_name: str) -> tuple[int, str]:
    order = {"train": 0, "val": 1, "test": 2, NOVEL_SPLIT_NAME: 3, "novel": 3}
    return (order.get(split_name, 99), split_name)


def group_records_by_dataset(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        dataset_name = str(record["dataset_name"])
        grouped.setdefault(dataset_name, []).append(record)

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


def flatten_records_by_dataset(records_by_dataset: Dict[str, Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for dataset_records in records_by_dataset.values():
        flattened.extend(dataset_records)
    return flattened


def unique_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def load_novel_examples(*, novel_dataset_path: Path, fold: int) -> List[Dict[str, Any]]:
    log(f"Loading novel public screens from {novel_dataset_path}")
    loader = BioGRIDDSPY(
        dataset_path=str(novel_dataset_path),
        split_type=None,
        fold=fold,
    )
    examples = loader.get_dspy_examples()
    log(f"Loaded {len(examples):,} novel public screens")
    return examples


def build_uid_lookup(
    novel_examples: Sequence[Dict[str, Any]],
    *,
    split_name: str,
) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for index, example in enumerate(novel_examples):
        dataset_name = str(example["dataset_name"])
        uid = f"novel_{dataset_name}"
        lookup[uid] = {
            "dataset_name": dataset_name,
            "example_key": f"{split_name}:{index}",
        }
    return lookup


def build_new_records(
    *,
    predictions_path: Path,
    uid_lookup: Dict[str, Dict[str, Any]],
    split_name: str,
    split_layout: str,
    include_uid: bool,
) -> List[Dict[str, Any]]:
    with open(predictions_path) as handle:
        predictions = json.load(handle)

    new_records: List[Dict[str, Any]] = []
    seen_datasets = set()
    for prediction in predictions:
        uid = str(prediction.get("uid", "")).strip()
        meta = uid_lookup.get(uid)
        if meta is None:
            raise KeyError(f"Unknown novel uid {uid!r} in {predictions_path}")

        dataset_name = str(meta["dataset_name"])
        if dataset_name in seen_datasets:
            raise ValueError(f"Duplicate dataset {dataset_name!r} in {predictions_path}")
        seen_datasets.add(dataset_name)

        genes = clean_gene_list(prediction.get("genes", []))[:100]
        record: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "split": split_name,
            "split_layout": split_layout,
            "predicted_genes": genes,
            "prediction_runs": [genes],
            "n_runs": 1,
            "example_key": str(meta["example_key"]),
        }
        if include_uid:
            record["uid"] = uid
        new_records.append(record)

    if len(new_records) != len(uid_lookup):
        missing = sorted(set(meta["dataset_name"] for meta in uid_lookup.values()) - seen_datasets)
        raise ValueError(
            f"Expected {len(uid_lookup)} novel screens in {predictions_path}, "
            f"found {len(new_records)}. Missing: {missing}"
        )

    return sorted(new_records, key=lambda item: str(item["dataset_name"]))


def update_one_model(
    *,
    spec: Dict[str, Any],
    novel_dataset_path: Path,
    uid_lookup: Dict[str, Dict[str, Any]],
    split_name: str,
    split_layout: str,
    backup: bool,
    dry_run: bool,
) -> None:
    results_path = Path(spec["results_path"])
    predictions_path = Path(spec["predictions_path"])
    include_uid = bool(spec["include_uid"])

    new_records = build_new_records(
        predictions_path=predictions_path,
        uid_lookup=uid_lookup,
        split_name=split_name,
        split_layout=split_layout,
        include_uid=include_uid,
    )

    if results_path.exists():
        with open(results_path) as handle:
            payload = json.load(handle)
        existing_records = flatten_records_by_dataset(payload.get("records_by_dataset", {}))
        existing_records = [
            record for record in existing_records if str(record.get("split")) != split_name
        ]
    else:
        payload = {
            "schema_version": 1,
            "model_name": spec["model_name"],
            "source_group": spec["source_group"],
            "source_files": [],
        }
        existing_records = []

    combined_records = existing_records + new_records
    payload["schema_version"] = int(payload.get("schema_version", 1))
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["model_name"] = spec["model_name"]
    payload["source_group"] = spec["source_group"]
    payload["source_files"] = unique_strings(
        list(payload.get("source_files", []))
        + [
            str(predictions_path),
            str(novel_dataset_path),
            str(Path(__file__).resolve()),
        ]
    )
    payload["n_records"] = len(combined_records)
    payload["n_unique_datasets"] = len(group_records_by_dataset(combined_records))
    payload["records_by_dataset"] = group_records_by_dataset(combined_records)

    log(
        f"Prepared {len(new_records):,} novel public records for "
        f"{spec['model_name']}"
    )

    if dry_run:
        log(f"Dry run completed without writing {results_path}")
        return

    if backup and results_path.exists():
        backup_path = results_path.with_suffix(results_path.suffix + ".bak")
        backup_path.write_text(results_path.read_text())
        log(f"Wrote backup to {backup_path}")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    log(f"Updated {results_path}")


def main() -> None:
    args = parse_args()
    variants = args.variant or list(MODEL_SPECS.keys())

    novel_examples = load_novel_examples(
        novel_dataset_path=args.novel_dataset_path,
        fold=args.fold,
    )
    uid_lookup = build_uid_lookup(
        novel_examples,
        split_name=args.split_name,
    )

    for variant in variants:
        update_one_model(
            spec=MODEL_SPECS[variant],
            novel_dataset_path=args.novel_dataset_path,
            uid_lookup=uid_lookup,
            split_name=args.split_name,
            split_layout=args.split_layout,
            backup=args.backup,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
