from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from datasets import load_from_disk
from journal_figures_common import DATASET_PATH, NOVEL_DATASET_PATHS, NOVEL_SPLIT_NAME, RESULTS_DIR
from screensqa.dataset.dataset import BioGRIDDSPY
from screensqa.utils.biogrid_maps import stratify_metrics_by_dataset_name


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[update_baseline_coarse_phenotype_novel_public {timestamp} UTC] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate novel_public_2026_dataset predictions for the "
            "baseline/coarse-phenotype-hit-freq model and merge them into the "
            "harmonized baseline JSON."
        )
    )
    parser.add_argument("--train-dataset-path", type=Path, default=Path(DATASET_PATH))
    parser.add_argument(
        "--novel-dataset-path",
        type=Path,
        default=Path(NOVEL_DATASET_PATHS[0]),
    )
    parser.add_argument("--train-split-type", type=str, default="year")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split-name", type=str, default=NOVEL_SPLIT_NAME)
    parser.add_argument("--split-layout", type=str, default="novel")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=RESULTS_DIR / "baseline__coarse-phenotype-hit-freq.json",
    )
    parser.add_argument("--seed", type=int, default=42)
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


def compute_global_hit_frequency(
    train_examples: List[Dict[str, Any]],
) -> Dict[str, float]:
    gene_hit_count: Dict[str, int] = defaultdict(int)
    gene_total_count: Dict[str, int] = defaultdict(int)

    for ex in train_examples:
        for gene, is_hit in zip(ex["relevance_genes"], ex["hit"]):
            gene_total_count[gene] += 1
            if is_hit:
                gene_hit_count[gene] += 1

    freq: Dict[str, float] = {}
    for gene, total in gene_total_count.items():
        freq[gene] = gene_hit_count[gene] / total if total > 0 else 0.0
    return freq


def rank_by_scores(
    screen_genes: List[str],
    score_dict: Dict[str, float],
) -> List[str]:
    scored = [(gene, score_dict.get(gene, 0.0)) for gene in screen_genes]
    random.shuffle(scored)
    scored.sort(key=lambda item: item[1], reverse=True)
    return [gene for gene, _ in scored]


def build_coarse_phenotype_map(
    examples: List[Dict[str, Any]],
) -> Dict[str, str]:
    dataset_names = list({str(ex["dataset_name"]) for ex in examples})
    phenotype_df = stratify_metrics_by_dataset_name({dataset_name: 0 for dataset_name in dataset_names})
    return dict(zip(phenotype_df["dataset_name"], phenotype_df["phenotype"]))


def compute_coarse_phenotype_hit_frequency(
    train_examples: List[Dict[str, Any]],
    ds_to_coarse: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )

    for ex in train_examples:
        coarse = ds_to_coarse.get(str(ex["dataset_name"]), "unknown")
        for gene, is_hit in zip(ex["relevance_genes"], ex["hit"]):
            counts[coarse][gene][1] += 1
            if is_hit:
                counts[coarse][gene][0] += 1

    freq_by_coarse: Dict[str, Dict[str, float]] = {}
    for coarse, gene_counts in counts.items():
        freq_by_coarse[coarse] = {}
        for gene, (hit_count, total_count) in gene_counts.items():
            freq_by_coarse[coarse][gene] = hit_count / total_count if total_count > 0 else 0.0

    return freq_by_coarse


def load_train_examples(
    *,
    train_dataset_path: Path,
    train_split_type: str,
    fold: int,
) -> List[Dict[str, Any]]:
    log(
        f"Loading train pool from {train_dataset_path} using split_type={train_split_type!r}, fold={fold}"
    )
    loader = BioGRIDDSPY(
        dataset_path=str(train_dataset_path),
        split_type=train_split_type,
        fold=fold,
    )
    train_examples, _, _ = loader.get_train_test_split()
    log(f"Loaded {len(train_examples):,} training screens")
    return train_examples


def load_novel_examples(
    *,
    novel_dataset_path: Path,
) -> List[Dict[str, Any]]:
    log(f"Loading novel public screens from {novel_dataset_path}")
    dataset = load_from_disk(str(novel_dataset_path))
    examples = list(dataset)
    log(f"Loaded {len(examples):,} novel public screens")
    return examples


def build_novel_records(
    *,
    novel_examples: Sequence[Dict[str, Any]],
    freq_by_coarse: Dict[str, Dict[str, float]],
    global_freq: Dict[str, float],
    split_name: str,
    split_layout: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, ex in enumerate(novel_examples):
        coarse = str(ex.get("cleaned_phenotype", "")).strip() or "unknown"
        freq = freq_by_coarse.get(coarse, global_freq)
        genes = rank_by_scores(list(ex["relevance_genes"]), freq)
        genes = clean_gene_list(genes)
        records.append(
            {
                "dataset_name": str(ex["dataset_name"]),
                "split": split_name,
                "split_layout": split_layout,
                "predicted_genes": genes,
                "prediction_runs": [genes],
                "n_runs": 1,
                "example_key": f"{split_name}:{index}",
            }
        )
    return sorted(records, key=lambda item: str(item["dataset_name"]))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_examples = load_train_examples(
        train_dataset_path=args.train_dataset_path,
        train_split_type=args.train_split_type,
        fold=args.fold,
    )
    novel_examples = load_novel_examples(
        novel_dataset_path=args.novel_dataset_path,
    )

    global_freq = compute_global_hit_frequency(train_examples)
    ds_to_coarse = build_coarse_phenotype_map(train_examples)
    freq_by_coarse = compute_coarse_phenotype_hit_frequency(train_examples, ds_to_coarse)
    log(f"Computed coarse-phenotype frequencies for {len(freq_by_coarse):,} phenotype buckets")

    missing_cleaned = [
        str(ex.get("dataset_name"))
        for ex in novel_examples
        if not str(ex.get("cleaned_phenotype", "")).strip()
    ]
    if missing_cleaned:
        raise ValueError(
            "Novel examples are missing cleaned_phenotype for: "
            + ", ".join(missing_cleaned)
        )

    novel_records = build_novel_records(
        novel_examples=novel_examples,
        freq_by_coarse=freq_by_coarse,
        global_freq=global_freq,
        split_name=args.split_name,
        split_layout=args.split_layout,
    )

    if args.results_path.exists():
        with open(args.results_path) as handle:
            payload = json.load(handle)
        existing_records = flatten_records_by_dataset(payload.get("records_by_dataset", {}))
        existing_records = [
            record for record in existing_records if str(record.get("split")) != args.split_name
        ]
    else:
        payload = {
            "schema_version": 1,
            "model_name": "baseline/coarse-phenotype-hit-freq",
            "source_group": "baseline_predictions",
            "source_files": [],
        }
        existing_records = []

    combined_records = existing_records + novel_records
    payload["schema_version"] = int(payload.get("schema_version", 1))
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["model_name"] = "baseline/coarse-phenotype-hit-freq"
    payload["source_group"] = "baseline_predictions"
    payload["source_files"] = unique_strings(
        list(payload.get("source_files", []))
        + [
            str(args.train_dataset_path),
            str(args.novel_dataset_path),
            str(Path(__file__).resolve()),
        ]
    )
    payload["n_records"] = len(combined_records)
    payload["n_unique_datasets"] = len(group_records_by_dataset(combined_records))
    payload["records_by_dataset"] = group_records_by_dataset(combined_records)

    log(
        f"Prepared {len(novel_records):,} novel public records for "
        "baseline/coarse-phenotype-hit-freq"
    )
    sample = novel_records[0]
    log(
        f"Example: {sample['dataset_name']} with {len(sample['predicted_genes'])} ranked genes"
    )

    if args.dry_run:
        log(f"Dry run completed without writing {args.results_path}")
        return

    if args.backup and args.results_path.exists():
        backup_path = args.results_path.with_suffix(args.results_path.suffix + ".bak")
        backup_path.write_text(args.results_path.read_text())
        log(f"Wrote backup to {backup_path}")

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    log(f"Updated {args.results_path}")


if __name__ == "__main__":
    main()
