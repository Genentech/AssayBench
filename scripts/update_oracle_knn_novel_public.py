from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from journal_figures_common import DATASET_PATH, NOVEL_DATASET_PATHS, NOVEL_SPLIT_NAME, RESULTS_DIR
from screensqa.dataset.dataset import BioGRIDDSPY

try:
    from run_ensemble_baseline import create_dspy_examples
except ImportError:
    from scripts.run_ensemble_baseline import create_dspy_examples


from screensqa.benchmark.ranking_metrics import RankingMetrics

def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[update_oracle_knn_novel_public {timestamp} UTC] {message}", flush=True)


LEGACY_KNN_RESULTS_PATH = Path(
    "/cv/home/edwarc24/code/PromptOptBioGrid/promptoptbase/output_latent_biology/knn_test/year_fold0/knn_results.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append or replace Oracle/Embedding kNN harmonized predictions for the "
            "novel public 2026 dataset."
        )
    )
    parser.add_argument(
        "--method",
        choices=("oracle", "embedding"),
        default="oracle",
        help="Which kNN baseline to update.",
    )
    parser.add_argument("--train-dataset-path", type=Path, default=Path(DATASET_PATH))
    parser.add_argument(
        "--novel-dataset-path",
        type=Path,
        default=Path(NOVEL_DATASET_PATHS[0]),
    )
    parser.add_argument("--train-split-type", type=str, default="year")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--split-name", type=str, default=NOVEL_SPLIT_NAME)
    parser.add_argument("--split-layout", type=str, default="novel")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--legacy-knn-results-path",
        type=Path,
        default=LEGACY_KNN_RESULTS_PATH,
        help="Legacy kNN results JSON used to source precomputed embedding novel matches.",
    )
    parser.add_argument(
        "--require-matching-reverse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only match train screens with the same reverse flag as the target screen.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a timestamp-free .bak copy next to the updated Oracle__kNN.json file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute records and print a summary without writing the JSON file.",
    )
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


def format_screen_name(dataset_name: str, reverse: bool) -> str:
    direction = "reverse" if reverse else "forward"
    return f"{dataset_name} ({direction})"


def rank_screen_genes(example: Any, top_k: int) -> List[str]:
    ranked_pairs = sorted(
        zip(example.genes, example.relevance_scores),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return clean_gene_list([gene for gene, _ in ranked_pairs[:top_k]])


def load_train_pool(
    *,
    train_dataset_path: Path,
    train_split_type: str,
    fold: int,
    top_k: int,
) -> tuple[List[Any], List[List[str]]]:
    log(
        f"Loading training pool from {train_dataset_path} using split_type={train_split_type!r}, fold={fold}"
    )
    train_loader = BioGRIDDSPY(
        dataset_path=str(train_dataset_path),
        split_type=train_split_type,
        fold=fold,
    )
    train_examples, _, _ = train_loader.get_train_test_split()
    train_dspy = create_dspy_examples(train_examples)
    log(f"Loaded {len(train_dspy):,} train screens")
    train_predictions = [rank_screen_genes(example, top_k=top_k) for example in train_dspy]
    return train_dspy, train_predictions


def load_novel_public_examples(
    *,
    novel_dataset_path: Path,
    fold: int,
) -> List[Any]:
    log(f"Loading novel public screens from {novel_dataset_path}")
    novel_loader = BioGRIDDSPY(
        dataset_path=str(novel_dataset_path),
        split_type=None,
        fold=fold,
    )
    novel_examples = novel_loader.get_dspy_examples()
    novel_dspy = create_dspy_examples(novel_examples)
    log(f"Loaded {len(novel_dspy):,} novel public screens")
    return novel_dspy


def compute_dcg(relevances: Sequence[float | None], k: int) -> float:
    penalized = list(relevances)
    if len(penalized) < k:
        penalized.extend([0.0] * (k - len(penalized)))
    penalized = penalized[:k]
    condensed = [rel for rel in penalized if rel is not None][:k]
    if not condensed:
        return 0.0
    return float(sum(rel / np.log2(index + 2) for index, rel in enumerate(condensed)))


def compute_adjusted_ndcg_at_k(
    predicted_genes: Sequence[str],
    target_score_lookup: Dict[str, float],
    target_relevance_scores: Sequence[float],
    k: int,
) -> float:
    deduped_predictions = list(dict.fromkeys(str(gene).strip().upper() for gene in predicted_genes if str(gene).strip()))
    predicted_relevances = [target_score_lookup.get(gene) for gene in deduped_predictions]

    dcg = compute_dcg(predicted_relevances, k)

    ideal_relevances = sorted((max(float(score), 0.0) for score in target_relevance_scores), reverse=True)
    idcg = compute_dcg(ideal_relevances, k)
    if idcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    mean_relevance = float(np.mean(target_relevance_scores)) if target_relevance_scores else 0.0
    random_relevances = [mean_relevance] * min(k, len(target_relevance_scores))
    ndcg_rand = compute_dcg(random_relevances, k) / idcg
    adjusted = (ndcg - ndcg_rand) / (1 - ndcg_rand)
    return max(float(adjusted), 0.0)


def build_novel_public_oracle_records(
    *,
    train_dataset_path: Path,
    novel_dataset_path: Path,
    train_split_type: str,
    fold: int,
    split_name: str,
    split_layout: str,
    top_k: int,
    require_matching_reverse: bool,
) -> List[Dict[str, Any]]:
    train_dspy, train_predictions = load_train_pool(
        train_dataset_path=train_dataset_path,
        train_split_type=train_split_type,
        fold=fold,
        top_k=top_k,
    )
    novel_dspy = load_novel_public_examples(
        novel_dataset_path=novel_dataset_path,
        fold=fold,
    )

    scorer = RankingMetrics(k_values=[5, 10, 20, 50, 100])

    records: List[Dict[str, Any]] = []
    for target_index, target in enumerate(novel_dspy):
        print(target["dataset_name"])
        target_reverse = bool(getattr(target, "reverse", False))
        target_score_lookup = {
            str(gene).strip().upper(): float(score)
            for gene, score in zip(target.genes, target.relevance_scores)
            if str(gene).strip()
        }
        best_train_index = None
        best_score = float("-inf")

        for train_index, train in enumerate(train_dspy):
            train_reverse = bool(getattr(train, "reverse", False))
            if require_matching_reverse and train_reverse != target_reverse:
                continue
            
            scores = scorer.evaluate(predicted_genes = train_predictions[train_index], 
                                     ground_truth_genes = target.genes,
                                     relevance_scores=target.relevance_scores)
            score = scores["adjusted_ndcg@100"]

            score_ = compute_adjusted_ndcg_at_k(
                predicted_genes=train_predictions[train_index],
                target_score_lookup=target_score_lookup,
                target_relevance_scores=target.relevance_scores,
                k=top_k,
            )
            if score > best_score:
                best_score = score
                best_train_index = train_index

        if best_train_index is None:
            raise RuntimeError(
                f"No Oracle kNN match found for {target.dataset_name!r}. "
                "Try disabling --require-matching-reverse."
            )

        matched_train = train_dspy[best_train_index]
        predicted_genes = train_predictions[best_train_index]
        records.append(
            {
                "dataset_name": str(target.dataset_name),
                "split": split_name,
                "split_layout": split_layout,
                "predicted_genes": predicted_genes,
                "prediction_runs": [predicted_genes],
                "n_runs": 1,
                "example_key": f"{split_name}:{target_index}",
                "target_screen_name": format_screen_name(str(target.dataset_name), target_reverse),
                "matched_train_name": format_screen_name(
                    str(matched_train.dataset_name),
                    bool(getattr(matched_train, "reverse", False)),
                ),
                "matched_train_index": int(best_train_index),
                "matched_train_dataset_name": str(matched_train.dataset_name),
            }
        )

    return records


def build_novel_public_embedding_records(
    *,
    train_dataset_path: Path,
    novel_dataset_path: Path,
    train_split_type: str,
    fold: int,
    split_name: str,
    split_layout: str,
    top_k: int,
    legacy_knn_results_path: Path,
) -> List[Dict[str, Any]]:
    train_dspy, train_predictions = load_train_pool(
        train_dataset_path=train_dataset_path,
        train_split_type=train_split_type,
        fold=fold,
        top_k=top_k,
    )
    novel_dspy = load_novel_public_examples(
        novel_dataset_path=novel_dataset_path,
        fold=fold,
    )

    if not legacy_knn_results_path.exists():
        raise FileNotFoundError(
            f"Legacy kNN results not found at {legacy_knn_results_path}"
        )
    with open(legacy_knn_results_path) as handle:
        legacy_knn_results = json.load(handle)

    target_names = legacy_knn_results.get("novel_screens", [])
    embedding_novel = legacy_knn_results.get("embedding", {}).get("novel", {})
    matched_train_names = embedding_novel.get("matched_train_names", [])
    matched_train_indices = embedding_novel.get("matched_train_indices", [])

    n_examples = min(len(novel_dspy), len(target_names), len(matched_train_names), len(matched_train_indices))
    if n_examples != len(novel_dspy):
        raise RuntimeError(
            "Legacy embedding novel matches do not align with the novel public dataset size: "
            f"{len(novel_dspy)} targets vs {n_examples} usable matches."
        )

    train_name_by_index = {
        index: format_screen_name(
            str(example.dataset_name),
            bool(getattr(example, "reverse", False)),
        )
        for index, example in enumerate(train_dspy)
    }

    records: List[Dict[str, Any]] = []
    for target_index in range(n_examples):
        target = novel_dspy[target_index]
        matched_train_index = int(matched_train_indices[target_index])
        if matched_train_index < 0 or matched_train_index >= len(train_dspy):
            raise IndexError(
                f"Matched train index {matched_train_index} is out of bounds for target #{target_index}"
            )
        matched_train = train_dspy[matched_train_index]
        predicted_genes = train_predictions[matched_train_index]
        target_screen_name = target_names[target_index]
        matched_train_name = matched_train_names[target_index]
        expected_train_name = train_name_by_index[matched_train_index]
        if matched_train_name != expected_train_name:
            log(
                "Warning: legacy embedding match name does not equal reconstructed train name "
                f"for index {matched_train_index}: {matched_train_name!r} vs {expected_train_name!r}"
            )

        records.append(
            {
                "dataset_name": str(target.dataset_name),
                "split": split_name,
                "split_layout": split_layout,
                "predicted_genes": predicted_genes,
                "prediction_runs": [predicted_genes],
                "n_runs": 1,
                "example_key": f"{split_name}:{target_index}",
                "target_screen_name": target_screen_name,
                "matched_train_name": matched_train_name,
                "matched_train_index": matched_train_index,
                "matched_train_dataset_name": str(matched_train.dataset_name),
            }
        )

    return records


def unique_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        text = str(value)
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def main() -> None:
    args = parse_args()
    method_label = "Oracle kNN" if args.method == "oracle" else "Embedding kNN"
    results_path = args.results_path
    if results_path is None:
        filename = "Oracle__kNN.json" if args.method == "oracle" else "Embedding__kNN.json"
        results_path = RESULTS_DIR / filename

    if args.method == "oracle":
        new_records = build_novel_public_oracle_records(
            train_dataset_path=args.train_dataset_path,
            novel_dataset_path=args.novel_dataset_path,
            train_split_type=args.train_split_type,
            fold=args.fold,
            split_name=args.split_name,
            split_layout=args.split_layout,
            top_k=args.top_k,
            require_matching_reverse=args.require_matching_reverse,
        )
    else:
        new_records = build_novel_public_embedding_records(
            train_dataset_path=args.train_dataset_path,
            novel_dataset_path=args.novel_dataset_path,
            train_split_type=args.train_split_type,
            fold=args.fold,
            split_name=args.split_name,
            split_layout=args.split_layout,
            top_k=args.top_k,
            legacy_knn_results_path=args.legacy_knn_results_path,
        )

    payload: Dict[str, Any]
    if results_path.exists():
        with open(results_path) as handle:
            payload = json.load(handle)
        existing_records = flatten_records_by_dataset(payload.get("records_by_dataset", {}))
        existing_records = [
            record
            for record in existing_records
            if str(record.get("split")) != args.split_name
        ]
    else:
        payload = {
            "schema_version": 1,
            "model_name": method_label,
            "source_group": "knn_predictions",
            "source_files": [],
        }
        existing_records = []

    combined_records = existing_records + new_records
    grouped = group_records_by_dataset(combined_records)
    payload["schema_version"] = int(payload.get("schema_version", 1))
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["model_name"] = method_label
    payload["source_group"] = "knn_predictions"
    payload["source_files"] = unique_strings(
        list(payload.get("source_files", []))
        + [
            str(args.train_dataset_path),
            str(args.novel_dataset_path),
            str(args.legacy_knn_results_path) if args.method == "embedding" else "",
            str(Path(__file__).resolve()),
        ]
    )
    payload["n_records"] = len(combined_records)
    payload["n_unique_datasets"] = len(grouped)
    payload["records_by_dataset"] = grouped

    log(
        f"Prepared {len(new_records):,} novel public {method_label} records "
        f"across {len({record['dataset_name'] for record in new_records}):,} datasets"
    )
    if new_records:
        sample = new_records[0]
        log(
            "Example match: "
            f"{sample['target_screen_name']} -> {sample['matched_train_name']} "
            f"with {len(sample['predicted_genes'])} predicted genes"
        )

    if args.dry_run:
        log(f"Dry run completed without writing {results_path.name}")
        return

    if args.backup and results_path.exists():
        backup_path = results_path.with_suffix(results_path.suffix + ".bak")
        backup_path.write_text(results_path.read_text())
        log(f"Wrote backup to {backup_path}")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    log(f"Updated {results_path}")


if __name__ == "__main__":
    main()
