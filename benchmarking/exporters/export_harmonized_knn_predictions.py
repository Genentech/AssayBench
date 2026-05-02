from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from journal_figures_common import NOVEL_SPLIT_NAME, SCRIPT_DIR


DEFAULT_OUTPUT_DIR = Path("/cv/data/braid/gnesys/datasets/screensQA/results")
SCHEMA_VERSION = 1
KNN_TEST_ROOT = SCRIPT_DIR.parent / "output_latent_biology" / "knn_test"
TRANSFER_MATRIX_DIR = SCRIPT_DIR.parent / "output" / "transfer_matrix"
SCREEN_GENES_PKL = TRANSFER_MATRIX_DIR / "screen_genes.pkl"


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[export_harmonized_knn {timestamp} UTC] {message}", flush=True)


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
        "source_group": "knn_predictions",
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


def _load_screen_lookup() -> Dict[str, Dict[str, Any]]:
    with open(SCREEN_GENES_PKL, "rb") as handle:
        data = pickle.load(handle)

    lookup: Dict[str, Dict[str, Any]] = {}
    for label, metadata, screen in zip(
        data["screen_labels"],
        data["screen_metadata"],
        data["screens"],
    ):
        direction = "reverse" if metadata.get("reverse") else "forward"
        display_name = f"{label} ({direction})"
        lookup[display_name] = {
            "dataset_name": str(metadata["dataset_name"]),
            "genes": screen["genes"],
            "relevance_scores": screen["relevance_scores"],
        }
    return lookup


def _rank_screen_genes(screen: Dict[str, Any], top_k: int = 100) -> List[str]:
    ranked_pairs = sorted(
        zip(screen["genes"], screen["relevance_scores"]),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [str(gene).strip() for gene, _ in ranked_pairs[:top_k] if str(gene).strip()]


def _normalize_split_name(split_name: str) -> str:
    return NOVEL_SPLIT_NAME if split_name == "novel" else split_name


def _build_knn_records_for_layout(
    *,
    split_layout: str,
    screen_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    knn_file = KNN_TEST_ROOT / f"{split_layout}_fold0" / "knn_results.json"
    if not knn_file.exists():
        return {"Oracle kNN": [], "Embedding kNN": []}

    with open(knn_file) as handle:
        results = json.load(handle)

    payloads: Dict[str, List[Dict[str, Any]]] = {
        "Oracle kNN": [],
        "Embedding kNN": [],
    }
    method_map = {
        "oracle": "Oracle kNN",
        "embedding": "Embedding kNN",
    }

    for method_key, model_name in method_map.items():
        method_data = results.get(method_key, {})
        for split_name in ("val", "test", "novel"):
            split_data = method_data.get(split_name)
            if split_data is None:
                continue
            target_names = results.get(f"{split_name}_screens", [])
            matched_train_names = split_data.get("matched_train_names", [])
            matched_train_indices = split_data.get("matched_train_indices", [])
            n_examples = min(len(target_names), len(matched_train_names), len(matched_train_indices))

            for index in range(n_examples):
                target_name = target_names[index]
                train_name = matched_train_names[index]
                target_screen = screen_lookup.get(target_name)
                train_screen = screen_lookup.get(train_name)
                if target_screen is None or train_screen is None:
                    continue
                predicted_genes = _rank_screen_genes(train_screen, top_k=100)
                normalized_split = _normalize_split_name(split_name)
                payloads[model_name].append(
                    make_record(
                        dataset_name=target_screen["dataset_name"],
                        split=normalized_split,
                        split_layout=split_layout,
                        predicted_runs=[predicted_genes],
                        example_key=f"{split_layout}:{normalized_split}:{index}",
                        extra={
                            "target_screen_name": target_name,
                            "matched_train_name": train_name,
                            "matched_train_index": int(matched_train_indices[index]),
                            "matched_train_dataset_name": train_screen["dataset_name"],
                        },
                    )
                )
    return payloads


def export_knn_payloads(selected_models: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    screen_lookup = _load_screen_lookup()
    source_files = [str(SCREEN_GENES_PKL)]
    combined_records: Dict[str, List[Dict[str, Any]]] = {
        "Oracle kNN": [],
        "Embedding kNN": [],
    }

    for split_layout in ("year", "random"):
        knn_file = KNN_TEST_ROOT / f"{split_layout}_fold0" / "knn_results.json"
        if not knn_file.exists():
            continue
        source_files.append(str(knn_file))
        records_by_model = _build_knn_records_for_layout(
            split_layout=split_layout,
            screen_lookup=screen_lookup,
        )
        for model_name, records in records_by_model.items():
            combined_records[model_name].extend(records)

    payloads: List[Dict[str, Any]] = []
    for model_name, records in combined_records.items():
        if selected_models and model_name not in selected_models:
            continue
        payloads.append(
            build_payload(
                model_name=model_name,
                source_files=source_files,
                records=records,
            )
        )
    return payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export harmonized prediction files for Oracle kNN and Embedding kNN.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_models = set(args.model) if args.model else None
    payloads = export_knn_payloads(selected_models=selected_models)

    for payload in payloads:
        log(
            f"Prepared {payload['model_name']} with "
            f"{payload['n_records']} records across {payload['n_unique_datasets']} datasets"
        )
        if args.dry_run:
            continue
        output_path = write_payload(payload, args.output_dir, overwrite=args.overwrite)
        log(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
