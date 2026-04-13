from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import dotenv

dotenv.load_dotenv(".env", override=True)

import dspy

from journal_figures_common import DATASET_PATH, NOVEL_DATASET_PATHS, NOVEL_SPLIT_NAME, RESULTS_DIR
from screensqa.dataset.dataset import BioGRIDDSPY

try:
    from run_ensemble_baseline import RankingModule, create_dspy_examples, create_ranking_signature, parse_genes_from_output
except ImportError:
    from scripts.run_ensemble_baseline import (
        RankingModule,
        create_dspy_examples,
        create_ranking_signature,
        parse_genes_from_output,
    )


DEFAULT_GEPA_SPLIT_PATH = Path(
    "./output/gepa/gemini-3-flash/llm_predictions/gemini-3-flash/test_predictions.json"
)


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[update_gepa_novel_public {timestamp} UTC] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GEPA Gemini predictions for novel_public_2026_dataset and "
            "merge them into the harmonized GEPA JSON."
        )
    )
    parser.add_argument("--model-name", type=str, default="gepa/gemini-3-flash")
    parser.add_argument("--lm-provider", type=str, default="gemini")
    parser.add_argument("--lm-model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--top-genes", type=int, default=100)
    parser.add_argument(
        "--train-dataset-path",
        type=Path,
        default=Path(DATASET_PATH),
    )
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
        default=RESULTS_DIR / "gepa__gemini-3-flash.json",
    )
    parser.add_argument(
        "--gepa-split-path",
        type=Path,
        default=DEFAULT_GEPA_SPLIT_PATH,
        help="Existing GEPA split predictions JSON used to source the optimized prompt and LM config.",
    )
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


def load_novel_pool(*, novel_dataset_path: Path, fold: int) -> List[Any]:
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


def load_gepa_config(path: Path) -> tuple[str, Dict[str, Any]]:
    with open(path) as handle:
        payload = json.load(handle)
    optimized_prompt = str(payload.get("optimized_prompt", "")).strip()
    if not optimized_prompt:
        raise ValueError(f"No optimized_prompt found in {path}")
    lm_config = dict(payload.get("lm_config", {}))
    return optimized_prompt, lm_config


def configure_lm(
    *,
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    optimized_prompt: str,
) -> Any:
    log(f"Initializing GEPA LM provider={provider!r} model={model!r}")
    if provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is required to call Gemini.")
        lm = dspy.LM(
            model=f"gemini/{model}",
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    dspy.configure(lm=lm)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    signature_class = create_ranking_signature(optimized_prompt)
    return RankingModule(signature_class=signature_class)


def run_predictions(
    *,
    model: Any,
    examples: Sequence[Any],
    n_runs: int,
    top_genes: int,
) -> List[List[List[str]]]:
    all_runs: List[List[List[str]]] = []
    for index, example in enumerate(examples):
        log(f"Predicting novel public screen {index + 1}/{len(examples)}")
        runs: List[List[str]] = []
        for run_index in range(n_runs):
            started = time.time()
            result = model(question=example.question)
            output_text = result.answer if hasattr(result, "answer") else str(result)
            genes = clean_gene_list(parse_genes_from_output(output_text))[:top_genes]
            runs.append(genes)
            log(
                f"  run {run_index + 1}/{n_runs}: {len(genes)} genes "
                f"in {time.time() - started:.1f}s"
            )
        all_runs.append(runs)
    return all_runs


def build_records(
    *,
    novel_examples: Sequence[Any],
    prediction_runs: Sequence[Sequence[Sequence[str]]],
    split_name: str,
    split_layout: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, (novel_ex, runs) in enumerate(zip(novel_examples, prediction_runs)):
        cleaned_runs = [clean_gene_list(run) for run in runs]
        primary_prediction = cleaned_runs[0] if cleaned_runs else []
        records.append(
            {
                "dataset_name": str(novel_ex.dataset_name),
                "split": split_name,
                "split_layout": split_layout,
                "predicted_genes": primary_prediction,
                "prediction_runs": cleaned_runs,
                "n_runs": len(cleaned_runs),
                "example_key": f"{split_name}:{index}",
            }
        )
    return records


def main() -> None:
    args = parse_args()

    optimized_prompt, source_lm_config = load_gepa_config(args.gepa_split_path)
    log(
        f"Loaded optimized prompt from {args.gepa_split_path} "
        f"({len(optimized_prompt):,} chars)"
    )

    provider = str(source_lm_config.get("provider", args.lm_provider))
    model_name = str(source_lm_config.get("model", args.lm_model))
    max_tokens = int(source_lm_config.get("max_tokens", args.max_tokens))
    temperature = float(source_lm_config.get("temperature", args.temperature))

    novel_examples = load_novel_pool(
        novel_dataset_path=args.novel_dataset_path,
        fold=args.fold,
    )
    model = configure_lm(
        provider=provider,
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        optimized_prompt=optimized_prompt,
    )
    prediction_runs = run_predictions(
        model=model,
        examples=novel_examples,
        n_runs=args.n_runs,
        top_genes=args.top_genes,
    )

    new_records = build_records(
        novel_examples=novel_examples,
        prediction_runs=prediction_runs,
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
            "model_name": args.model_name,
            "source_group": "gepa_predictions",
            "source_files": [],
        }
        existing_records = []

    combined_records = existing_records + new_records
    payload["schema_version"] = int(payload.get("schema_version", 1))
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["model_name"] = args.model_name
    payload["source_group"] = "gepa_predictions"
    payload["source_files"] = unique_strings(
        list(payload.get("source_files", []))
        + [
            str(args.gepa_split_path),
            str(args.train_dataset_path),
            str(args.novel_dataset_path),
            str(Path(__file__).resolve()),
        ]
    )
    payload["n_records"] = len(combined_records)
    payload["n_unique_datasets"] = len(group_records_by_dataset(combined_records))
    payload["records_by_dataset"] = group_records_by_dataset(combined_records)

    log(
        f"Prepared {len(new_records):,} novel public GEPA records "
        f"for {args.model_name}"
    )
    sample = new_records[0]
    log(
        f"Example: {sample['dataset_name']} with "
        f"{sample['n_runs']} run(s) and {len(sample['predicted_genes'])} predicted genes"
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
