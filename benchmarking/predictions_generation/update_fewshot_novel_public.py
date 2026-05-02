from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import dotenv

dotenv.load_dotenv(".env", override=True)

import dspy
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity

from journal_figures_common import DATASET_PATH, NOVEL_DATASET_PATHS, NOVEL_SPLIT_NAME, RESULTS_DIR
from screensqa.dataset.dataset import BioGRIDDSPY

try:
    from run_ensemble_baseline import create_dspy_examples, create_ranking_signature, parse_genes_from_output
except ImportError:
    from scripts.run_ensemble_baseline import (
        create_dspy_examples,
        create_ranking_signature,
        parse_genes_from_output,
    )


SCREEN_EMBEDDING_CACHE_PATHS = [
    Path("output_latent_biology/text_embedding_cache.pkl"),
]

INSTRUCTION_SUFFIX = (
    "\n\nYour goal is to provide a list of genes that meet the screen criteria, "
    "even if you do not have access to the actual experimental data. The genes must "
    "use HGNC symbols. Use your knowledge of biology, gene function, and relevant "
    "pathways to predict which genes are most likely to be hits. Do not refuse to "
    "answer or say you need more data—make your best predictions based on your "
    "understanding of the biological context."
)


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[update_fewshot_novel_public {timestamp} UTC] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate few-shot Gemini predictions for novel_public_2026_dataset and "
            "merge them into the harmonized few-shot JSON."
        )
    )
    parser.add_argument("--model-name", type=str, default="fewshot/gemini-3-pro-fewshot-knn10")
    parser.add_argument("--lm-provider", type=str, default="gemini")
    parser.add_argument("--lm-model", type=str, default="gemini-3-pro-preview")
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
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
        default=RESULTS_DIR / "fewshot__gemini-3-pro-fewshot-knn10.json",
    )
    parser.add_argument(
        "--require-matching-reverse",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    top_genes: int,
) -> tuple[List[Any], np.ndarray, List[bool], List[List[str]]]:
    log(
        f"Loading train pool from {train_dataset_path} using split_type={train_split_type!r}, fold={fold}"
    )
    train_loader = BioGRIDDSPY(
        dataset_path=str(train_dataset_path),
        split_type=train_split_type,
        fold=fold,
    )
    train_examples, _, _ = train_loader.get_train_test_split()
    train_dspy = create_dspy_examples(train_examples)
    train_questions = [ex.question for ex in train_dspy]
    train_reverse = [bool(getattr(ex, "reverse", False)) for ex in train_dspy]
    ranked_train_genes = [rank_screen_genes(ex, top_genes) for ex in train_dspy]
    log(f"Loaded {len(train_dspy):,} training screens")
    return train_dspy, np.array(train_questions, dtype=object), train_reverse, ranked_train_genes


def load_novel_pool(
    *,
    novel_dataset_path: Path,
    fold: int,
) -> tuple[List[Any], np.ndarray, List[bool]]:
    log(f"Loading novel public screens from {novel_dataset_path}")
    novel_loader = BioGRIDDSPY(
        dataset_path=str(novel_dataset_path),
        split_type=None,
        fold=fold,
    )
    novel_examples = novel_loader.get_dspy_examples()
    novel_dspy = create_dspy_examples(novel_examples)
    novel_questions = [ex.question for ex in novel_dspy]
    novel_reverse = [bool(getattr(ex, "reverse", False)) for ex in novel_dspy]
    log(f"Loaded {len(novel_dspy):,} novel public screens")
    return novel_dspy, np.array(novel_questions, dtype=object), novel_reverse


def load_embedding_cache() -> Dict[str, np.ndarray]:
    for cache_path in SCREEN_EMBEDDING_CACHE_PATHS:
        if cache_path.exists():
            log(f"Loading screen embedding cache from {cache_path}")
            with open(cache_path, "rb") as handle:
                cache = pickle.load(handle)
            log(f"Loaded {len(cache):,} cached text embeddings")
            return cache
    log("No screen embedding cache found; embeddings will be computed on demand")
    return {}


def get_embedding_client() -> AzureOpenAI | None:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("API_KEY_ES2")
    if not endpoint or not api_key:
        return None
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2025-04-01-preview",
    )


def embed_text(text: str, client: AzureOpenAI | None, cache: Dict[str, np.ndarray]) -> np.ndarray:
    if text in cache:
        return cache[text]
    if client is None:
        raise RuntimeError(
            "Missing Azure embedding credentials and required text was not found in the local embedding cache."
        )
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    embedding = np.array(response.data[0].embedding, dtype=np.float32)
    cache[text] = embedding
    return embedding


def compute_knn_neighbors(
    *,
    train_embeddings: np.ndarray,
    novel_embeddings: np.ndarray,
    train_reverse: Sequence[bool],
    novel_reverse: Sequence[bool],
    k: int,
    require_matching_reverse: bool,
) -> List[List[int]]:
    similarities = cosine_similarity(novel_embeddings, train_embeddings)
    neighbors: List[List[int]] = []
    for novel_index in range(len(novel_embeddings)):
        row = similarities[novel_index].copy()
        if require_matching_reverse:
            for train_index in range(len(train_embeddings)):
                if train_reverse[train_index] != novel_reverse[novel_index]:
                    row[train_index] = -np.inf
        neighbors.append(np.argsort(row)[::-1][:k].tolist())
    return neighbors


def build_fewshot_question(
    *,
    target_question: str,
    neighbor_indices: Sequence[int],
    train_examples: Sequence[Any],
    ranked_train_genes: Sequence[Sequence[str]],
    top_genes: int,
) -> str:
    parts = ["Here are examples of similar genetic screens and their top gene hits:\n"]
    for rank, train_index in enumerate(neighbor_indices, 1):
        example = train_examples[train_index]
        gene_str = ", ".join(ranked_train_genes[train_index][:top_genes])
        parts.append(f"Example {rank}:\nScreen: {example.question}\nTop genes: {gene_str}\n")
    parts.append(f"Now predict the top genes for this screen:\n{target_question}")
    parts.append(INSTRUCTION_SUFFIX)
    return "\n".join(parts)


def configure_lm(
    *,
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Any:
    log(f"Initializing LM provider={provider!r} model={model!r}")
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
    signature = create_ranking_signature()
    return dspy.Predict(signature)


def run_predictions(
    *,
    predictor: Any,
    prompts: Sequence[str],
    n_runs: int,
) -> List[List[List[str]]]:
    all_runs: List[List[List[str]]] = []
    for index, prompt in enumerate(prompts):
        log(f"Predicting novel public screen {index + 1}/{len(prompts)}")
        runs: List[List[str]] = []
        for run_index in range(n_runs):
            started = time.time()
            result = predictor(question=prompt)
            output_text = result.answer if hasattr(result, "answer") else str(result)
            genes = clean_gene_list(parse_genes_from_output(output_text))
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
    train_examples: Sequence[Any],
    ranked_train_genes: Sequence[Sequence[str]],
    neighbor_indices: Sequence[Sequence[int]],
    prediction_runs: Sequence[Sequence[Sequence[str]]],
    split_name: str,
    split_layout: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, (novel_ex, neighbors, runs) in enumerate(zip(novel_examples, neighbor_indices, prediction_runs)):
        primary_match = neighbors[0]
        matched_train = train_examples[primary_match]
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
                "target_screen_name": format_screen_name(
                    str(novel_ex.dataset_name),
                    bool(getattr(novel_ex, "reverse", False)),
                ),
                "matched_train_name": format_screen_name(
                    str(matched_train.dataset_name),
                    bool(getattr(matched_train, "reverse", False)),
                ),
                "matched_train_index": int(primary_match),
                "matched_train_dataset_name": str(matched_train.dataset_name),
                "knn_neighbor_indices": [int(value) for value in neighbors],
                "knn_neighbor_names": [
                    format_screen_name(
                        str(train_examples[neighbor_index].dataset_name),
                        bool(getattr(train_examples[neighbor_index], "reverse", False)),
                    )
                    for neighbor_index in neighbors
                ],
            }
        )
    return records


def main() -> None:
    args = parse_args()

    train_examples, train_questions, train_reverse, ranked_train_genes = load_train_pool(
        train_dataset_path=args.train_dataset_path,
        train_split_type=args.train_split_type,
        fold=args.fold,
        top_genes=args.top_genes,
    )
    novel_examples, novel_questions, novel_reverse = load_novel_pool(
        novel_dataset_path=args.novel_dataset_path,
        fold=args.fold,
    )

    emb_cache = load_embedding_cache()
    emb_client = get_embedding_client()
    train_embeddings = np.stack(
        [embed_text(question, emb_client, emb_cache) for question in tqdm(train_questions.tolist(), desc="Train embeddings")]
    )
    novel_embeddings = np.stack(
        [embed_text(question, emb_client, emb_cache) for question in tqdm(novel_questions.tolist(), desc="Novel embeddings")]
    )
    neighbor_indices = compute_knn_neighbors(
        train_embeddings=train_embeddings,
        novel_embeddings=novel_embeddings,
        train_reverse=train_reverse,
        novel_reverse=novel_reverse,
        k=args.k,
        require_matching_reverse=args.require_matching_reverse,
    )
    prompts = [
        build_fewshot_question(
            target_question=novel_ex.question,
            neighbor_indices=neighbors,
            train_examples=train_examples,
            ranked_train_genes=ranked_train_genes,
            top_genes=args.top_genes,
        )
        for novel_ex, neighbors in zip(novel_examples, neighbor_indices)
    ]

    predictor = configure_lm(
        provider=args.lm_provider,
        model=args.lm_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    prediction_runs = run_predictions(
        predictor=predictor,
        prompts=prompts,
        n_runs=args.n_runs,
    )

    new_records = build_records(
        novel_examples=novel_examples,
        train_examples=train_examples,
        ranked_train_genes=ranked_train_genes,
        neighbor_indices=neighbor_indices,
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
            "source_group": "fewshot_predictions",
            "source_files": [],
        }
        existing_records = []

    combined_records = existing_records + new_records
    payload["schema_version"] = int(payload.get("schema_version", 1))
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["model_name"] = args.model_name
    payload["source_group"] = "fewshot_predictions"
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
        f"Prepared {len(new_records):,} novel public few-shot records "
        f"for {args.model_name}"
    )
    sample = new_records[0]
    log(
        "Example: "
        f"{sample['target_screen_name']} <- {sample['matched_train_name']} "
        f"with {sample['n_runs']} run(s)"
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
