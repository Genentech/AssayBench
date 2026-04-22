"""
Few-shot kNN prediction: use embedding-based nearest neighbors as in-context examples.

For each test screen this script:
1. Finds the k nearest training screens via cosine similarity on text-embedding-3-small embeddings
2. Constructs a prompt containing those k examples (screen question + top gene hits)
3. Queries the LLM (e.g. gemini-3-pro) to predict genes for the test screen
4. Saves predictions in the same JSON format as collect_llm_predictions.py

Usage:
  # Quick test with Flash
  python scripts/collect_fewshot_predictions.py \
    model_name=gemini-3-flash-fewshot-knn10 lm.model=gemini-3-flash

  # Full run with Pro (defaults)
  python scripts/collect_fewshot_predictions.py

  # Override k or split
  python scripts/collect_fewshot_predictions.py fewshot.k=5 dataset.split_type=author
"""

import rootutils
rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import dotenv
dotenv.load_dotenv('.env', override=True)

import os
import json
import time
import pickle
import hydra
from pathlib import Path
from omegaconf import DictConfig
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

import dspy
from dspy import Example

from screensqa.dataset.dataset import BioGRIDDSPY
from openai import AzureOpenAI

from scripts.run_ensemble_baseline import create_dspy_examples
from scripts.collect_llm_predictions import (
    create_ranking_signature,
    parse_genes_from_output,
    extract_genes_from_raw_response,
    check_for_truncation,
)


# ---------------------------------------------------------------------------
# Screen embedding utilities (mirrored from kNN_test.py)
# ---------------------------------------------------------------------------

SCREEN_EMBEDDING_CACHE_PATHS = [
    Path("output_latent_biology/text_embedding_cache.pkl"),
]


def _load_embedding_cache() -> Dict[str, np.ndarray]:
    for cache_path in SCREEN_EMBEDDING_CACHE_PATHS:
        if cache_path.exists():
            print(f"Loading screen embedding cache from: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            print(f"  Loaded {len(cache)} cached embeddings")
            return cache
    print("Warning: no embedding cache found – will compute on-the-fly")
    return {}


def _get_embedding_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["API_KEY_ES2"],
        api_version="2025-04-01-preview",
    )


def _embed_text(text: str, client: AzureOpenAI, cache: Dict) -> np.ndarray:
    if text in cache:
        return cache[text]
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    emb = np.array(response.data[0].embedding, dtype=np.float32)
    cache[text] = emb
    return emb


# ---------------------------------------------------------------------------
# kNN retrieval
# ---------------------------------------------------------------------------

def compute_knn_neighbors(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_reverse: List[bool],
    test_reverse: List[bool],
    k: int = 10,
    require_matching_reverse: bool = True,
) -> List[List[int]]:
    """Return top-k training indices for each test screen (cosine similarity)."""
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(test_embeddings, train_embeddings)  # (n_test, n_train)
    neighbors: List[List[int]] = []

    for test_idx in range(len(test_embeddings)):
        row = sims[test_idx].copy()
        if require_matching_reverse:
            for train_idx in range(len(train_embeddings)):
                if train_reverse[train_idx] != test_reverse[test_idx]:
                    row[train_idx] = -np.inf
        top_k = np.argsort(row)[::-1][:k].tolist()
        neighbors.append(top_k)

    return neighbors


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

INSTRUCTION_SUFFIX = (
    "\n\nYour goal is to provide a list of genes that meet the screen criteria, "
    "even if you do not have access to the actual experimental data. The genes must "
    "use HGNC symbols. Use your knowledge of biology, gene function, and relevant "
    "pathways to predict which genes are most likely to be hits. Do not refuse to "
    "answer or say you need more data—make your best predictions based on your "
    "understanding of the biological context."
)


def build_fewshot_question(
    test_question: str,
    neighbor_indices: List[int],
    train_examples: List[Example],
    top_genes: int = 100,
) -> str:
    """Build a single prompt with kNN examples baked in, then the test question."""
    parts = ["Here are examples of similar genetic screens and their top gene hits:\n"]

    for rank, train_idx in enumerate(neighbor_indices, 1):
        ex = train_examples[train_idx]
        genes = list(ex.genes)
        scores = list(ex.relevance_scores) if hasattr(ex, 'relevance_scores') else []
        if scores and len(scores) == len(genes):
            ordered = sorted(zip(genes, scores), key=lambda p: p[1], reverse=True)
            genes = [g for g, _ in ordered]
        gene_str = ", ".join(genes[:top_genes])
        parts.append(f"Example {rank}:\nScreen: {ex.question}\nTop genes: {gene_str}\n")

    parts.append(f"Now predict the top genes for this screen:\n{test_question}")
    parts.append(INSTRUCTION_SUFFIX)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

def process_single_example(
    predictor,
    question: str,
    example_idx: int,
    n_runs: int,
    verbose: bool = True,
) -> Tuple[List[List[str]], List[bool], List[str | None]]:
    """Run the LLM n_runs times on a single constructed prompt."""
    predictions: List[List[str]] = []
    truncation_flags: List[bool] = []
    reasoning_list: List[str | None] = []

    MIN_EXPECTED_GENES = 20

    for run_idx in range(n_runs):
        try:
            req_start = time.time()
            result = predictor(question=question)
            output_text = result.answer if hasattr(result, 'answer') else str(result)
            genes = parse_genes_from_output(output_text)

            raw_response_text = None
            try:
                lm = dspy.settings.lm
                if hasattr(lm, 'history') and lm.history:
                    last = lm.history[-1]
                    if isinstance(last, dict):
                        raw = last.get('response', {})
                        raw_response_text = raw.get('content', None) if isinstance(raw, dict) else (raw if isinstance(raw, str) else None)
            except Exception:
                pass

            if len(genes) < MIN_EXPECTED_GENES and raw_response_text:
                fallback = extract_genes_from_raw_response(raw_response_text)
                if len(fallback) > len(genes):
                    if verbose:
                        print(f"  [Ex {example_idx+1}] Run {run_idx+1}: quality gate "
                              f"({len(genes)}→{len(fallback)} genes via fallback)", flush=True)
                    genes = fallback

            was_truncated = check_for_truncation(result, dspy.settings.lm)
            predictions.append(genes)
            truncation_flags.append(was_truncated)

            reasoning_text = None
            if hasattr(result, 'reasoning') and result.reasoning:
                reasoning_text = result.reasoning
            elif raw_response_text:
                reasoning_text = raw_response_text
            reasoning_list.append(reasoning_text)

            if verbose:
                tag = " [TRUNCATED]" if was_truncated else ""
                print(f"  [Ex {example_idx+1}] Run {run_idx+1}: {len(genes)} genes{tag}", flush=True)

        except Exception as e:
            error_msg = str(e)
            fallback_genes: List[str] = []
            raw_text = None
            if 'cannot be serialized' in error_msg or 'Failed to parse' in error_msg:
                try:
                    lm = dspy.settings.lm
                    if hasattr(lm, 'history') and lm.history:
                        last = lm.history[-1]
                        if isinstance(last, dict):
                            raw_text = last.get('response', {})
                            if isinstance(raw_text, dict):
                                raw_text = raw_text.get('content', '')
                        else:
                            raw_text = str(last)
                        fallback_genes = extract_genes_from_raw_response(raw_text)
                except Exception:
                    pass

            if fallback_genes:
                print(f"  [Ex {example_idx+1}] Run {run_idx+1}: {len(fallback_genes)} genes (fallback)", flush=True)
                predictions.append(fallback_genes)
                truncation_flags.append(False)
                reasoning_list.append(raw_text)
            else:
                print(f"  [Ex {example_idx+1}] ERROR run {run_idx+1}: {error_msg[:500]}", flush=True)
                predictions.append([])
                truncation_flags.append('truncat' in error_msg.lower())
                reasoning_list.append(None)

    return predictions, truncation_flags, reasoning_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../configs", config_name="collect-fewshot-predictions")
def main(cfg: DictConfig):
    print("=" * 60)
    print("FEW-SHOT kNN PREDICTION")
    print("=" * 60)

    model_name = cfg.get('model_name', 'fewshot-unnamed')
    k = cfg.fewshot.k
    top_genes = cfg.fewshot.top_genes
    require_matching_reverse = cfg.fewshot.require_matching_reverse

    print(f"\nModel: {model_name}")
    print(f"Provider: {cfg.lm.provider}  LM: {cfg.lm.model}")
    print(f"Few-shot k={k}, top_genes={top_genes}, match_reverse={require_matching_reverse}")
    print(f"Runs per example: {cfg.collection.n_runs}")

    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ------------------------------------------------------------------
    # 1. Initialize LM
    # ------------------------------------------------------------------
    print(f"\nInitializing LM ({cfg.lm.provider})...")

    if cfg.lm.provider == 'gemini':
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY required in .env or environment")
        print(f"  Using Google Gemini: {cfg.lm.model}")
        lm = dspy.LM(
            model=f"gemini/{cfg.lm.model}",
            max_tokens=cfg.lm.get('max_tokens', 32000),
            temperature=cfg.lm.get('temperature', 1.0),
        )
    elif cfg.lm.provider == 'azure':
        api_key = os.environ.get('AZURE_API_KEY') or os.environ.get('AZURE_OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Azure OpenAI requires AZURE_API_KEY or AZURE_OPENAI_API_KEY")
        print(f"  Using Azure OpenAI: {cfg.lm.model}")
        lm = dspy.LM(
            model=f"azure/{cfg.lm.model}",
            max_tokens=cfg.lm.get('max_tokens', 32000),
            temperature=cfg.lm.get('temperature', 1.0),
        )
    elif cfg.lm.provider == 'anthropic':
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY required")
        print(f"  Using Anthropic: {cfg.lm.model}")
        lm = dspy.LM(
            model=f"anthropic/{cfg.lm.model}",
            max_tokens=cfg.lm.get('max_tokens', 32000),
            temperature=cfg.lm.get('temperature', 1.0),
        )
    else:
        print(f"  Using OpenAI-compatible: {cfg.lm.model}")
        lm = dspy.LM(
            model=cfg.lm.model,
            max_tokens=cfg.lm.get('max_tokens', 32000),
            temperature=cfg.lm.get('temperature', 1.0),
        )

    dspy.configure(lm=lm)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    RankingSignature = create_ranking_signature()
    predictor = dspy.Predict(RankingSignature)

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    print(f"\nLoading dataset...")
    dataset = BioGRIDDSPY(
        dataset_path=cfg.dataset.dataset_path,
        split_type=cfg.dataset.split_type,
        fold=cfg.dataset.fold,
    )
    train_raw, val_raw, test_raw = dataset.get_train_test_split()

    train_dspy = create_dspy_examples(train_raw)
    val_dspy = create_dspy_examples(val_raw)
    test_dspy = create_dspy_examples(test_raw)

    max_examples = cfg.dataset.get('max_examples', None)

    splits_to_collect: List[Tuple[str, List[Example]]] = []
    if cfg.collection.collect_train:
        subset = train_dspy[:max_examples] if max_examples else train_dspy
        splits_to_collect.append(('train', subset))
    if cfg.collection.collect_val:
        subset = val_dspy[:max_examples] if max_examples else val_dspy
        splits_to_collect.append(('val', subset))
    if cfg.collection.collect_test:
        subset = test_dspy[:max_examples] if max_examples else test_dspy
        splits_to_collect.append(('test', subset))

    print(f"  Train: {len(train_dspy)},  Val: {len(val_dspy)},  Test: {len(test_dspy)}")
    if max_examples:
        print(f"  max_examples={max_examples} (limiting each collected split)")
    print(f"  Collecting: {[(s, len(e)) for s, e in splits_to_collect]}")

    # ------------------------------------------------------------------
    # 3. Compute embeddings
    # ------------------------------------------------------------------
    print("\nComputing screen embeddings...")
    emb_cache = _load_embedding_cache()
    emb_client = _get_embedding_client()

    train_questions = [ex.question for ex in train_dspy]
    train_reverse = [getattr(ex, 'reverse', False) for ex in train_dspy]

    train_embeddings = np.stack([
        _embed_text(q, emb_client, emb_cache) for q in tqdm(train_questions, desc="Train embeddings")
    ])
    print(f"  Train embeddings: {train_embeddings.shape}")

    # ------------------------------------------------------------------
    # 4. Collect predictions per split
    # ------------------------------------------------------------------
    output_base = Path(cfg.output.save_dir) / model_name
    output_base.mkdir(parents=True, exist_ok=True)

    for split_name, split_examples in splits_to_collect:
        print(f"\n{'=' * 60}")
        print(f"COLLECTING {split_name.upper()} PREDICTIONS  ({len(split_examples)} examples)")
        print(f"{'=' * 60}")

        # Embed this split
        split_questions = [ex.question for ex in split_examples]
        split_reverse = [getattr(ex, 'reverse', False) for ex in split_examples]
        split_embeddings = np.stack([
            _embed_text(q, emb_client, emb_cache) for q in tqdm(split_questions, desc=f"{split_name} embeddings")
        ])

        # kNN retrieval
        print(f"  Computing {k}-NN neighbors...")
        all_neighbors = compute_knn_neighbors(
            train_embeddings, split_embeddings,
            train_reverse, split_reverse,
            k=k,
            require_matching_reverse=require_matching_reverse,
        )

        # Load existing predictions for resume support
        output_file = output_base / f"{split_name}_predictions.json"
        existing_by_idx: Dict[int, Dict] = {}
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    prev = json.load(f)
                for idx, entry in enumerate(prev.get('predictions', [])):
                    if len(entry.get('predictions', [])) >= cfg.collection.n_runs:
                        existing_by_idx[idx] = entry
                print(f"  Resuming: {len(existing_by_idx)} examples already complete")
            except Exception:
                pass

        all_predictions: List[Dict] = [None] * len(split_examples)
        neighbor_metadata: List[List[int]] = [None] * len(split_examples)

        for idx, ex in enumerate(split_examples):
            existing = existing_by_idx.get(idx)
            if existing is not None:
                has_failed = any(len(p) == 0 for p in existing.get('predictions', []))
                if not has_failed or not cfg.collection.retry_failed:
                    all_predictions[idx] = existing
                    neighbor_metadata[idx] = all_neighbors[idx]
                    continue

            print(f"\n{split_name} - Example {idx+1}/{len(split_examples)}")
            question = build_fewshot_question(
                ex.question, all_neighbors[idx], train_dspy, top_genes=top_genes,
            )
            preds, truncs, reasoning = process_single_example(
                predictor, question, idx, cfg.collection.n_runs,
            )

            all_predictions[idx] = {
                'question': ex.question,
                'predictions': preds,
                'truncated': truncs,
                'any_truncated': any(truncs),
                'reasoning': reasoning,
                'knn_neighbor_indices': all_neighbors[idx],
            }
            neighbor_metadata[idx] = all_neighbors[idx]

            # Periodic save
            save_freq = cfg.collection.get('save_frequency', 5)
            if (idx + 1) % save_freq == 0 or idx == len(split_examples) - 1:
                _save_split(output_file, model_name, split_name,
                            cfg.collection.n_runs, all_predictions,
                            neighbor_metadata, status='in_progress')
                print(f"  Saved progress ({idx+1}/{len(split_examples)})", flush=True)

        _save_split(output_file, model_name, split_name,
                    cfg.collection.n_runs, all_predictions,
                    neighbor_metadata, status='complete')
        print(f"\nSaved {split_name} predictions to {output_file}")

    print(f"\n{'=' * 60}")
    print("COLLECTION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Predictions saved to: {output_base}/")


def _save_split(
    output_file: Path,
    model_name: str,
    split_name: str,
    n_runs: int,
    predictions: List[Dict | None],
    neighbor_metadata: List[List[int] | None],
    status: str = 'in_progress',
):
    completed = [p for p in predictions if p is not None]
    num_truncated = sum(1 for p in completed if p.get('any_truncated', False))

    data = {
        'model_name': model_name,
        'n_runs': n_runs,
        'split': split_name,
        'num_examples': len(completed),
        'num_examples_with_truncation': num_truncated,
        'predictions': completed,
        'knn_neighbor_metadata': [n for n in neighbor_metadata if n is not None],
        'status': status,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
