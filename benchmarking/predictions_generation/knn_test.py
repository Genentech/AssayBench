"""
kNN-based transfer analysis with multiple methods.

This script evaluates different kNN approaches for selecting the best training screen
for each test screen:

1. Oracle kNN: Uses actual transfer matrix (upper bound - perfect selection)
2. Embedding kNN: Uses screen embeddings (full question text)
3. Model kNN: Uses trained ScreenTransferPredictor model to predict transfer scores

Usage:
    # Basic usage (oracle + embedding kNN only)
    uv run python scripts/latent_biology/kNN_test.py \
        --split-type easy_split --fold 0
    
    # With trained model
    uv run python scripts/latent_biology/kNN_test.py \
        --split-type easy_split --fold 0 \
        --model-checkpoint output_latent_biology/screen/best_model.pt
"""

import rootutils
rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Sequence, Tuple, Optional
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from screensqa.dataset.dataset import BioGRIDDSPY
from screensqa.benchmark.ranking_metrics import RankingMetrics
from openai import AzureOpenAI

# Add parent scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_ensemble_baseline import create_dspy_examples

from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# Screen Embedding Utilities
# ============================================================================

# Screen embedding cache paths (in priority order)
SCREEN_EMBEDDING_CACHE_PATHS = [
    Path("output_latent_biology/text_embedding_cache.pkl"),
]


def get_text_embedding_client():
    """Initialize Azure OpenAI client for text embeddings."""
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["API_KEY_ES2"],
        api_version="2025-04-01-preview"
    )


def load_screen_embedding_cache() -> Dict[str, np.ndarray]:
    """Load pre-computed screen text embedding cache from the first available path."""
    for cache_path in SCREEN_EMBEDDING_CACHE_PATHS:
        if cache_path.exists():
            print(f"Loading pre-computed screen embeddings from: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            print(f"  Loaded {len(cache)} pre-computed embeddings")
            return cache
    
    print(f"Warning: No screen embedding cache found. Will compute embeddings on-the-fly.")
    print(f"  Searched paths: {[str(p) for p in SCREEN_EMBEDDING_CACHE_PATHS]}")
    return {}


def get_text_embedding(text: str, client: AzureOpenAI, cache: Dict = None) -> np.ndarray:
    """Get text embedding from Azure OpenAI, with caching."""
    if cache is not None and text in cache:
        return cache[text]
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = np.array(response.data[0].embedding, dtype=np.float32)
    
    if cache is not None:
        cache[text] = embedding
    
    return embedding


def compute_screen_embedding(
    question: str,
    text_client: AzureOpenAI,
    text_cache: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute screen embedding from the full question text.
    
    The question text includes all screen details (cell line, library type,
    experimental setup, phenotype, hit definition, ranking criteria, etc.)
    so a single embedding captures the full context.
    """
    return get_text_embedding(question, text_client, text_cache)


# ============================================================================
# Transfer Predictor Model (copied from train_transfer_predictor.py)
# ============================================================================

class ScreenTransferPredictor(nn.Module):
    """
    Model to predict screen-to-screen transfer score.
    
    Input: [source_screen_emb, target_screen_emb]
    Output: transfer score (scalar)
    """
    
    def __init__(
        self,
        screen_dim: int = 1536,  # text-embedding-3-small from full question text
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Project each screen embedding
        self.source_proj = nn.Linear(screen_dim, 256)
        self.target_proj = nn.Linear(screen_dim, 256)
        
        # Combined embedding + interaction
        combined_dim = 256 + 256 + 256  # source + target + interaction
        
        # MLP
        layers = []
        in_dim = combined_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, source_emb, target_emb):
        """
        Args:
            source_emb: (B, screen_dim)
            target_emb: (B, screen_dim)
        
        Returns:
            scores: (B,) transfer scores
        """
        source_proj = F.relu(self.source_proj(source_emb))
        target_proj = F.relu(self.target_proj(target_emb))
        
        # Element-wise interaction
        interaction = source_proj * target_proj
        
        # Concatenate
        combined = torch.cat([source_proj, target_proj, interaction], dim=1)
        
        # Predict
        scores = self.mlp(combined).squeeze(-1)
        
        return scores


# ============================================================================
# kNN Methods
# ============================================================================

def compute_oracle_knn(
    transfer_matrix: np.ndarray,
    train_indices: List[int],
    test_indices: List[int],
    train_reverse: List[bool],
    test_reverse: List[bool],
    require_matching_reverse: bool = True
) -> Tuple[List[int], List[float]]:
    """
    Oracle kNN: Use actual transfer matrix to find best training screen.
    
    Args:
        transfer_matrix: Full N×N transfer matrix
        train_indices: Indices of training screens in the matrix
        test_indices: Indices of test screens in the matrix
        train_reverse: Reverse flags for training screens
        test_reverse: Reverse flags for test screens
        require_matching_reverse: Only consider same reverse type
    
    Returns:
        matched_train_indices: Best training screen index for each test screen
        scores: Transfer score achieved for each test screen
    """
    matched_train_indices = []
    scores = []
    
    for test_idx, test_matrix_idx in enumerate(test_indices):
        test_rev = test_reverse[test_idx]
        
        best_train_idx = None
        best_score = -np.inf
        
        for train_idx, train_matrix_idx in enumerate(train_indices):
            if require_matching_reverse and train_reverse[train_idx] != test_rev:
                continue
            
            score = transfer_matrix[train_matrix_idx, test_matrix_idx]
            if score > best_score:
                best_score = score
                best_train_idx = train_idx
        
        matched_train_indices.append(best_train_idx if best_train_idx is not None else 0)
        scores.append(best_score if best_score > -np.inf else 0.0)
    
    return matched_train_indices, scores


def compute_embedding_knn(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_reverse: List[bool],
    test_reverse: List[bool],
    require_matching_reverse: bool = True
) -> List[int]:
    """
    Embedding kNN: Use cosine similarity on screen embeddings.
    
    Args:
        train_embeddings: (n_train, embed_dim) array
        test_embeddings: (n_test, embed_dim) array
        train_reverse: Reverse flags for training screens
        test_reverse: Reverse flags for test screens
        require_matching_reverse: Only consider same reverse type
    
    Returns:
        matched_train_indices: Best training screen index for each test screen
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Compute all pairwise cosine similarities: (n_test, n_train)
    similarities = cosine_similarity(test_embeddings, train_embeddings)
    
    matched_train_indices = []
    for test_idx in range(len(test_embeddings)):
        test_rev = test_reverse[test_idx]
        sims = similarities[test_idx].copy()
        
        if require_matching_reverse:
            for train_idx in range(len(train_embeddings)):
                if train_reverse[train_idx] != test_rev:
                    sims[train_idx] = -np.inf
        
        best_train_idx = np.argmax(sims)
        matched_train_indices.append(best_train_idx)
    
    return matched_train_indices


def compute_model_knn(
    model: ScreenTransferPredictor,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_reverse: List[bool],
    test_reverse: List[bool],
    device: torch.device,
    require_matching_reverse: bool = True
) -> Tuple[List[int], np.ndarray]:
    """
    Model kNN: Use trained transfer predictor to find best training screen.
    
    Args:
        model: Trained ScreenTransferPredictor
        train_embeddings: (n_train, embed_dim) array
        test_embeddings: (n_test, embed_dim) array
        train_reverse: Reverse flags for training screens
        test_reverse: Reverse flags for test screens
        device: Torch device
        require_matching_reverse: Only consider same reverse type
    
    Returns:
        matched_train_indices: Best training screen index for each test screen
        predicted_scores: (n_test, n_train) matrix of predicted transfer scores
    """
    model.eval()
    n_train = len(train_embeddings)
    n_test = len(test_embeddings)
    
    # Predict all (train, test) pairs
    predicted_scores = np.zeros((n_test, n_train))
    
    with torch.no_grad():
        for test_idx in range(n_test):
            test_emb = torch.from_numpy(test_embeddings[test_idx:test_idx+1]).to(device)
            test_emb_expanded = test_emb.expand(n_train, -1)
            
            train_embs = torch.from_numpy(train_embeddings).to(device)
            
            # Predict transfer scores from all training screens to this test screen
            scores = model(train_embs, test_emb_expanded)
            predicted_scores[test_idx] = scores.cpu().numpy()
    
    # Find best match for each test screen
    matched_train_indices = []
    for test_idx in range(n_test):
        test_rev = test_reverse[test_idx]
        scores = predicted_scores[test_idx].copy()
        
        if require_matching_reverse:
            for train_idx in range(n_train):
                if train_reverse[train_idx] != test_rev:
                    scores[train_idx] = -np.inf
        
        best_train_idx = np.argmax(scores)
        matched_train_indices.append(best_train_idx)
    
    return matched_train_indices, predicted_scores


EVAL_METRICS = ("adjusted_ndcg@100", "precision@100", "inverse_precision@100")

DEFAULT_NOVEL_PATHS = [
    "./data/novel_public_2026_dataset/",
]


def load_novel_split(
    dataset_paths: List[str],
) -> Tuple[List, List[Dict]]:
    """
    Load novel screens from additional HuggingFace datasets.

    Returns (dspy_examples, ground_truth) where ground_truth mirrors the
    structure used by the transfer-matrix screens (keys: genes, relevance_scores).
    """
    from datasets import load_from_disk
    from screensqa.utils.prompt_loaders import load_objective_prompt

    prompt_template = load_objective_prompt("biogrid_ranking_prompt")
    all_examples: List[Dict[str, Any]] = []
    all_gt: List[Dict] = []

    for ds_path in dataset_paths:
        ds_path_obj = Path(ds_path)
        if not ds_path_obj.exists():
            print(f"  WARNING: Novel dataset path not found, skipping: {ds_path}")
            continue
        ds = load_from_disk(str(ds_path_obj))
        print(f"  Loaded {len(ds)} novel examples from {ds_path}")

        for item in ds:
            phenotype = item.get("phenotype", "")
            if phenotype and phenotype[-1] == ".":
                item = dict(item)
                item["phenotype"] = phenotype[:-1]
            prompt = prompt_template.format(**item)

            top10_args = np.argsort(item["relevance_scores"])[::-1]
            top10_genes = np.array(item["relevance_genes"])[top10_args][:10].tolist()

            example = {
                "question": prompt,
                "relevance_genes": item["relevance_genes"],
                "relevance_scores": item["relevance_scores"],
                "hit": item.get("hit", []),
                "dataset_name": item.get("dataset_name", "unknown"),
                "phenotype": item.get("phenotype", "Not specified"),
                "num_genes": len(item["relevance_genes"]),
                "reverse": item.get("reverse", False),
                "answer": ", ".join(top10_genes),
            }
            all_examples.append(example)
            all_gt.append({
                "genes": item["relevance_genes"],
                "relevance_scores": item["relevance_scores"],
            })

    dspy_examples = create_dspy_examples(all_examples)
    print(f"  Novel split total: {len(dspy_examples)} screens")
    return dspy_examples, all_gt


def evaluate_knn_transfer(
    train_ground_truth: List[Dict],
    target_ground_truth: List[Dict],
    matched_train_indices: List[int],
    metrics_evaluator: RankingMetrics,
    top_k: int = 100,
    metrics: Tuple[str, ...] = EVAL_METRICS,
) -> Dict[str, List[float]]:
    """
    Evaluate transfer using kNN-selected training screens.
    
    For each target screen, uses the ground truth from its matched training screen.
    
    Returns:
        Dict mapping metric name to list of per-screen scores.
    """
    per_metric: Dict[str, List[float]] = {m: [] for m in metrics}
    
    for target_idx, train_idx in enumerate(matched_train_indices):
        source_genes = train_ground_truth[train_idx]['genes']
        source_scores = train_ground_truth[train_idx]['relevance_scores']
        
        gene_score_pairs = list(zip(source_genes, source_scores))
        gene_score_pairs.sort(key=lambda x: x[1], reverse=True)
        top_genes = [gene for gene, _ in gene_score_pairs[:top_k]]
        
        target_genes = target_ground_truth[target_idx]['genes']
        target_scores = target_ground_truth[target_idx]['relevance_scores']
        
        results = metrics_evaluator.evaluate(
            predicted_genes=top_genes,
            ground_truth_genes=target_genes,
            relevance_scores=target_scores
        )
        
        for m in metrics:
            per_metric[m].append(results.get(m, 0.0))
    
    return per_metric


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="kNN-based transfer analysis")
    
    # Data
    parser.add_argument("--transfer-matrix-dir", type=str, 
                        default="output/transfer_matrix",
                        help="Directory containing pre-computed transfer matrix")
    parser.add_argument("--split-type", type=str, default="easy_split",
                        help="Type of split")
    parser.add_argument("--fold", type=int, default=0, help="Fold to use")
    
    # Model (optional)
    parser.add_argument("--model-checkpoint", type=str, default=None,
                        help="Path to trained ScreenTransferPredictor checkpoint")
    
    # Options
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of top genes to use as predictions")
    parser.add_argument("--reverse-weight", type=float, default=5.0,
                        help="Weight for reverse indicator in embeddings")
    parser.add_argument("--require-matching-reverse", action="store_true", default=True,
                        help="Only match screens with same reverse type")
    parser.add_argument("--no-require-matching-reverse", dest="require_matching_reverse", 
                        action="store_false")
    
    # Novel split
    parser.add_argument("--novel-dataset-paths", type=str, nargs="*", default=None,
                        help="Paths to novel HuggingFace datasets (default: built-in novel_public_2026)")
    parser.add_argument("--no-novel", action="store_true",
                        help="Skip novel split evaluation")
    
    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: output_latent_biology/knn_test)")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"output_latent_biology/knn_test/{args.split_type}_fold{args.fold}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("kNN-BASED TRANSFER ANALYSIS")
    print("=" * 70)
    print(f"Split: {args.split_type}, Fold: {args.fold}")
    print(f"Output: {output_dir}")
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    # Load transfer matrix
    transfer_matrix_dir = Path(args.transfer_matrix_dir)
    print(f"\nLoading transfer matrix from {transfer_matrix_dir}...")
    
    transfer_matrix = np.load(transfer_matrix_dir / "transfer_matrix.npy")
    
    with open(transfer_matrix_dir / "screen_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    with open(transfer_matrix_dir / "screen_genes.pkl", 'rb') as f:
        screen_genes_data = pickle.load(f)
    
    all_screen_labels = screen_genes_data['screen_labels']
    all_screen_metadata = screen_genes_data['screen_metadata']
    all_screens = screen_genes_data['screens']
    
    n_total = len(all_screen_labels)
    print(f"  Loaded {n_total} screens")
    
    # Load dataset to get split information
    print(f"\nLoading dataset for split info...")
    dataset = BioGRIDDSPY(
        dataset_path="./data/biogrid_v0.4_combined",
        split_type=args.split_type,
        fold=args.fold,
    )
    
    train_examples, val_examples, test_examples = dataset.get_train_test_split()
    
    # Create dspy examples to get screen names and question text
    train_dspy = create_dspy_examples(train_examples)
    val_dspy = create_dspy_examples(val_examples)
    test_dspy = create_dspy_examples(test_examples)
    
    # Map (screen_name, reverse) pairs to matrix indices
    screen_pair_to_idx = {}
    for i, label in enumerate(all_screen_labels):
        reverse = all_screen_metadata[i]['reverse']
        screen_pair_to_idx[(label, reverse)] = i
    
    def format_screen_name(name: str, reverse: bool) -> str:
        return f"{name} ({'reverse' if reverse else 'forward'})"
    
    def _prepare_split(dspy_examples, split_label: str):
        """Extract names, reverse flags, questions, matrix indices, ground truth."""
        names_raw = [ex.dataset_name for ex in dspy_examples]
        reverses = [ex.reverse for ex in dspy_examples]
        questions = [ex.question for ex in dspy_examples]
        names = [format_screen_name(n, r) for n, r in zip(names_raw, reverses)]
        matrix_indices = [screen_pair_to_idx[(n, r)] for n, r in zip(names_raw, reverses)]
        ground_truth = [all_screens[i] for i in matrix_indices]
        n = len(names)
        print(f"  {split_label} screens: {n}")
        print(f"    reverse=True: {sum(reverses)}, False: {n - sum(reverses)}")
        return names_raw, names, reverses, questions, matrix_indices, ground_truth
    
    train_names_raw, train_names, train_reverse, train_questions, train_matrix_indices, train_ground_truth = \
        _prepare_split(train_dspy, "Train")
    val_names_raw, val_names, val_reverse, val_questions, val_matrix_indices, val_ground_truth = \
        _prepare_split(val_dspy, "Val")
    test_names_raw, test_names, test_reverse, test_questions, test_matrix_indices, test_ground_truth = \
        _prepare_split(test_dspy, "Test")
    
    n_train = len(train_names)
    n_val = len(val_names)
    n_test = len(test_names)
    
    # Create metrics evaluator
    metrics_evaluator = RankingMetrics(
        k_values=[5, 10, 20, 50, 100],
        use_thresholded_scoring=True
    )
    
    # ========================================================================
    # Compute Screen Embeddings
    # ========================================================================
    print("\nComputing screen embeddings from full question text...")
    
    text_cache = load_screen_embedding_cache()
    text_client = get_text_embedding_client()
    
    def _embed_questions(questions, desc):
        embs = []
        for q in tqdm(questions, desc=desc):
            embs.append(compute_screen_embedding(q, text_client, text_cache))
        return np.stack(embs)
    
    train_embeddings = _embed_questions(train_questions, "Train embeddings")
    val_embeddings = _embed_questions(val_questions, "Val embeddings")
    test_embeddings = _embed_questions(test_questions, "Test embeddings")
    
    print(f"  Train embeddings: {train_embeddings.shape}")
    print(f"  Val embeddings:   {val_embeddings.shape}")
    print(f"  Test embeddings:  {test_embeddings.shape}")
    
    # ========================================================================
    # Load novel split (optional)
    # ========================================================================
    novel_names = []
    novel_ground_truth = []
    novel_embeddings = None
    novel_reverse = []
    n_novel = 0
    
    if not args.no_novel:
        novel_paths = args.novel_dataset_paths if args.novel_dataset_paths else DEFAULT_NOVEL_PATHS
        print(f"\nLoading novel split from {len(novel_paths)} path(s)...")
        novel_dspy, novel_ground_truth = load_novel_split(novel_paths)
        n_novel = len(novel_dspy)
        if n_novel > 0:
            novel_names = [
                format_screen_name(ex.dataset_name, getattr(ex, "reverse", False))
                for ex in novel_dspy
            ]
            novel_reverse = [getattr(ex, "reverse", False) for ex in novel_dspy]
            novel_questions = [ex.question for ex in novel_dspy]
            novel_embeddings = _embed_questions(novel_questions, "Novel embeddings")
            print(f"  Novel embeddings: {novel_embeddings.shape}")
        else:
            print("  No novel examples found, skipping novel split.")
    
    # ========================================================================
    # Evaluate all kNN methods on val, test, and novel splits
    # ========================================================================
    
    target_splits = {
        "val": {
            "names": val_names,
            "reverse": val_reverse,
            "matrix_indices": val_matrix_indices,
            "ground_truth": val_ground_truth,
            "embeddings": val_embeddings,
            "n": n_val,
            "has_transfer_matrix": True,
        },
        "test": {
            "names": test_names,
            "reverse": test_reverse,
            "matrix_indices": test_matrix_indices,
            "ground_truth": test_ground_truth,
            "embeddings": test_embeddings,
            "n": n_test,
            "has_transfer_matrix": True,
        },
    }
    
    if n_novel > 0 and novel_embeddings is not None:
        target_splits["novel"] = {
            "names": novel_names,
            "reverse": novel_reverse,
            "matrix_indices": None,
            "ground_truth": novel_ground_truth,
            "embeddings": novel_embeddings,
            "n": n_novel,
            "has_transfer_matrix": False,
        }
    
    # Load model if checkpoint provided
    transfer_model = None
    if args.model_checkpoint:
        checkpoint_path = Path(args.model_checkpoint)
        if not checkpoint_path.exists():
            print(f"  WARNING: Model checkpoint not found: {checkpoint_path}")
        else:
            print(f"\nLoading model checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            saved_args = checkpoint['args']
            print(f"  Model trained on: {saved_args['split_type']}, fold {saved_args['fold']}")
            print(f"  Best epoch: {checkpoint['epoch']}")
            transfer_model = ScreenTransferPredictor(
                screen_dim=train_embeddings.shape[1],
                hidden_dims=saved_args['hidden_dims'],
                dropout=saved_args['dropout']
            ).to(device)
            transfer_model.load_state_dict(checkpoint['model_state_dict'])
            transfer_model.eval()
    
    # method_name -> split_name -> {matched_train_indices, matched_train_names, <metric>: {avg, scores}}
    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    model_predicted_matrices: Dict[str, np.ndarray] = {}
    
    for split_name, split_data in target_splits.items():
        tgt_names = split_data["names"]
        tgt_reverse = split_data["reverse"]
        tgt_matrix_indices = split_data["matrix_indices"]
        tgt_ground_truth = split_data["ground_truth"]
        tgt_embeddings = split_data["embeddings"]
        has_tm = split_data.get("has_transfer_matrix", False)
        
        print(f"\n{'=' * 70}")
        print(f"EVALUATING SPLIT: {split_name} ({split_data['n']} screens)")
        print(f"{'=' * 70}")
        
        # --- Oracle kNN (only when transfer matrix indices exist) ---
        if has_tm and tgt_matrix_indices is not None:
            print(f"\n  1. Oracle kNN (upper bound)")
            oracle_matches, _ = compute_oracle_knn(
                transfer_matrix, train_matrix_indices, tgt_matrix_indices,
                train_reverse, tgt_reverse, args.require_matching_reverse
            )
            oracle_metrics = evaluate_knn_transfer(
                train_ground_truth, tgt_ground_truth, oracle_matches,
                metrics_evaluator, args.top_k
            )
            oracle_split_result = {
                "matched_train_indices": oracle_matches,
                "matched_train_names": [train_names[i] for i in oracle_matches],
            }
            for m in EVAL_METRICS:
                avg = float(np.mean(oracle_metrics[m]))
                oracle_split_result[m] = {
                    "avg": avg,
                    "scores": [float(s) for s in oracle_metrics[m]],
                }
                print(f"     {m}: {avg:.4f}")
            all_results.setdefault("oracle", {})[split_name] = oracle_split_result
        else:
            print(f"\n  1. Oracle kNN — skipped (no transfer matrix for {split_name})")
        
        # --- Embedding kNN ---
        print(f"\n  2. Embedding kNN")
        embedding_matches = compute_embedding_knn(
            train_embeddings, tgt_embeddings,
            train_reverse, tgt_reverse, args.require_matching_reverse
        )
        emb_metrics = evaluate_knn_transfer(
            train_ground_truth, tgt_ground_truth, embedding_matches,
            metrics_evaluator, args.top_k
        )
        emb_split_result = {
            "matched_train_indices": [int(i) for i in embedding_matches],
            "matched_train_names": [train_names[i] for i in embedding_matches],
        }
        for m in EVAL_METRICS:
            avg = float(np.mean(emb_metrics[m]))
            emb_split_result[m] = {
                "avg": avg,
                "scores": [float(s) for s in emb_metrics[m]],
            }
            print(f"     {m}: {avg:.4f}")
        all_results.setdefault("embedding", {})[split_name] = emb_split_result
        
        # --- Model kNN (optional) ---
        if transfer_model is not None:
            print(f"\n  3. Model kNN")
            model_matches, predicted_matrix = compute_model_knn(
                transfer_model, train_embeddings, tgt_embeddings,
                train_reverse, tgt_reverse, device, args.require_matching_reverse
            )
            model_predicted_matrices[split_name] = predicted_matrix
            model_met = evaluate_knn_transfer(
                train_ground_truth, tgt_ground_truth, model_matches,
                metrics_evaluator, args.top_k
            )
            model_split_result = {
                "matched_train_indices": [int(i) for i in model_matches],
                "matched_train_names": [train_names[i] for i in model_matches],
            }
            for m in EVAL_METRICS:
                avg = float(np.mean(model_met[m]))
                model_split_result[m] = {
                    "avg": avg,
                    "scores": [float(s) for s in model_met[m]],
                }
                print(f"     {m}: {avg:.4f}")
            all_results.setdefault("model", {})[split_name] = model_split_result
    
    # ========================================================================
    # Results Summary
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY (adjusted_ndcg@100)")
    print(f"{'=' * 70}")
    
    ref_metric = "adjusted_ndcg@100"
    for split_name in target_splits:
        emb_avg = all_results["embedding"][split_name][ref_metric]["avg"]
        has_oracle = "oracle" in all_results and split_name in all_results["oracle"]
        print(f"\n  {split_name}:")
        if has_oracle:
            oracle_avg = all_results["oracle"][split_name][ref_metric]["avg"]
            print(f"    {'Oracle kNN (upper bound)':<30} {oracle_avg:.4f}")
            pct = 100 * emb_avg / oracle_avg if oracle_avg > 0 else 0
            print(f"    {'Embedding kNN':<30} {emb_avg:.4f}  ({pct:.1f}% of oracle)")
        else:
            print(f"    {'Embedding kNN':<30} {emb_avg:.4f}")
        if "model" in all_results and split_name in all_results["model"]:
            mod_avg = all_results["model"][split_name][ref_metric]["avg"]
            if has_oracle and oracle_avg > 0:
                pct_m = 100 * mod_avg / oracle_avg
                print(f"    {'Model kNN':<30} {mod_avg:.4f}  ({pct_m:.1f}% of oracle)")
            else:
                print(f"    {'Model kNN':<30} {mod_avg:.4f}")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    results = {
        "split_type": args.split_type,
        "fold": args.fold,
        "require_matching_reverse": args.require_matching_reverse,
        "top_k": args.top_k,
        "metrics": list(EVAL_METRICS),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_novel": n_novel,
        "train_screens": train_names,
        "val_screens": val_names,
        "test_screens": test_names,
        "novel_screens": novel_names,
    }
    
    for method_name, splits_dict in all_results.items():
        results[method_name] = splits_dict
    
    if args.model_checkpoint and "model" in results:
        results["model"]["checkpoint"] = str(args.model_checkpoint)
    
    results_file = output_dir / "knn_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")
    
    for split_name, pred_mat in model_predicted_matrices.items():
        out_path = output_dir / f"model_predicted_transfer_{split_name}.npy"
        np.save(out_path, pred_mat)
        print(f"Saved model predictions to {out_path}")
    
    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

