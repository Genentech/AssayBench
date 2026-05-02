"""
Train a neural network classifier to predict gene relevance categories in CRISPR screens.

The model takes as input:
- Text embedding of screen description
- Text embedding of phenotype
- Embedding for reverse True/False
- Gene embedding

And predicts the relevance category: negative (0), zero (1), or positive (2).

Usage:
    # Training mode
    uv run python scripts/train_relevance_predictor_classifier.py --epochs 50 --batch-size 256 --use-class-weights --split-type easy_split --fold 0
    T
    # Evaluation mode (test set evaluation of a saved model)
    uv run python scripts/train_relevance_predictor_classifier.py --eval-run-path debroue1/screensQA/mj05dzbq
"""

import rootutils
rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn.metrics import roc_auc_score
import wandb
from transformers import AutoTokenizer, AutoModel

from screensqa.dataset.dataset import BioGRIDDSPY as ScreenRankingDataset
from promptopt.utils.pert_embeddings import PertGeneEmbeddings
from screensqa.benchmark.ranking_metrics import RankingMetrics
from openai import AzureOpenAI
from bollm_gene_embeddings import BOLLMGeneEmbeddings

NUM_WORKERS=8
# ============================================================================
# Text Embedding Utilities (includes gene embeddings)
# ============================================================================

def get_text_embedding_client():
    """Initialize Azure OpenAI client for text embeddings."""
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["API_KEY_ES2"],
        api_version="2025-04-01-preview"
    )


def get_text_embedding(text: str, client: AzureOpenAI, cache: Dict = None) -> np.ndarray:
    """
    Get text embedding from Azure OpenAI, with caching.
    
    Args:
        text: Input text
        client: Azure OpenAI client
        cache: Optional cache dictionary
    
    Returns:
        Embedding vector
    """
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


def batch_get_text_embeddings(texts: List[str], client: AzureOpenAI, cache: Dict = None) -> np.ndarray:
    """
    Get text embeddings in batch, with caching.
    
    Args:
        texts: List of input texts
        client: Azure OpenAI client
        cache: Optional cache dictionary
    
    Returns:
        Array of embeddings (N x D)
    """
    embeddings = []
    texts_to_embed = []
    indices_to_embed = []
    
    for i, text in enumerate(texts):
        if cache is not None and text in cache:
            embeddings.append(cache[text])
        else:
            texts_to_embed.append(text)
            indices_to_embed.append(i)
            embeddings.append(None)
    
    # Batch embed uncached texts
    if texts_to_embed:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts_to_embed
        )
        
        for idx, embedding_data in zip(indices_to_embed, response.data):
            emb = np.array(embedding_data.embedding, dtype=np.float32)
            embeddings[idx] = emb
            if cache is not None:
                cache[texts_to_embed[indices_to_embed.index(idx)]] = emb
    
    return np.array(embeddings)


def get_genes(example: Dict[str, Any]) -> List[str]:
    """Get genes from example, handling both ScreensQADSPY and BioGRIDDSPY formats."""
    return example.get('genes', example.get('relevance_genes', []))


def get_description(example: Dict[str, Any]) -> str:
    """Get description from example, handling both ScreensQADSPY and BioGRIDDSPY formats."""
    return example.get('description', example.get('screen_rationale', ''))


def get_phenotype(example: Dict[str, Any]) -> str:
    """Get phenotype from example, handling both ScreensQADSPY and BioGRIDDSPY formats."""
    return example.get('phenotype', 'Not specified')


def relevance_score_to_class(score: float, negative_threshold: float = -0.5, positive_threshold: float = 0.5) -> int:
    """
    Convert relevance score to class label.
    
    Args:
        score: Relevance score
        negative_threshold: Threshold below which score is considered negative
        positive_threshold: Threshold above which score is considered positive
    
    Returns:
        Class label: 0 (negative), 1 (zero), 2 (positive)
    """
    if score < negative_threshold:
        return 0  # negative
    elif score > positive_threshold:
        return 2  # positive
    else:
        return 1  # zero


# ============================================================================
# SciBERT Encoder (trainable)
# ============================================================================

class SciBERTEncoder(nn.Module):
    """
    Trainable SciBERT encoder for text embeddings.
    """
    
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", max_length: int = 512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        
        # Output dimension (SciBERT hidden size is 768)
        self.output_dim = self.model.config.hidden_size
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            embeddings: (B, hidden_dim) tensor
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to same device as model
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        
        # Encode
        outputs = self.model(**encoded)
        
        # Use [CLS] token representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings


# ============================================================================
# Dataset
# ============================================================================

class GeneRelevanceClassificationDataset(Dataset):
    """
    PyTorch dataset for gene relevance classification.
    
    Each example is (description_emb, phenotype_emb, reverse_emb, gene_emb) -> class_label
    """
    
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        gene_embedder: PertGeneEmbeddings,
        text_embedding_cache: Dict[str, np.ndarray],
        text_client: Optional[AzureOpenAI],
        negative_threshold: float = -0.5,
        positive_threshold: float = 0.5,
        precompute_embeddings: bool = True,
        use_scibert: bool = False
    ):
        """
        Args:
            examples: List of screen examples from ScreenRankingDataset
            gene_embedder: Gene embedding model
            text_embedding_cache: Cache for text embeddings
            text_client: Azure OpenAI client (None if using SciBERT)
            negative_threshold: Threshold for negative class
            positive_threshold: Threshold for positive class
            precompute_embeddings: Whether to precompute all embeddings upfront
            use_scibert: If True, return raw text instead of precomputed embeddings
        """
        self.examples = examples
        self.gene_embedder = gene_embedder
        self.text_embedding_cache = text_embedding_cache
        self.text_client = text_client
        self.negative_threshold = negative_threshold
        self.positive_threshold = positive_threshold
        self.use_scibert = use_scibert
        
        # Create flat dataset: (example_idx, gene_idx) -> class_label
        self.data_points = []
        class_counts = {0: 0, 1: 0, 2: 0}
        
        for ex_idx, example in enumerate(examples):
            genes = get_genes(example)
            relevance_scores = example['relevance_scores']
            for gene_idx, (gene, score) in enumerate(zip(genes, relevance_scores)):
                class_label = relevance_score_to_class(
                    score, 
                    self.negative_threshold, 
                    self.positive_threshold
                )
                self.data_points.append({
                    'example_idx': ex_idx,
                    'gene': gene,
                    'relevance_score': score,
                    'class_label': class_label
                })
                class_counts[class_label] += 1
        
        # Print class distribution
        total = sum(class_counts.values())
        print(f"  Class distribution:")
        print(f"    Negative (0): {class_counts[0]:,} ({100*class_counts[0]/total:.1f}%)")
        print(f"    Zero (1):     {class_counts[1]:,} ({100*class_counts[1]/total:.1f}%)")
        print(f"    Positive (2): {class_counts[2]:,} ({100*class_counts[2]/total:.1f}%)")
        
        # Precompute embeddings if requested (only for GPT embeddings)
        if precompute_embeddings and not self.use_scibert:
            print("Precomputing embeddings...")
            self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Precompute all text embeddings (GPT only)."""
        if self.use_scibert:
            return  # No precomputation needed for SciBERT
        
        # Get unique descriptions and phenotypes
        unique_descriptions = list(set(get_description(ex) for ex in self.examples))
        unique_phenotypes = list(set(get_phenotype(ex) for ex in self.examples))
        
        print(f"  Embedding {len(unique_descriptions)} unique descriptions...")
        for desc in tqdm(unique_descriptions):
            get_text_embedding(desc, self.text_client, self.text_embedding_cache)
        
        print(f"  Embedding {len(unique_phenotypes)} unique phenotypes...")
        for pheno in tqdm(unique_phenotypes):
            get_text_embedding(pheno, self.text_client, self.text_embedding_cache)
    
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, idx):
        data_point = self.data_points[idx]
        example = self.examples[data_point['example_idx']]
        
        if self.use_scibert:
            # Return raw text for SciBERT encoding
            return {
                'desc_text': get_description(example),
                'pheno_text': get_phenotype(example),
                'reverse_emb': torch.tensor([1.0 if example.get('reverse', False) else 0.0], dtype=torch.float32),
                'gene_emb': torch.from_numpy(self.gene_embedder.get_embedding(data_point['gene']).astype(np.float32)),
                'class_label': torch.tensor(data_point['class_label'], dtype=torch.long)
            }
        else:
            # Get precomputed embeddings (GPT)
            desc_emb = get_text_embedding(
                get_description(example), 
                self.text_client, 
                self.text_embedding_cache
            )
            pheno_emb = get_text_embedding(
                get_phenotype(example), 
                self.text_client, 
                self.text_embedding_cache
            )
            
            # Reverse embedding (simple binary)
            reverse_emb = np.array([1.0 if example.get('reverse', False) else 0.0], dtype=np.float32)
            
            # Gene embedding
            gene_emb = self.gene_embedder.get_embedding(data_point['gene']).astype(np.float32)

            # Class label (target)
            class_label = data_point['class_label']
            
            return {
                'desc_emb': torch.from_numpy(desc_emb),
                'pheno_emb': torch.from_numpy(pheno_emb),
                'reverse_emb': torch.from_numpy(reverse_emb),
                'gene_emb': torch.from_numpy(gene_emb),
                'class_label': torch.tensor(class_label, dtype=torch.long)
            }
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalanced data."""
        class_counts = {0: 0, 1: 0, 2: 0}
        for dp in self.data_points:
            class_counts[dp['class_label']] += 1
        
        total = len(self.data_points)
        num_classes_present = sum(1 for c in class_counts.values() if c > 0)
        
        # Handle empty classes by setting weight to 0
        weights = torch.tensor([
            total / (num_classes_present * class_counts[0]) if class_counts[0] > 0 else 0.0,
            total / (num_classes_present * class_counts[1]) if class_counts[1] > 0 else 0.0,
            total / (num_classes_present * class_counts[2]) if class_counts[2] > 0 else 0.0
        ], dtype=torch.float32)
        
        return weights


# ============================================================================
# Collate Functions
# ============================================================================

def collate_fn_scibert(batch):
    """Custom collate function for SciBERT (handles text)."""
    return {
        'desc_text': [item['desc_text'] for item in batch],
        'pheno_text': [item['pheno_text'] for item in batch],
        'reverse_emb': torch.stack([item['reverse_emb'] for item in batch]),
        'gene_emb': torch.stack([item['gene_emb'] for item in batch]),
        'class_label': torch.stack([item['class_label'] for item in batch])
    }


def collate_fn_gpt(batch):
    """Standard collate function for GPT embeddings."""
    return {
        'desc_emb': torch.stack([item['desc_emb'] for item in batch]),
        'pheno_emb': torch.stack([item['pheno_emb'] for item in batch]),
        'reverse_emb': torch.stack([item['reverse_emb'] for item in batch]),
        'gene_emb': torch.stack([item['gene_emb'] for item in batch]),
        'class_label': torch.stack([item['class_label'] for item in batch])
    }


# ============================================================================
# Model
# ============================================================================

class RelevanceClassifier(nn.Module):
    """
    Neural network to classify gene relevance into 3 categories.
    
    Architecture:
    - Separate encoders for each embedding type
    - Concatenate all embeddings
    - MLP to predict class logits
    """
    
    def __init__(
        self,
        desc_dim: int = 1536,      # text-embedding-3-small dimension
        pheno_dim: int = 1536,
        reverse_dim: int = 1,
        gene_dim: int = 3072,      # Mean of gene embeddings
        num_classes: int = 3,
        hidden_dims: List[int] = [1024, 512, 256],
        dropout: float = 0.2,
        use_scibert: bool = False,
        scibert_model: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.use_scibert = use_scibert
        
        # Text encoders
        if use_scibert:
            if scibert_model is None:
                raise ValueError("scibert_model must be provided when use_scibert=True")
            self.scibert_encoder = scibert_model
            # SciBERT output dimension (768)
            text_dim = self.scibert_encoder.output_dim
            self.desc_proj = nn.Linear(text_dim, 256)
            self.pheno_proj = nn.Linear(text_dim, 256)
        else:
            # Input projections (to normalize different embedding spaces)
            self.desc_proj = nn.Linear(desc_dim, 256)
            self.pheno_proj = nn.Linear(pheno_dim, 256)
        
        self.reverse_proj = nn.Linear(reverse_dim, 16)
        self.gene_proj = nn.Linear(gene_dim, 512)
        
        # Concatenated dimension
        concat_dim = 256 + 256 + 16 + 512  # 1040 
        
        # MLP layers
        layers = []
        in_dim = concat_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer (3 class logits)
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, desc_input, pheno_input, reverse_emb, gene_emb):
        """
        Forward pass.
        
        Args:
            desc_input: (B, desc_dim) embedding OR list of description texts
            pheno_input: (B, pheno_dim) embedding OR list of phenotype texts
            reverse_emb: (B, reverse_dim)
            gene_emb: (B, gene_dim)
        
        Returns:
            logits: (B, num_classes)
        """
        if self.use_scibert:
            # Encode text using SciBERT
            desc_emb = self.scibert_encoder(desc_input)
            pheno_emb = self.scibert_encoder(pheno_input)
        else:
            # Use precomputed embeddings
            desc_emb = desc_input
            pheno_emb = pheno_input
        
        # Project each embedding
        desc_proj = F.relu(self.desc_proj(desc_emb))
        pheno_proj = F.relu(self.pheno_proj(pheno_emb))
        reverse_proj = F.relu(self.reverse_proj(reverse_emb))
        gene_proj = F.relu(self.gene_proj(gene_emb))
        
        # Concatenate
        combined = torch.cat([desc_proj, pheno_proj, reverse_proj, gene_proj], dim=1)
        
        # MLP
        logits = self.mlp(combined)
        
        return logits


# ============================================================================
# Training
# ============================================================================

def compute_ranking_metrics(
    model: nn.Module,
    dataset: 'GeneRelevanceClassificationDataset',
    device: torch.device,
    use_scibert: bool = False,
    batch_size: int = 1024
) -> Dict[str, float]:
    """
    Compute nDCG metrics by ranking genes for each example.
    
    For each example, predict probabilities for all genes and rank by positive class probability.
    Uses batched inference for speed.
    """
    model.eval()
    
    # Initialize metrics evaluator
    metrics_evaluator = RankingMetrics(
        k_values=[5, 10, 20, 50, 100],
        use_thresholded_scoring=True
    )
    
    all_results = []
    
    with torch.no_grad():
        for example in tqdm(dataset.examples, desc="Computing ranking metrics"):
            genes = get_genes(example)
            relevance_scores = example['relevance_scores']
            num_genes = len(genes)
            
            if use_scibert:
                # Use raw text for SciBERT
                desc_input = get_description(example)
                pheno_input = get_phenotype(example)
            else:
                # Get embeddings for this example
                desc_emb = get_text_embedding(
                    get_description(example),
                    dataset.text_client,
                    dataset.text_embedding_cache
                )
                pheno_emb = get_text_embedding(
                    get_phenotype(example),
                    dataset.text_client,
                    dataset.text_embedding_cache
                )
            
            reverse_val = 1.0 if example.get('reverse', False) else 0.0
            
            # Batch all gene embeddings
            gene_embs = np.stack([
                dataset.gene_embedder.get_embedding(gene).astype(np.float32) 
                for gene in genes
            ])
            
            # Process in batches to avoid OOM
            all_positive_probs = []
            for batch_start in range(0, num_genes, batch_size):
                batch_end = min(batch_start + batch_size, num_genes)
                batch_gene_embs = gene_embs[batch_start:batch_end]
                curr_batch_size = batch_end - batch_start
                
                if use_scibert:
                    # Repeat text for batch
                    desc_batch = [desc_input] * curr_batch_size
                    pheno_batch = [pheno_input] * curr_batch_size
                    reverse_tensor = torch.full((curr_batch_size, 1), reverse_val, dtype=torch.float32, device=device)
                    gene_tensor = torch.from_numpy(batch_gene_embs).to(device)
                    
                    logits = model(desc_batch, pheno_batch, reverse_tensor, gene_tensor)
                else:
                    # Repeat embeddings for batch
                    desc_tensor = torch.from_numpy(desc_emb).unsqueeze(0).expand(curr_batch_size, -1).to(device)
                    pheno_tensor = torch.from_numpy(pheno_emb).unsqueeze(0).expand(curr_batch_size, -1).to(device)
                    reverse_tensor = torch.full((curr_batch_size, 1), reverse_val, dtype=torch.float32, device=device)
                    gene_tensor = torch.from_numpy(batch_gene_embs).to(device)
                    
                    logits = model(desc_tensor, pheno_tensor, reverse_tensor, gene_tensor)
                
                probs = F.softmax(logits, dim=1)
                positive_probs = probs[:, 2].cpu().numpy()
                all_positive_probs.extend(positive_probs)
            
            gene_scores = all_positive_probs
            
            # Rank genes by positive probability (descending)
            ranked_indices = np.argsort(gene_scores)[::-1]
            predicted_genes = [genes[i] for i in ranked_indices]
            predicted_values = [gene_scores[i] for i in ranked_indices]
            
            # Create output dict similar to predictor output
            output = {
                'predicted_genes': predicted_genes,
                'predicted_values': predicted_values
            }
            
            # Evaluate
            results = metrics_evaluator.evaluate_from_output(
                output="<Final Answer>" + ", ".join(output["predicted_genes"][:100]) + "</Final Answer>",
                ground_truth_genes=genes,
                relevance_scores=relevance_scores
            )
            all_results.append(results)
    
    # Compute average metrics (only NDCG-like metrics)
    avg_metrics = {}
    if all_results:
        metric_keys = [k for k in all_results[0].keys() 
                      if 'ndcg' in k.lower() or 'precision' in k.lower() or 'recall' in k.lower()]
        
        for key in metric_keys:
            values = [r[key] for r in all_results if key in r]
            if values:
                avg_metrics[f'ranking_{key}'] = float(np.mean(values))
    
    return avg_metrics


def generate_predictions(
    model: nn.Module,
    dataset: 'GeneRelevanceClassificationDataset',
    device: torch.device,
    use_scibert: bool = False,
    batch_size: int = 1024
) -> List[Dict[str, Any]]:
    """
    Generate per-example predictions with ranked genes and scores.
    
    For each example, predicts positive-class probability for every gene,
    ranks them, and returns predictions in the LLM-predictions format.
    
    Args:
        model: Trained RelevanceClassifier
        dataset: GeneRelevanceClassificationDataset (contains .examples)
        device: Torch device
        use_scibert: Whether the model uses SciBERT text encoder
        batch_size: Batch size for inference over genes
    
    Returns:
        List of prediction dicts, one per example, each containing:
            - question: The prompt/question text (if available)
            - predictions: [[gene1, gene2, ...]]  (ranked by score, descending)
            - predicted_scores: [{gene1: score, gene2: score, ...}]  (all genes)
            - truncated: [False]
            - any_truncated: False
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for example in tqdm(dataset.examples, desc="Generating predictions"):
            genes = get_genes(example)
            num_genes = len(genes)
            
            if use_scibert:
                desc_input = get_description(example)
                pheno_input = get_phenotype(example)
            else:
                desc_emb = get_text_embedding(
                    get_description(example),
                    dataset.text_client,
                    dataset.text_embedding_cache
                )
                pheno_emb = get_text_embedding(
                    get_phenotype(example),
                    dataset.text_client,
                    dataset.text_embedding_cache
                )
            
            reverse_val = 1.0 if example.get('reverse', False) else 0.0
            
            # Batch all gene embeddings
            gene_embs = np.stack([
                dataset.gene_embedder.get_embedding(gene).astype(np.float32)
                for gene in genes
            ])
            
            # Process in batches to avoid OOM
            all_positive_probs = []
            for batch_start in range(0, num_genes, batch_size):
                batch_end = min(batch_start + batch_size, num_genes)
                batch_gene_embs = gene_embs[batch_start:batch_end]
                curr_batch_size = batch_end - batch_start
                
                if use_scibert:
                    desc_batch = [desc_input] * curr_batch_size
                    pheno_batch = [pheno_input] * curr_batch_size
                    reverse_tensor = torch.full((curr_batch_size, 1), reverse_val, dtype=torch.float32, device=device)
                    gene_tensor = torch.from_numpy(batch_gene_embs).to(device)
                    logits = model(desc_batch, pheno_batch, reverse_tensor, gene_tensor)
                else:
                    desc_tensor = torch.from_numpy(desc_emb).unsqueeze(0).expand(curr_batch_size, -1).to(device)
                    pheno_tensor = torch.from_numpy(pheno_emb).unsqueeze(0).expand(curr_batch_size, -1).to(device)
                    reverse_tensor = torch.full((curr_batch_size, 1), reverse_val, dtype=torch.float32, device=device)
                    gene_tensor = torch.from_numpy(batch_gene_embs).to(device)
                    logits = model(desc_tensor, pheno_tensor, reverse_tensor, gene_tensor)
                
                probs = F.softmax(logits, dim=1)
                positive_probs = probs[:, 2].cpu().numpy()
                all_positive_probs.extend(positive_probs)
            
            gene_scores = list(all_positive_probs)
            
            # Rank genes by positive probability (descending)
            ranked_indices = np.argsort(gene_scores)[::-1]
            predicted_genes = [genes[i] for i in ranked_indices]
            
            # Build gene -> score mapping for all genes
            gene_score_map = {gene: float(score) for gene, score in zip(genes, gene_scores)}
            
            # Get question text if available
            question_text = example.get('question', '')
            
            pred_entry = {
                'question': question_text,
                'predictions': [predicted_genes],
                'predicted_scores': [gene_score_map],
                'truncated': [False],
                'any_truncated': False
            }
            all_predictions.append(pred_entry)
    
    return all_predictions


def save_split_predictions(
    predictions: List[Dict[str, Any]],
    output_path: Path,
    model_name: str,
    split: str,
    config: Dict[str, Any] = None
):
    """
    Save predictions to a JSON file in the LLM-predictions format.
    
    Args:
        predictions: List of prediction dicts from generate_predictions()
        output_path: Path to save the JSON file
        model_name: Name of the model (e.g., "relevance_classifier")
        split: Data split name ("train", "val", or "test")
        config: Optional config dict to include in the output
    """
    output = {
        'model_name': model_name,
        'split': split,
        'num_examples': len(predictions),
        'predictions': predictions
    }
    if config:
        output['config'] = config
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"  Saved {split} predictions ({len(predictions)} examples) to {output_path}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_scibert: bool = False
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        if use_scibert:
            # Text inputs for SciBERT
            desc_input = batch['desc_text']
            pheno_input = batch['pheno_text']
        else:
            # Precomputed embeddings for GPT
            desc_input = batch['desc_emb'].to(device)
            pheno_input = batch['pheno_emb'].to(device)
        
        reverse_emb = batch['reverse_emb'].to(device)
        gene_emb = batch['gene_emb'].to(device)
        targets = batch['class_label'].to(device)
        
        # Forward pass
        logits = model(desc_input, pheno_input, reverse_emb, gene_emb)
        
        # Cross-entropy loss
        loss = criterion(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        _, predicted = torch.max(logits, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dataset: 'GeneRelevanceClassificationDataset' = None,
    use_scibert: bool = False
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate the model with classification and ranking metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class metrics
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    
    # Collect predictions and targets for ROC AUC
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            if use_scibert:
                # Text inputs for SciBERT
                desc_input = batch['desc_text']
                pheno_input = batch['pheno_text']
            else:
                # Precomputed embeddings for GPT
                desc_input = batch['desc_emb'].to(device)
                pheno_input = batch['pheno_emb'].to(device)
            
            reverse_emb = batch['reverse_emb'].to(device)
            gene_emb = batch['gene_emb'].to(device)
            targets = batch['class_label'].to(device)
            
            logits = model(desc_input, pheno_input, reverse_emb, gene_emb)
            probs = F.softmax(logits, dim=1)
            
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # Per-class accuracy
            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == targets[i]:
                    class_correct[label] += 1
            
            # Store for ROC AUC
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # Per-class accuracies
    class_accuracies = {
        f'class_{k}_acc': class_correct[k] / class_total[k] if class_total[k] > 0 else 0.0
        for k in [0, 1, 2]
    }
    
    # Compute ROC AUC scores (one-vs-rest for each class)
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    try:
        # Negative (0) vs rest
        roc_auc_negative = roc_auc_score(
            (all_targets == 0).astype(int),
            all_probs[:, 0]
        )
        # Zero (1) vs rest
        roc_auc_zero = roc_auc_score(
            (all_targets == 1).astype(int),
            all_probs[:, 1]
        )
        # Positive (2) vs rest
        roc_auc_positive = roc_auc_score(
            (all_targets == 2).astype(int),
            all_probs[:, 2]
        )
        
        class_accuracies.update({
            'roc_auc_negative': roc_auc_negative,
            'roc_auc_zero': roc_auc_zero,
            'roc_auc_positive': roc_auc_positive
        })
    except ValueError as e:
        # Handle case where only one class is present
        print(f"  Warning: Could not compute ROC AUC: {e}")
        class_accuracies.update({
            'roc_auc_negative': 0.0,
            'roc_auc_zero': 0.0,
            'roc_auc_positive': 0.0
        })
    
    # Compute nDCG metrics if dataset is provided
    if dataset is not None:
        print(" Computing nDCG metrics...")
        ndcg_metrics = compute_ranking_metrics(model, dataset, device, use_scibert)
        class_accuracies.update(ndcg_metrics)
    
    return avg_loss, accuracy, class_accuracies


def evaluate_saved_model(args):
    """Load a saved model and evaluate it on the test set."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse wandb run path (e.g., "debroue1/screensQA/mj05dzbq")
    run_path = args.eval_run_path
    if run_path.count('/') != 2:
        raise ValueError(f"Invalid run path format: {run_path}. Expected format: entity/project/run_id")
    
    entity, project, run_id = run_path.split('/')
    
    print("="*80)
    print("EVALUATING SAVED MODEL ON TEST SET")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Loading model from wandb run: {run_path}")
    
    # Get the output directory from wandb run
    api = wandb.Api()
    run = api.run(run_path)
    
    # Get the output directory from the run config
    run_output_dir = run.config.get('output_dir')
    if not run_output_dir:
        raise ValueError(f"No output_dir found in run config for {run_path}")
    
    # Load model checkpoint from the output directory
    checkpoint_path = Path(run_output_dir) / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get training args from checkpoint
    train_args = checkpoint['args']
    print(f"\nModel training configuration:")
    print(f"  Negative threshold: {train_args['negative_threshold']}")
    print(f"  Positive threshold: {train_args['positive_threshold']}")
    print(f"  Hidden dims: {train_args['hidden_dims']}")
    print(f"  Dropout: {train_args['dropout']}")
    print(f"  Text encoder: {train_args.get('text_encoder', 'gpt')}")
    
    # Determine encoder type
    use_scibert = (train_args.get('text_encoder', 'gpt') == 'scibert')
    
    # Load dataset
    print("\nLoading ScreensQA dataset...")
    #dataset = ScreenRankingDataset(max_examples=args.max_examples)
    train_examples, val_examples, test_examples = dataset.get_train_test_split()
    
    print(f"Val examples: {len(val_examples)}")
    print(f"Test examples: {len(test_examples)}")
    val_points = sum(len(get_genes(ex)) for ex in val_examples)
    test_points = sum(len(get_genes(ex)) for ex in test_examples)
    print(f"Val data points: {val_points:,}")
    print(f"Test data points: {test_points:,}")
    
    # Initialize embedders
    print("\nInitializing embedders...")
    # Get gene embedding config from saved model
    gene_embedding_type = train_args.get('gene_embedding', 'genept')
    if gene_embedding_type == "genept":
        print("Using GenePT gene embeddings")
        gene_embedder = PertGeneEmbeddings(embedding_name="genept")
        gene_dim = 3072
    elif gene_embedding_type == "bollm":
        bollm_option = train_args.get('bollm_option', 'GenePT_protein')
        bollm_combine = train_args.get('bollm_combine', 'mean')
        print(f"Using BOLLM gene embeddings (option={bollm_option}, combine={bollm_combine})")
        try:
            bollm_option_parsed = int(bollm_option) if str(bollm_option).isdigit() else bollm_option
        except:
            bollm_option_parsed = bollm_option
        gene_embedder = BOLLMGeneEmbeddings(
            embedding_option=bollm_option_parsed,
            combine=bollm_combine
        )
        gene_dim = gene_embedder.get_gene_dim()
    else:
        raise ValueError(f"Unknown gene embedding: {gene_embedding_type}")
    
    # Initialize text encoder based on saved model configuration
    if use_scibert:
        print(f"Using trainable SciBERT encoder: {train_args.get('scibert_model', 'allenai/scibert_scivocab_uncased')}")
        scibert_encoder = SciBERTEncoder(
            model_name=train_args.get('scibert_model', 'allenai/scibert_scivocab_uncased'),
            max_length=train_args.get('scibert_max_length', 512)
        ).to(device)
        text_client = None
        text_embedding_cache = {}
    else:
        print("Using Azure OpenAI GPT embeddings (fixed)")
        scibert_encoder = None
        text_client = get_text_embedding_client()
        text_embedding_cache = {}
        
        # Load cache from the persistent location (same as training)
        cache_dir = Path("output/relevance_classifier")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "text_embedding_cache.pkl"
        if cache_file.exists():
            print(f"Loading text embedding cache from {cache_file}")
            with open(cache_file, 'rb') as f:
                text_embedding_cache = pickle.load(f)
    
    # Create output directory for evaluation results
    if args.output_dir:
        eval_output_dir = Path(args.output_dir)
    else:
        eval_output_dir = Path("output/relevance_classifier_eval")
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create validation dataset
    print("\nCreating validation dataset:")
    val_dataset = GeneRelevanceClassificationDataset(
        val_examples,
        gene_embedder,
        text_embedding_cache,
        text_client,
        negative_threshold=train_args['negative_threshold'],
        positive_threshold=train_args['positive_threshold'],
        precompute_embeddings=not use_scibert,
        use_scibert=use_scibert
    )
    
    # Create test dataset
    print("\nCreating test dataset:")
    test_dataset = GeneRelevanceClassificationDataset(
        test_examples,
        gene_embedder,
        text_embedding_cache,
        text_client,
        negative_threshold=train_args['negative_threshold'],
        positive_threshold=train_args['positive_threshold'],
        precompute_embeddings=not use_scibert,
        use_scibert=use_scibert
    )
    
    # Save embedding cache (only for GPT)
    if not use_scibert:
        print(f"\nSaving text embedding cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(text_embedding_cache, f)
    
    # Create dataloaders with appropriate collate function
    collate_fn = collate_fn_scibert if use_scibert else collate_fn_gpt
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )
    
    # Initialize model with same architecture
    print("\nInitializing model...")
    model = RelevanceClassifier(
        gene_dim=gene_dim,
        hidden_dims=train_args['hidden_dims'],
        dropout=train_args['dropout'],
        use_scibert=use_scibert,
        scibert_model=scibert_encoder
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Initialize new wandb run for evaluation
    if not args.no_wandb:
        eval_wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"eval_{run_id}",
            tags=(args.wandb_tags or []) + ['evaluation'],
            config={
                'eval_run_path': run_path,
                'original_run_id': run_id,
                'val_examples': len(val_examples),
                'test_examples': len(test_examples),
                'val_points': val_points,
                'test_points': test_points,
                'negative_threshold': train_args['negative_threshold'],
                'positive_threshold': train_args['positive_threshold'],
                'hidden_dims': train_args['hidden_dims'],
                'dropout': train_args['dropout'],
                'batch_size': args.batch_size,
                'num_parameters': num_params
            },
            dir=str(eval_output_dir)
        )
    
    # Evaluate on validation set
    print("\n" + "="*80)
    print("EVALUATING ON VALIDATION SET")
    print("="*80)
    
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, val_class_accs = evaluate(
        model, val_loader, criterion, device, val_dataset, use_scibert
    )
    
    # Prepare validation metrics
    val_metrics = {
        'val/loss': val_loss,
        'val/accuracy': val_acc,
        'val/class_0_acc': val_class_accs['class_0_acc'],
        'val/class_1_acc': val_class_accs['class_1_acc'],
        'val/class_2_acc': val_class_accs['class_2_acc'],
        'val/roc_auc_negative': val_class_accs['roc_auc_negative'],
        'val/roc_auc_zero': val_class_accs['roc_auc_zero'],
        'val/roc_auc_positive': val_class_accs['roc_auc_positive']
    }
    
    # Add ranking metrics if available
    val_ranking_keys = [k for k in val_class_accs.keys() if k.startswith('ranking_')]
    for key in val_ranking_keys:
        val_metrics[f'val/{key}'] = val_class_accs[key]
    
    # Log validation results to console
    print(f"\nValidation Results:")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f}")
    print(f"  Val Class Accuracies:")
    print(f"    Negative: {val_class_accs['class_0_acc']:.4f}")
    print(f"    Zero:     {val_class_accs['class_1_acc']:.4f}")
    print(f"    Positive: {val_class_accs['class_2_acc']:.4f}")
    print(f"  Val ROC AUC (one-vs-rest):")
    print(f"    Negative vs rest: {val_class_accs['roc_auc_negative']:.4f}")
    print(f"    Zero vs rest:     {val_class_accs['roc_auc_zero']:.4f}")
    print(f"    Positive vs rest: {val_class_accs['roc_auc_positive']:.4f}")
    
    # Print validation ranking metrics if available
    if val_ranking_keys:
        print(f"  Val Ranking Metrics (nDCG):")
        for key in sorted(val_ranking_keys):
            metric_name = key.replace('ranking_', '')
            print(f"    {metric_name}: {val_class_accs[key]:.4f}")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_class_accs = evaluate(
        model, test_loader, criterion, device, test_dataset, use_scibert
    )
    
    # Prepare test metrics
    test_metrics = {
        'test/loss': test_loss,
        'test/accuracy': test_acc,
        'test/class_0_acc': test_class_accs['class_0_acc'],
        'test/class_1_acc': test_class_accs['class_1_acc'],
        'test/class_2_acc': test_class_accs['class_2_acc'],
        'test/roc_auc_negative': test_class_accs['roc_auc_negative'],
        'test/roc_auc_zero': test_class_accs['roc_auc_zero'],
        'test/roc_auc_positive': test_class_accs['roc_auc_positive']
    }
    
    # Add ranking metrics if available
    ranking_keys = [k for k in test_class_accs.keys() if k.startswith('ranking_')]
    for key in ranking_keys:
        test_metrics[f'test/{key}'] = test_class_accs[key]
    
    # Log to console
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Class Accuracies:")
    print(f"    Negative: {test_class_accs['class_0_acc']:.4f}")
    print(f"    Zero:     {test_class_accs['class_1_acc']:.4f}")
    print(f"    Positive: {test_class_accs['class_2_acc']:.4f}")
    print(f"  Test ROC AUC (one-vs-rest):")
    print(f"    Negative vs rest: {test_class_accs['roc_auc_negative']:.4f}")
    print(f"    Zero vs rest:     {test_class_accs['roc_auc_zero']:.4f}")
    print(f"    Positive vs rest: {test_class_accs['roc_auc_positive']:.4f}")
    
    # Print ranking metrics if available
    if ranking_keys:
        print(f"  Test Ranking Metrics (nDCG):")
        for key in sorted(ranking_keys):
            metric_name = key.replace('ranking_', '')
            print(f"    {metric_name}: {test_class_accs[key]:.4f}")
    
    # Combine all metrics
    all_metrics = {**val_metrics, **test_metrics}
    
    # ========================================================================
    # Generate and save per-example predictions for val/test
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS FOR VAL/TEST")
    print("="*80)
    
    predictions_dir = eval_output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    model_config = {
        'hidden_dims': train_args['hidden_dims'],
        'dropout': train_args['dropout'],
        'negative_threshold': train_args['negative_threshold'],
        'positive_threshold': train_args['positive_threshold'],
        'text_encoder': train_args.get('text_encoder', 'gpt'),
        'gene_embedding': train_args.get('gene_embedding', 'genept'),
        'eval_run_path': run_path,
    }
    
    model_name = "relevance_classifier"
    
    print("\nGenerating val predictions...")
    val_preds = generate_predictions(model, val_dataset, device, use_scibert, batch_size=args.batch_size)
    save_split_predictions(val_preds, predictions_dir / "val_predictions.json", model_name, "val", model_config)
    
    print("\nGenerating test predictions...")
    test_preds = generate_predictions(model, test_dataset, device, use_scibert, batch_size=args.batch_size)
    save_split_predictions(test_preds, predictions_dir / "test_predictions.json", model_name, "test", model_config)
    
    # Log to wandb
    if not args.no_wandb:
        wandb.log(all_metrics)
        # Also update summary
        for key, value in all_metrics.items():
            wandb.run.summary[key] = value
        
        # Save results to file (both validation and test)
        results_path = eval_output_dir / f"eval_results_{run_id}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'run_path': run_path,
                'val_metrics': {k: float(v) for k, v in val_metrics.items()},
                'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                'eval_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        # Upload results as artifact
        artifact = wandb.Artifact(
            name=f"{eval_wandb_run.name or eval_wandb_run.id}-eval-results",
            type="evaluation",
            description=f"Validation and test set evaluation results for model from {run_path}"
        )
        artifact.add_file(str(results_path))
        # Also upload prediction files
        for pred_file in predictions_dir.glob("*.json"):
            artifact.add_file(str(pred_file))
        wandb.log_artifact(artifact)
        
        wandb.finish()
    else:
        # Save results to file even without wandb
        results_path = eval_output_dir / f"eval_results_{run_id}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'run_path': run_path,
                'val_metrics': {k: float(v) for k, v in val_metrics.items()},
                'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                'eval_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Predictions saved to: {predictions_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description="Train gene relevance classifier")
    
    # Mode selection
    parser.add_argument("--eval-run-path", type=str, default=None,
                        help="W&B run path to evaluate (e.g., debroue1/screensQA/mj05dzbq). "
                             "If provided, will load the saved model and evaluate on test set.")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dims", type=int, nargs='+', default=[1024, 512, 256], help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--negative-threshold", type=float, default=-0.5, help="Threshold for negative class")
    parser.add_argument("--positive-threshold", type=float, default=0.5, help="Threshold for positive class")
    parser.add_argument("--use-class-weights", action="store_true", help="Use class weights for imbalanced data")
    parser.add_argument("--max-examples", type=int, default=None, help="Max dataset examples (for testing)")
    parser.add_argument("--split-type", type=str, default="easy_split", help="Type of split to use (easy_split - medium_split - or hard_split)")
    parser.add_argument("--fold", type=int, default=0, help="fold to use")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Text encoder selection
    parser.add_argument("--text-encoder", type=str, default="gpt", choices=["gpt", "scibert"],
                        help="Text encoder to use: 'gpt' for Azure OpenAI embeddings (default), 'scibert' for trainable SciBERT")
    parser.add_argument("--scibert-model", type=str, default="allenai/scibert_scivocab_uncased",
                        help="SciBERT model name (only used when --text-encoder=scibert)")
    parser.add_argument("--scibert-max-length", type=int, default=512,
                        help="Max sequence length for SciBERT (only used when --text-encoder=scibert)")
    
    # Gene embeddings selection
    parser.add_argument("--gene-embedding", type=str, default="genept", choices=["genept", "bollm"],
                        help="Gene embedding source: 'genept' (default) or 'bollm'")
    parser.add_argument("--bollm-option", type=str, default="GenePT_protein",
                        help="BOLLM embedding option (name or index). Use 'all' for all options. Examples: 'GenePT_protein', '6', 'all'")
    parser.add_argument("--bollm-combine", type=str, default="mean", choices=["mean", "concat"],
                        help="How to combine multiple BOLLM options: 'mean' or 'concat' (only used with --bollm-option=all)")
    
    # Wandb arguments
    parser.add_argument("--wandb-project", type=str, default="screensQA", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-tags", type=str, nargs='+', default=None, help="W&B tags")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    args = parser.parse_args()
    
    # Check if evaluation mode
    if args.eval_run_path:
        evaluate_saved_model(args)
        return
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb first to get the run directory
    if not args.no_wandb:
        # Use wandb's directory for saving models
        wandb_output_dir = None  # Will be set after init
        
        # Prepare config without output_dir first
        config_dict = vars(args).copy()
        original_output_dir = config_dict.pop('output_dir', None)
        
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config=config_dict
        )
        
        # Use wandb's directory for saving models
        output_dir = Path(wandb.run.dir)
        
        # Now update config with the actual output_dir used
        wandb.config.update({
            "device": str(device),
            "output_dir": str(output_dir),
            "original_output_dir_arg": str(original_output_dir) if original_output_dir else None
        }, allow_val_change=True)
    else:
        # Fallback to specified or default output directory if wandb is disabled
        output_dir = Path(args.output_dir) if args.output_dir else Path("output/relevance_classifier")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TRAINING GENE RELEVANCE CLASSIFIER")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Negative threshold: {args.negative_threshold}")
    print(f"Positive threshold: {args.positive_threshold}")
    
    # Load dataset
    print("\nLoading ScreensQA dataset...")
    dataset = ScreenRankingDataset(dataset_path = "./data/biogrid_v0.4_combined",
                                   split_type=args.split_type,
                                   fold=args.fold)
    train_examples, val_examples, test_examples = dataset.get_train_test_split()
    
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples: {len(val_examples)}")
    
    # Count total data points
    train_points = sum(len(get_genes(ex)) for ex in train_examples)
    val_points = sum(len(get_genes(ex)) for ex in val_examples)
    print(f"Train data points: {train_points:,}")
    print(f"Val data points: {val_points:,}")
    
    # Log dataset info to wandb
    if not args.no_wandb:
        wandb.config.update({
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "train_points": train_points,
            "val_points": val_points
        })
    
    # Initialize embedders
    print("\nInitializing embedders...")
    if args.gene_embedding == "genept":
        print("Using GenePT gene embeddings")
        gene_embedder = PertGeneEmbeddings(embedding_name="genept")
        gene_dim = 3072  # GenePT dimension
    elif args.gene_embedding == "bollm":
        print(f"Using BOLLM gene embeddings (option={args.bollm_option}, combine={args.bollm_combine})")
        # Parse BOLLM option (could be name, index, or 'all')
        try:
            bollm_option = int(args.bollm_option) if args.bollm_option.isdigit() else args.bollm_option
        except:
            bollm_option = args.bollm_option
        gene_embedder = BOLLMGeneEmbeddings(
            embedding_option=bollm_option,
            combine=args.bollm_combine
        )
        gene_dim = gene_embedder.get_gene_dim()
    else:
        raise ValueError(f"Unknown gene embedding: {args.gene_embedding}")
    
    # Initialize text encoder based on selection
    use_scibert = (args.text_encoder == "scibert")
    
    if use_scibert:
        print(f"Using trainable SciBERT encoder: {args.scibert_model}")
        scibert_encoder = SciBERTEncoder(
            model_name=args.scibert_model,
            max_length=args.scibert_max_length
        ).to(device)
        text_client = None
        text_embedding_cache = {}
    else:
        print("Using Azure OpenAI GPT embeddings (fixed)")
        scibert_encoder = None
        text_client = get_text_embedding_client()
        text_embedding_cache = {}
        
        # Load cache from a persistent location (not wandb run dir)
        cache_dir = Path("output/relevance_classifier")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "text_embedding_cache.pkl"
        if cache_file.exists():
            print(f"Loading text embedding cache from {cache_file}")
            with open(cache_file, 'rb') as f:
                text_embedding_cache = pickle.load(f)
    
    # Create datasets
    print("\nCreating PyTorch datasets...")
    print("Train dataset:")
    train_dataset = GeneRelevanceClassificationDataset(
        train_examples, 
        gene_embedder,
        text_embedding_cache, 
        text_client,
        negative_threshold=args.negative_threshold,
        positive_threshold=args.positive_threshold,
        precompute_embeddings=not use_scibert,
        use_scibert=use_scibert
    )
    
    print("\nVal dataset:")
    val_dataset = GeneRelevanceClassificationDataset(
        val_examples, 
        gene_embedder,
        text_embedding_cache, 
        text_client,
        negative_threshold=args.negative_threshold,
        positive_threshold=args.positive_threshold,
        precompute_embeddings=not use_scibert,
        use_scibert=use_scibert
    )
    
    # Save embedding cache (only for GPT)
    if not use_scibert:
        print(f"\nSaving text embedding cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(text_embedding_cache, f)
    
    # Create dataloaders with appropriate collate function
    collate_fn = collate_fn_scibert if use_scibert else collate_fn_gpt
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = RelevanceClassifier(
        gene_dim=gene_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        use_scibert=use_scibert,
        scibert_model=scibert_encoder
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if not args.no_wandb:
        wandb.config.update({
            "num_parameters": num_params,
            "trainable_parameters": trainable_params,
            "text_encoder": args.text_encoder,
            "gene_embedding": args.gene_embedding,
            "gene_dim": gene_dim
        })
        if args.gene_embedding == "bollm":
            wandb.config.update({
                "bollm_option": args.bollm_option,
                "bollm_combine": args.bollm_combine
            })
        wandb.watch(model, log="all", log_freq=100)
    
    # Loss function with optional class weights
    if args.use_class_weights:
        class_weights = train_dataset.get_class_weights().to(device)
        print(f"Using class weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        if not args.no_wandb:
            wandb.config.update({
                "class_weights": class_weights.cpu().numpy().tolist()
            })
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_val_loss = float('inf')
    best_val_ndcg = 0.0
    history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, use_scibert)
        
        # Evaluate
        val_loss, val_acc, class_accs = evaluate(model, val_loader, criterion, device, val_dataset, use_scibert)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Prepare metrics for logging
        metrics_log = {
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'val/class_0_acc': class_accs['class_0_acc'],
            'val/class_1_acc': class_accs['class_1_acc'],
            'val/class_2_acc': class_accs['class_2_acc'],
            'val/roc_auc_negative': class_accs['roc_auc_negative'],
            'val/roc_auc_zero': class_accs['roc_auc_zero'],
            'val/roc_auc_positive': class_accs['roc_auc_positive'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        # Add ranking metrics if available
        ranking_keys = [k for k in class_accs.keys() if k.startswith('ranking_')]
        for key in ranking_keys:
            metrics_log[f'val/{key}'] = class_accs[key]
        
        # Log to wandb
        if not args.no_wandb:
            wandb.log(metrics_log)
        
        # Log to console
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Val Class Accuracies:")
        print(f"    Negative: {class_accs['class_0_acc']:.4f}")
        print(f"    Zero:     {class_accs['class_1_acc']:.4f}")
        print(f"    Positive: {class_accs['class_2_acc']:.4f}")
        print(f"  Val ROC AUC (one-vs-rest):")
        print(f"    Negative vs rest: {class_accs['roc_auc_negative']:.4f}")
        print(f"    Zero vs rest:     {class_accs['roc_auc_zero']:.4f}")
        print(f"    Positive vs rest: {class_accs['roc_auc_positive']:.4f}")
        
        # Print ranking metrics if available
        ranking_keys = [k for k in class_accs.keys() if k.startswith('ranking_')]
        if ranking_keys:
            print(f"  Val Ranking Metrics (nDCG):")
            for key in sorted(ranking_keys):
                metric_name = key.replace('ranking_', '')
                print(f"    {metric_name}: {class_accs[key]:.4f}")
        
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            **class_accs,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save best model (based on adjusted_ndcg@100)
        val_ndcg = class_accs.get('ranking_adjusted_ndcg@100', 0.0)
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_val_loss = val_loss
            model_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_ndcg': val_ndcg,
                'class_accuracies': class_accs,
                'args': vars(args)
            }, model_path)
            print(f"  ✓ Saved best model to {model_path} (adjusted_ndcg@100: {val_ndcg:.4f})")
            
            # Log to wandb
            if not args.no_wandb:
                wandb.run.summary["best_val_ndcg"] = best_val_ndcg
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch + 1
    
    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'class_accuracies': class_accs,
        'args': vars(args)
    }, final_model_path)
    
    # Save history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # ========================================================================
    # Evaluate best model on test set
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("="*80)
    
    # Create test dataset
    print("\nCreating test dataset:")
    test_points = sum(len(get_genes(ex)) for ex in test_examples)
    print(f"Test examples: {len(test_examples)}")
    print(f"Test data points: {test_points:,}")
    
    test_dataset = GeneRelevanceClassificationDataset(
        test_examples,
        gene_embedder,
        text_embedding_cache,
        text_client,
        negative_threshold=args.negative_threshold,
        positive_threshold=args.positive_threshold,
        precompute_embeddings=not use_scibert,
        use_scibert=use_scibert
    )
    
    # Save updated embedding cache (only for GPT)
    if not use_scibert:
        print(f"\nSaving text embedding cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(text_embedding_cache, f)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )
    
    # Load best model checkpoint
    print(f"\nLoading best model from {output_dir / 'best_model.pt'}")
    best_checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Best model from epoch {best_checkpoint['epoch']} (val adjusted_ndcg@100: {best_checkpoint['val_ndcg']:.4f})")
    
    # Evaluate on test set
    test_loss, test_acc, test_class_accs = evaluate(
        model, test_loader, criterion, device, test_dataset, use_scibert
    )
    
    # Prepare test metrics
    test_metrics = {
        'test/loss': test_loss,
        'test/accuracy': test_acc,
        'test/class_0_acc': test_class_accs['class_0_acc'],
        'test/class_1_acc': test_class_accs['class_1_acc'],
        'test/class_2_acc': test_class_accs['class_2_acc'],
        'test/roc_auc_negative': test_class_accs['roc_auc_negative'],
        'test/roc_auc_zero': test_class_accs['roc_auc_zero'],
        'test/roc_auc_positive': test_class_accs['roc_auc_positive']
    }
    
    # Add ranking metrics if available
    ranking_keys = [k for k in test_class_accs.keys() if k.startswith('ranking_')]
    for key in ranking_keys:
        test_metrics[f'test/{key}'] = test_class_accs[key]
    
    # Log to console
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Class Accuracies:")
    print(f"    Negative: {test_class_accs['class_0_acc']:.4f}")
    print(f"    Zero:     {test_class_accs['class_1_acc']:.4f}")
    print(f"    Positive: {test_class_accs['class_2_acc']:.4f}")
    print(f"  Test ROC AUC (one-vs-rest):")
    print(f"    Negative vs rest: {test_class_accs['roc_auc_negative']:.4f}")
    print(f"    Zero vs rest:     {test_class_accs['roc_auc_zero']:.4f}")
    print(f"    Positive vs rest: {test_class_accs['roc_auc_positive']:.4f}")
    
    # Print ranking metrics if available
    if ranking_keys:
        print(f"  Test Ranking Metrics (nDCG):")
        for key in sorted(ranking_keys):
            metric_name = key.replace('ranking_', '')
            print(f"    {metric_name}: {test_class_accs[key]:.4f}")
    
    # Log to wandb
    if not args.no_wandb:
        wandb.log(test_metrics)
        # Also update summary
        for key, value in test_metrics.items():
            wandb.run.summary[key] = value
    
    # Save test results to file
    test_results_path = output_dir / "test_results.json"
    with open(test_results_path, 'w') as f:
        json.dump({
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'best_epoch': int(best_checkpoint['epoch']),
            'best_val_ndcg': float(best_checkpoint['val_ndcg']),
            'eval_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    print(f"\nTest results saved to {test_results_path}")
    
    # ========================================================================
    # Generate and save per-example predictions for train/val/test
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS FOR TRAIN/VAL/TEST")
    print("="*80)
    
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    model_config = {
        'hidden_dims': args.hidden_dims,
        'dropout': args.dropout,
        'negative_threshold': args.negative_threshold,
        'positive_threshold': args.positive_threshold,
        'text_encoder': args.text_encoder,
        'gene_embedding': args.gene_embedding,
        'best_epoch': int(best_checkpoint['epoch']),
        'best_val_ndcg': float(best_checkpoint['val_ndcg']),
        'split_type': args.split_type,
        'fold': args.fold,
    }
    
    model_name = "relevance_classifier"
    
    print("\nGenerating train predictions...")
    train_preds = generate_predictions(model, train_dataset, device, use_scibert, batch_size=args.batch_size)
    save_split_predictions(train_preds, predictions_dir / "train_predictions.json", model_name, "train", model_config)
    
    print("\nGenerating val predictions...")
    val_preds = generate_predictions(model, val_dataset, device, use_scibert, batch_size=args.batch_size)
    save_split_predictions(val_preds, predictions_dir / "val_predictions.json", model_name, "val", model_config)
    
    print("\nGenerating test predictions...")
    test_preds = generate_predictions(model, test_dataset, device, use_scibert, batch_size=args.batch_size)
    save_split_predictions(test_preds, predictions_dir / "test_predictions.json", model_name, "test", model_config)
    
    # Upload artifacts to wandb
    if not args.no_wandb:
        artifact = wandb.Artifact(
            name=f"{wandb.run.name or wandb.run.id}-model",
            type="model",
            description="Trained relevance classifier model"
        )
        artifact.add_file(str(output_dir / "best_model.pt"))
        artifact.add_file(str(output_dir / "final_model.pt"))
        artifact.add_file(str(history_path))
        artifact.add_file(str(test_results_path))
        # Also upload prediction files
        for pred_file in predictions_dir.glob("*.json"):
            artifact.add_file(str(pred_file))
        wandb.log_artifact(artifact)
        
        # Finish wandb run
        wandb.finish()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Val adjusted_ndcg@100: {best_val_ndcg:.4f} (Loss: {best_val_loss:.4f})")
    print(f"Test adjusted_ndcg@100: {test_class_accs.get('ranking_adjusted_ndcg@100', 0.0):.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"Predictions saved to: {predictions_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
