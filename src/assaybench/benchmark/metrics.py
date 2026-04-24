"""Ranking evaluation metrics (NDCG, MRR, Precision@K, Recall@K)."""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Set, Union
import csv
import random
from importlib.resources import files
from pathlib import Path


def roc_auc_score(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    pos_mask = y_true == 1
    n_pos = pos_mask.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Only one class present in y_true.")

    order = np.argsort(y_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_scores) + 1, dtype=float)

    sorted_scores = y_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    return (ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

# All available metric groups
ALL_METRIC_GROUPS = {"ndcg", "adjusted_ndcg", "auroc", "mrr", "precision", "recall", "fdr"}

# Import GeneMapper from the package utils
from assaybench.utils.gene_mapper import GeneMapper

class RankingMetrics:
    """
    Compute ranking metrics like nDCG, MRR, Precision@K, and Recall@K.
    
    Supports two relevance scoring modes:
    - Thresholded: relevance = |effect_score| if p-value < alpha, else 0
    - Standard: relevance = -log10(p-value)
    """
    
    def __init__(
        self,
        k_values: List[int] = None,
        use_thresholded_scoring: bool = True,
        no_hgnc_penalty: float = None,
        metric_groups: Optional[List[str]] = None,
        use_gene_mapper: bool = True,
        gene_mapper: Optional["GeneMapper"] = None,
    ):
        """
        Initialize the ranking metrics.
        
        Args:
            k_values: List of k values for Precision@K and Recall@K
                     (default: [5, 10, 20, 50, 100])
            use_thresholded_scoring: If True, use thresholded scoring where
                                     relevance = |effect_score| if pvalue < alpha,
                                     else 0. If False, use -log10(pvalue) as relevance.
            no_hgnc_penalty: Relevance given to predictions that are not valid HGNC gene symbols. By default, None - considers it is a non-scored gene.
            metric_groups: List of metric groups to compute. Valid groups are:
                          'ndcg', 'adjusted_ndcg', 'auroc', 'mrr', 'precision', 'recall'.
                          If None, all metric groups are computed (default).
            use_gene_mapper: If True, use GeneMapper to normalize predicted gene symbols
                            to HGNC symbols before evaluation. This handles common LLM
                            output patterns like "BBC3 (PUMA)", numbered lists, etc.
            gene_mapper: Optional pre-initialized GeneMapper instance. If None and
                        use_gene_mapper=True, a new GeneMapper will be created.
        """
        self.k_values = k_values or [5, 10, 20, 50, 100]
        self.use_thresholded_scoring = use_thresholded_scoring

        self.num_random_samples = 10

        hgnc_path = files("assaybench.data.hgnc") / "all_genes.tsv"
        with open(hgnc_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            self._hgnc_approved_symbols = {row["Approved symbol"] for row in reader}

        self.no_hgnc_penalty = no_hgnc_penalty

        # Initialize gene mapper if requested
        self.use_gene_mapper = use_gene_mapper
        if use_gene_mapper:
            if gene_mapper is not None:
                self.gene_mapper = gene_mapper
            else:
                self.gene_mapper = GeneMapper(verbose=False)
        else:
            self.gene_mapper = None

        # Validate and store metric groups
        if metric_groups is not None:
            invalid_groups = set(metric_groups) - ALL_METRIC_GROUPS
            if invalid_groups:
                raise ValueError(
                    f"Invalid metric groups: {invalid_groups}. "
                    f"Valid groups are: {ALL_METRIC_GROUPS}"
                )
            self.metric_groups: Set[str] = set(metric_groups)
        else:
            self.metric_groups: Set[str] = ALL_METRIC_GROUPS

    
    def extract_genes_from_output(self, output: str) -> List[str]:
        """
        Extract predicted gene ranking from model output.
        
        Looks for genes in <Final Answer>...</Final Answer> tags or as comma-separated list.
        
        Args:
            output: Model output string
            
        Returns:
            List of gene names in predicted order
        """
        # Try to find <Final Answer> tags
        match = re.search(r'<Final Answer>(.*?)</Final Answer>', output, re.DOTALL | re.IGNORECASE)
        if match:
            genes_text = match.group(1)
        else:
            # Use the whole output if no tags found
            genes_text = output

        # Split by commas and clean up
        genes = [g.strip().upper() for g in genes_text.split(',')]
        
        # Filter out empty strings
        genes = [g for g in genes if g]
        
        return genes
    
    
    def compute_dcg(self, relevances: List[float], k: Optional[int] = None) -> float:
        """
        Compute Discounted Cumulative Gain.
        
        Args:
            relevances: List of relevance scores in ranked order
            k: Cutoff position (default: None = use all)
            
        Returns:
            DCG score
        """

        # we penalize the model if not enough samples are provided
        penalized_relevances = relevances + [0.0] * (k - len(relevances)) if k is not None and len(relevances) < k else relevances

        # We only take the top k elements (will penalize genes without a relevance score)
        if k is not None:
            penalized_relevances = penalized_relevances[:k]

        # Skip unscored genes (None) using condensed list evaluation: genes not
        # in the ground truth are removed and the remaining genes are packed into
        # consecutive positions. This is appropriate because models cannot know
        # which genes were tested in a given screen.
        condensed_relevances = [rel for rel in penalized_relevances if rel is not None]

        if k is not None:
            condensed_relevances = condensed_relevances[:k]
        
        if len(condensed_relevances) == 0:
            return 0.0
        
        # DCG = sum(rel_i / log2(i + 2)) for i in [0, len-1]
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(condensed_relevances))
        return dcg
    
    def compute_ndcg(
        self,
        predicted_relevances: List[float],
        all_relevances,
        k: Optional[int] = None
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain.
        
        Args:
            predicted_relevances: Relevance scores in predicted order
            k: Cutoff position (default: None = use all)
            
        Returns:
            nDCG score (0 to 1)
        """
        # Compute DCG for predicted ranking (negatives penalize the score)
        dcg = self.compute_dcg(predicted_relevances, k)
        
        # Compute ideal DCG. Clip negative relevances to 0 because the ideal
        # predictor would avoid genes with negative relevance (predicting an
        # unknown gene at 0.0 is always better than predicting a negative one).
        # Without this, negatives deflate IDCG, allowing DCG/IDCG > 1.
        ideal_relevances = sorted([max(r, 0.0) for r in all_relevances], reverse=True)
        idcg = self.compute_dcg(ideal_relevances, k = k)
        
        # Normalize
        if idcg == 0:
            return 0.0
        
        return dcg / idcg


    def compute_adjusted_ndcg(
        self,
        predicted_relevances: List[float],
        all_relevances,
        k: Optional[int] = None
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain.
        
        Args:
            predicted_relevances: Relevance scores in predicted order
            k: Cutoff position (default: None = use all)
            
        Returns:
            adj_nDCG score (-inf to 1)
        """
        # Compute DCG for predicted ranking
        ndcg = self.compute_ndcg(predicted_relevances, all_relevances, k)

        ndcg_rand = self.compute_ndcg(min(k, len(all_relevances)) * [np.mean(all_relevances)], all_relevances, k)

        #print('\n')
        #print('k', k, flush=True)
        #print('ndcg', ndcg, flush=True)
        #print('ndcg_rand', ndcg_rand, '±', np.std(ndcg_rand_list), flush=True)
        #print('ndcg_rand_list', ndcg_rand_list, flush=True)
        #print('adjusted_ndcg', (ndcg - ndcg_rand) / (1 - ndcg_rand), flush=True)

        adj_ndcg = (ndcg - ndcg_rand) / (1 - ndcg_rand)

        return max(adj_ndcg, 0) #we don't care about measuring how much worse than random it is, and values < 0 make the mean across screens confusing


    def compute_mrr(self, predicted_relevances: List[float]) -> float:
        """
        Compute Mean Reciprocal Rank.
        
        Finds the position of the first relevant item (relevance > 0).
        
        Args:
            predicted_relevances: Relevance scores in predicted order
            
        Returns:
            MRR score (1/rank of first relevant item, or 0 if none)
        """
        condensed_relevances = [rel for rel in predicted_relevances if rel is not None]
        for i, rel in enumerate(condensed_relevances):
            if rel > 0:
                return 1.0 / (i + 1)
        return 0.0
    
    def compute_precision_at_k(
        self,
        predicted_relevances: List[float],
        k: int,
        relevance_threshold: float,
        normalization: Optional[float] = None
    ) -> float:
        """
        Compute Precision@K.
        
        Args:
            predicted_relevances: Relevance scores in predicted order
            k: Cutoff position
            relevance_threshold: Threshold for considering an item "relevant"
            normalization: Optional normalization factor (default: None)
        Returns:
            Precision@K score (fraction of top-k items that are relevant)
        """
        condensed_relevances = [rel for rel in predicted_relevances if rel is not None]
        if k > len(condensed_relevances):
            k = len(condensed_relevances)
        
        if k == 0:
            return 0.0
        
        top_k = condensed_relevances[:k]
        num_relevant = sum(1 for rel in top_k if rel > relevance_threshold)
        
        if normalization is not None:
            if normalization == 0:
                return np.nan
            else:
                return num_relevant / min(normalization,k)
        
        return num_relevant / k
    
    def compute_recall_at_k(
        self,
        predicted_relevances: List[float],
        all_relevances: List[float],
        k: int,
        relevance_threshold: float
    ) -> float:
        """
        Compute Recall@K.
        
        Args:
            predicted_relevances: Relevance scores in predicted order
            all_relevances: All relevance scores (for computing total relevant items)
            k: Cutoff position
            relevance_threshold: Threshold for considering an item "relevant"
            
        Returns:
            Recall@K score (fraction of all relevant items in top-k)
        """
        condensed_relevances = [rel for rel in predicted_relevances if rel is not None]
        if k > len(condensed_relevances):
            k = len(condensed_relevances)
        
        total_relevant = sum(1 for rel in all_relevances if rel > relevance_threshold)
        
        if total_relevant == 0:
            return 0.0
        
        top_k = condensed_relevances[:k]
        num_relevant_retrieved = sum(1 for rel in top_k if rel > relevance_threshold)
        
        return num_relevant_retrieved / total_relevant

    def compute_fdr_at_k(
        self,
        predicted_relevances: List[float],
        k: int,
        relevance_threshold: float,
        normalization: Optional[float] = None
    ) -> float:
        """
        Compute Inverse Precision@K (proportion of negative values in top K)
        
        Args:
            predicted_relevances: Relevance scores in predicted order
            k: Cutoff position
            relevance_threshold: Threshold for considering an item "relevant"
            normalization: Optional normalization factor (default: None)
            
        Returns:
            Inverse Precision@K score (fraction of top-k items that are not relevant)
        """
        condensed_relevances = [rel for rel in predicted_relevances if rel is not None]
        if k > len(condensed_relevances):
            k = len(condensed_relevances)
        
        if k == 0:
            return 0.0
        
        top_k = condensed_relevances[:k]
        num_relevant = sum(1 for rel in top_k if rel < relevance_threshold)

        if normalization is not None:
            if normalization == 0:
                return np.nan
            else:
                return num_relevant / min(normalization,k)
        
        return num_relevant / k
    
    def compute_auroc(
        self,
        predicted_genes: List[str],
        ground_truth_genes: List[str],
        relevance_scores: List[float],
        relevance_threshold: float,
        k: Optional[int] = None
    ) -> float:
        """
        Compute Area Under ROC Curve using rank-based scoring.
        
        Args:
            predicted_genes: Predicted gene ranking
            ground_truth_genes: Ground truth gene list
            relevance_scores: Relevance scores for each ground truth gene
            relevance_threshold: Threshold for considering an item "relevant"
            k: Cutoff position (default: None = use all)
            
        Returns:
            AUROC score (0 to 1)
        """
        # For AUROC@K, select top K positive genes + all negative genes
        if k is not None:
            # Split genes into positives and negatives by threshold
            positive_pairs = [(gene, score) for gene, score in zip(ground_truth_genes, relevance_scores) 
                             if score > relevance_threshold]
            negative_pairs = [(gene, score) for gene, score in zip(ground_truth_genes, relevance_scores) 
                             if score <= relevance_threshold]
            
            if k < len(positive_pairs):
                # Sort positive genes by relevance score (descending) and take top K
                positive_pairs.sort(key=lambda x: x[1], reverse=True)
                selected_positive_pairs = positive_pairs[:k]
                
                # Combine top K positives + all negatives
                selected_pairs = selected_positive_pairs + negative_pairs
                ground_truth_genes = [pair[0] for pair in selected_pairs]
                relevance_scores = [pair[1] for pair in selected_pairs]
        
        # Create binary labels: relevance_score > threshold is positive class
        y_true = [1 if score > relevance_threshold else 0 for score in relevance_scores]
        
        # If no positive or no negative samples, return NaN
        if sum(y_true) == 0 or sum(y_true) == len(y_true):
            return np.nan
        
        # Create gene-to-rank mapping for O(1) lookup using original predicted list
        gene_to_rank = {gene: rank for rank, gene in enumerate(predicted_genes)}
        total_genes = len(ground_truth_genes)
        
        # Assign rank-based scores: higher rank = higher score
        y_scores = [
            total_genes - gene_to_rank[gt_gene] if gt_gene in gene_to_rank else 0
            for gt_gene in ground_truth_genes
        ]
        
        try:
            return roc_auc_score(y_true, y_scores)
        except ValueError:
            # Handle edge cases where AUROC cannot be computed
            return np.nan
    
    def normalize_gene(self, gene: str, not_found = "original") -> str:
        """
        Normalize a gene symbol using the gene mapper if enabled.
        
        Args:
            gene: Raw gene symbol from model output
            not_found: if "original", returns the original (uppercased) gene symbol when no mapping is found. If None (default), returns the normalized gene symbol if found, otherwise None.
            
        Returns:
            Normalized HGNC symbol if found, otherwise the original (uppercased)
        """
        gene = gene.strip().upper()
        
        if self.gene_mapper is not None:
            mapped = self.gene_mapper.map_gene(gene)
            if mapped is not None:
                return mapped
        if not_found == "original":
            return gene
        else:
            return None
    
    def normalize_genes(self, genes: List[str], not_found = "original") -> List[str]:
        """
        Normalize a list of gene symbols.
        
        Args:
            genes: List of raw gene symbols
            
        Returns:
            List of normalized gene symbols
        """
        return [self.normalize_gene(g, not_found=not_found) for g in genes]
    
    def gene_scoring_fn(self, scoring_dict, gene):
        # First try to normalize the gene if mapper is enabled
        normalized_gene = self.normalize_gene(gene) if self.gene_mapper else gene.strip().upper()
        
        if normalized_gene in scoring_dict:
            return scoring_dict[normalized_gene]
        elif normalized_gene not in self._hgnc_approved_symbols:
            return self.no_hgnc_penalty  # unknown gene
        else:
            return None
    
    def evaluate(
        self,
        predicted_genes: List[str],
        ground_truth_genes: List[str],
        relevance_scores: Optional[List[float]],
    ) -> Dict[str, float]:
        """
        Evaluate a predicted gene ranking.
        
        Args:
            predicted_genes: Predicted gene ranking
            ground_truth_genes: Ground truth gene list
            relevance_scores: Relevance scores for each ground truth gene
            
        Returns:
            Dictionary of metric scores
        """

        # rescaled relevance_scores
        hit_scaled_relevance_scores = np.array(relevance_scores) 
        n_hits = sum(hit_scaled_relevance_scores > 0)
        n_neg_hits = sum(hit_scaled_relevance_scores < 0)       
        N = len(hit_scaled_relevance_scores)
        if n_hits > 0:
            hit_scaled_relevance_scores[hit_scaled_relevance_scores > 0] = 1 - ((1-hit_scaled_relevance_scores[hit_scaled_relevance_scores > 0]) * (N / n_hits))
        if n_neg_hits>0:
            hit_scaled_relevance_scores[hit_scaled_relevance_scores < 0] = -1 + ((1+hit_scaled_relevance_scores[hit_scaled_relevance_scores < 0]) * (N / n_neg_hits))
        hit_scaled_relevance_scores = hit_scaled_relevance_scores.tolist()
        
        #remove duplicates from predicted genes
        
        predicted_genes = list(dict.fromkeys(predicted_genes))

        # Store raw predictions before normalization
        raw_predicted_genes = [g.strip() for g in predicted_genes]
        raw_ground_truth_genes = [g.strip() for g in ground_truth_genes]
        
        # Normalize predicted genes (uses gene mapper if enabled)
        if self.gene_mapper is not None:
            predicted_genes = self.normalize_genes(predicted_genes)
            
            predicted_genes_none = self.normalize_genes(predicted_genes, not_found = None)
            num_hallucinated_genes = sum(1 for g in predicted_genes_none if g is None)
            hallucination_rate = num_hallucinated_genes / len(raw_predicted_genes) if len(raw_predicted_genes) > 0 else 0.0
            
            # Also normalize ground truth genes to ensure fair comparison
            ground_truth_genes = self.normalize_genes(ground_truth_genes)
        else:
            predicted_genes = [g.strip().upper() for g in predicted_genes]
            ground_truth_genes = [g.strip().upper() for g in ground_truth_genes]

        #remove duplicates from gene list after performing mapping
        predicted_genes = list(dict.fromkeys(predicted_genes))
        
        scoring_dict = dict(zip(ground_truth_genes, relevance_scores)) 
        hit_normalized_scoring_dict = dict(zip(ground_truth_genes, hit_scaled_relevance_scores))
        # Get relevance scores for predicted ranking
        predicted_values = [
            self.gene_scoring_fn(scoring_dict, gene) for gene in predicted_genes
        ]
        hit_scaled_predicted_values = [
            self.gene_scoring_fn(hit_normalized_scoring_dict, gene) for gene in predicted_genes
        ]
        # Compute metrics
        results = {
            'num_predicted': len(predicted_genes),
            'num_genes': len(ground_truth_genes),
            'num_positive_relevance_genes': sum(1 for r in relevance_scores if r > 0),
            'predicted_genes': predicted_genes,
            'raw_predicted_genes': raw_predicted_genes,  # Store original predictions
            'ground_truth_genes': ground_truth_genes,
            'raw_ground_truth_genes': raw_ground_truth_genes,  # Store original ground truth
            'predicted_values': predicted_values,
            'hit_scaled_predicted_values': hit_scaled_predicted_values,
            "hallucination_rate": hallucination_rate if self.gene_mapper is not None else None
        }
        
        # Add gene mapper stats if enabled
        if self.gene_mapper is not None:
            num_predictions_normalized = sum(1 for raw, norm in zip(raw_predicted_genes, predicted_genes) 
                                if raw.strip().upper() != norm)
            num_gt_normalized = sum(1 for raw, norm in zip(raw_ground_truth_genes, ground_truth_genes) 
                                if raw.strip().upper() != norm)
            results['num_predictions_normalized'] = num_predictions_normalized
            results['num_ground_truth_normalized'] = num_gt_normalized


        # nDCG at different cutoffs
        if "ndcg" in self.metric_groups or "adjusted_ndcg" in self.metric_groups:
            for k in self.k_values:
                if "ndcg" in self.metric_groups:
                    results[f'ndcg@{k}'] = self.compute_ndcg(predicted_values, relevance_scores, k)
                    results[f'hit_scaled_ndcg@{k}'] = self.compute_ndcg(hit_scaled_predicted_values, hit_scaled_relevance_scores, k)
                if "adjusted_ndcg" in self.metric_groups:
                    results[f'adjusted_ndcg@{k}'] = self.compute_adjusted_ndcg(predicted_values, relevance_scores, k)
                    results[f'hit_scaled_adjusted_ndcg@{k}'] = self.compute_adjusted_ndcg(hit_scaled_predicted_values, hit_scaled_relevance_scores, k)
        # Use median relevance as threshold for "relevant"
        #relevance_threshold = np.median(relevance_scores) if len(relevance_scores) > 0 else 0.0
        relevance_threshold = 0 # Significant genes have relevance > 0 in our setting
        
        # AUROC at different cutoffs
        if "auroc" in self.metric_groups:
            for k in self.k_values:
                results[f'auroc@{k}'] = self.compute_auroc(predicted_genes, ground_truth_genes, relevance_scores, relevance_threshold, k)
            # Overall AUROC
            results['auroc'] = self.compute_auroc(predicted_genes, ground_truth_genes, relevance_scores, relevance_threshold)

        # Overall nDCG
        if "ndcg" in self.metric_groups:
            results['ndcg'] = self.compute_ndcg(predicted_values, relevance_scores, k = len(predicted_values))
        if "adjusted_ndcg" in self.metric_groups:
            results['adjusted_ndcg'] = self.compute_adjusted_ndcg(predicted_values, relevance_scores, k = len(predicted_values))

        # MRR
        if "mrr" in self.metric_groups:
            results['mrr'] = self.compute_mrr(predicted_values)
        
        # Precision@K and Recall@K
        if "precision" in self.metric_groups or "recall" in self.metric_groups:
            for k in self.k_values:
                if k <= len(predicted_values):
                    if "precision" in self.metric_groups:
                        results[f'precision@{k}'] = self.compute_precision_at_k(
                            predicted_values, k, relevance_threshold
                        )
                        n_pos_rel = sum(1 for r in relevance_scores if r > relevance_threshold)
                        results[f'normalized_precision@{k}'] = self.compute_precision_at_k(predicted_values, k, relevance_threshold, normalization = n_pos_rel)
                    if "recall" in self.metric_groups:
                        results[f'recall@{k}'] = self.compute_recall_at_k(
                            predicted_values, relevance_scores, k, relevance_threshold
                        )
        if "fdr" in self.metric_groups:
            for k in self.k_values:
                if k <= len(predicted_values):
                    results[f'fdr@{k}'] = self.compute_fdr_at_k(
                        predicted_values, k, relevance_threshold
                    )
                    n_neg_rel = sum(1 for rel in relevance_scores if rel < relevance_threshold)
                    results[f'normalized_fdr@{k}'] = self.compute_fdr_at_k(predicted_values, k, relevance_threshold, normalization = n_neg_rel)
                    
 
        return results
    
    def evaluate_from_output(
        self,
        output: str,
        ground_truth_genes: List[str],
        relevance_scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model output string.
        
        Args:
            output: Model output string containing predicted genes
            ground_truth_genes: Ground truth gene list
            pvalues: P-values for each ground truth gene
            effect_scores: Effect scores for each ground truth gene
            alpha: Significance threshold
            
        Returns:
            Dictionary of metric scores
        """
        # Extract genes from output
        predicted_genes = self.extract_genes_from_output(output)
        # Evaluate
        return self.evaluate(
            predicted_genes,
            ground_truth_genes,
            relevance_scores=relevance_scores,
        )
