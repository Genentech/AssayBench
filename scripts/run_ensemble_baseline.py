"""
Script for evaluating an ensemble baseline on the ScreenBench ranking task.

This script:
1. Loads the ScreenBench ranking dataset
2. Runs the model multiple times (n_runs) for each question
3. Ensembles predictions by combining rank and frequency information
4. Evaluates the ensemble predictions on the test set
"""

import rootutils
rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

# Load environment variables from .env file
import dotenv
dotenv.load_dotenv('.env')

import os
import sys
import json
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

import dspy
from dspy import Example

from screensqa.dataset.dataset import ScreensQADSPY
from screensqa.benchmark.ranking_metrics import RankingMetrics



def create_ranking_signature(system_prompt: str = None):
    """
    Create a RankingSignature class with the given system prompt as docstring.
    
    Args:
        system_prompt: System prompt to use as the signature's docstring.
                      If None, uses a default prompt.
    
    Returns:
        A dspy.Signature class for ranking
    """
    if system_prompt is None:
        system_prompt = "Signature for gene ranking task."
    
    # Create a new signature class with the provided docstring
    class RankingSignature(dspy.Signature):
        __doc__ = system_prompt
        
        question = dspy.InputField(desc="The gene ranking task description")
        answer = dspy.OutputField(desc="Comma-separated list of genes in ranked order")
    
    return RankingSignature


class RankingModule(dspy.Module):
    """DSPy module for gene ranking."""
    
    def __init__(self, signature_class=None):
        super().__init__()
        # Use provided signature class or create a default one
        if signature_class is None:
            signature_class = create_ranking_signature()
        
        # Always use a standard DSPy predictor
        # The LM backend (GNEsys or standard) is configured via dspy.configure()
        self.predictor = dspy.ChainOfThought(signature_class)
    
    def forward(self, question: str) -> dspy.Prediction:
        """
        Forward pass for ranking.
        
        Args:
            question: The ranking task question
            
        Returns:
            dspy.Prediction with 'answer' field
        """
        result = self.predictor(question=question)
        return result


def create_dspy_examples(dataset_examples: List[Dict[str, Any]]) -> List[Example]:
    """
    Convert dataset examples to DSPy Example format.
    
    Supports both ScreensQADSPY and BioGRIDDSPY dataset formats:
    - ScreensQADSPY uses 'genes' key
    - BioGRIDDSPY uses 'relevance_genes' key
    
    Args:
        dataset_examples: List of examples from ScreensQADSPY or BioGRIDDSPY
        
    Returns:
        List of dspy.Example objects
    """
    dspy_examples = []
    
    for ex in dataset_examples:
        # Create DSPy example with question and ground truth answer
        dspy_ex = dspy.Example(
            question=ex['question'],
            answer=ex.get('answer', '')
        ).with_inputs("question")
        
        # Store additional metadata for evaluation
        # Handle both ScreensQADSPY ('genes') and BioGRIDDSPY ('relevance_genes')
        dspy_ex.genes = ex.get('genes', ex.get('relevance_genes', []))
        dspy_ex.relevance_scores = ex.get('relevance_scores', [])
        dspy_ex.dataset_name = ex.get('dataset_name', '')
        dspy_ex.phenotype = ex.get('phenotype', '')
        dspy_ex.num_genes = ex.get('num_genes', 0)
        
        # ScreensQADSPY-specific fields (with defaults for BioGRID)
        dspy_ex.alpha = ex.get('alpha', None)
        dspy_ex.ranking_method = ex.get('ranking_method', None)
        dspy_ex.description = ex.get('description', ex.get('screen_rationale', ''))
        dspy_ex.split = ex.get('split', None)
        dspy_ex.reverse = ex.get('reverse', False)
        
        # Additional metadata fields for analysis (shared)
        dspy_ex.public_screen = ex.get('public_screen', None)
        dspy_ex.crispr_type = ex.get('crispr_type', None)
        dspy_ex.cell_type = ex.get('cell_type', None)
        dspy_ex.screen_name = ex.get('screen_name', None)
        dspy_ex.screen_year = ex.get('screen_year', None)
        dspy_ex.caveats = ex.get('caveats', None)
        
        # BioGRIDDSPY-specific fields
        dspy_ex.hit = ex.get('hit', None)
        dspy_ex.cell_line = ex.get('cell_line', None)
        dspy_ex.screen_type = ex.get('screen_type', None)
        dspy_ex.library_methodology = ex.get('library_methodology', None)
        dspy_ex.screen_rationale = ex.get('screen_rationale', None)
        
        dspy_examples.append(dspy_ex)
    
    return dspy_examples


def parse_genes_from_output(output_text: str) -> List[str]:
    """
    Parse gene list from model output.
    
    Args:
        output_text: Raw output from the model
        
    Returns:
        List of gene symbols in order
    """
    # Simple comma-separated parsing
    genes = [g.strip() for g in output_text.split(',')]
    genes = [g for g in genes if g]  # Remove empty strings
    return genes


def ensemble_predictions(
    predictions: List[List[str]],
    method: str = 'rrf',
    top_k: int = 100,
    min_appearances: int = 2,
    topk_threshold: int = 20,
    rrf_k: float = 60.0,
) -> List[str]:
    """
    Ensemble multiple predictions into a single ranked list.
    
    Args:
        predictions: List of gene lists, one per run
        method: Ensemble method ('rrf', 'rank_frequency', 'frequency', 'rank_average',
                'reciprocal_rank', 'intersection_only', 'topk_threshold')
        top_k: Number of top genes to return
        min_appearances: For 'intersection_only', minimum number of runs a gene must appear in
        topk_threshold: For 'topk_threshold', only consider genes ranked in top-N of at least one run
        rrf_k: Dampening constant for RRF (standard default: 60)
        
    Returns:
        Ensembled list of genes
    """
    if method == 'rrf':
        scores: Dict[str, float] = defaultdict(float)
        for pred in predictions:
            for rank_0, gene in enumerate(pred):
                scores[gene] += 1.0 / (rrf_k + rank_0 + 1)
        sorted_genes = sorted(scores, key=scores.get, reverse=True)
        return sorted_genes[:top_k]

    gene_scores = defaultdict(lambda: {'ranks': [], 'count': 0})
    
    # Collect rank and frequency information
    for pred in predictions:
        for rank, gene in enumerate(pred):
            gene_scores[gene]['ranks'].append(rank + 1)  # 1-indexed rank
            gene_scores[gene]['count'] += 1

    # Compute ensemble scores
    if method == 'rank_frequency':
        # Combine average rank (lower is better) with frequency (higher is better)
        # Normalize both to [0, 1] range and combine
        max_count = max(gs['count'] for gs in gene_scores.values())
        max_rank = max(max(gs['ranks']) for gs in gene_scores.values())
        
        for gene, gs in gene_scores.items():
            avg_rank = np.mean(gs['ranks'])
            freq = gs['count']
            
            # Lower rank score is better, higher frequency is better
            # Normalize: rank_score in [0, 1] where 0 is best
            # freq_score in [0, 1] where 1 is best
            rank_score = (avg_rank - 1) / max_rank if max_rank > 0 else 0
            freq_score = freq / max_count if max_count > 0 else 0
            
            # Combined score: minimize rank, maximize frequency
            # Lower combined score is better
            gene_scores[gene]['score'] = rank_score - freq_score
    
    elif method == 'frequency':
        # Sort by frequency only
        for gene, gs in gene_scores.items():
            gene_scores[gene]['score'] = -gs['count']  # Negative for sorting
    
    elif method == 'rank_average':
        # Sort by average rank only
        for gene, gs in gene_scores.items():
            gene_scores[gene]['score'] = np.mean(gs['ranks'])
    
    elif method == 'reciprocal_rank':
        # Use reciprocal rank sum (common in ensemble methods)
        n_runs = len(predictions)
        for gene in gene_scores:
            if gene_scores[gene]['count'] < n_runs:
                gene_scores[gene]['ranks'] = gene_scores[gene]['ranks'] + [top_k+1] * (n_runs - gene_scores[gene]['count'])
        
        for gene, gs in gene_scores.items():
            rr_sum = sum(1.0 / rank if rank < top_k+1 else 0 for rank in gs['ranks'])
            gene_scores[gene]['score'] = -rr_sum  # Negative for sorting
            #print('gsranks',len(gs['ranks']), gs['ranks'], -rr_sum)
    
    elif method == 'intersection_only':
        # Option 2: Only ensemble genes that appear in at least min_appearances runs
        # This focuses on genes with consistent presence across runs
        filtered_genes = {gene: gs for gene, gs in gene_scores.items() 
                         if gs['count'] >= min_appearances}
        
        #if len(filtered_genes) < top_k:
        #    print(f"  Warning: Only {len(filtered_genes)} genes appear in >={min_appearances} runs, "
        #          f"less than top_k={top_k}. Using all filtered genes.", flush=True)
        
        # Score by average rank among the runs where it appears
        for gene, gs in filtered_genes.items():
            avg_rank = np.mean(gs['ranks'])
            freq = gs['count']
            # Combine: lower rank is better, higher frequency is better
            # Weight frequency more heavily for intersection method
            filtered_genes[gene]['score'] = avg_rank / (freq ** 2)
        
        # Sort and return
        sorted_genes = sorted(filtered_genes.items(), key=lambda x: x[1]['score'])
        ensembled_genes = [gene for gene, _ in sorted_genes[:top_k]]
        
        return ensembled_genes
    
    elif method == 'topk_threshold':
        # Option 4: Only consider genes that ranked in top-N of at least one run
        # This focuses on genes that were highly ranked by at least one model
        filtered_genes = {gene: gs for gene, gs in gene_scores.items()
                         if min(gs['ranks']) <= topk_threshold}
        
        #if len(filtered_genes) < top_k:
        #    print(f"  Warning: Only {len(filtered_genes)} genes ranked in top-{topk_threshold}, "
        #          f"less than top_k={top_k}. Using all filtered genes.", flush=True)
        
        n_runs = len(predictions)
        for gene in gene_scores:
            if gene_scores[gene]['count'] < n_runs:
                gene_scores[gene]['ranks'] = gene_scores[gene]['ranks'] + [top_k+1] * (n_runs - gene_scores[gene]['count'])
        
        # Use reciprocal rank scoring for genes that pass the threshold
        for gene, gs in filtered_genes.items():
            rr_sum = sum(1.0 / rank if rank < top_k+1 else 0 for rank in gs['ranks'])
            filtered_genes[gene]['score'] = -rr_sum  # Negative for sorting (higher RR is better)
            #print('gsranks',len(gs['ranks']), gs['ranks'], -rr_sum)

        # Sort and return
        sorted_genes = sorted(filtered_genes.items(), key=lambda x: x[1]['score'])
        ensembled_genes = [gene for gene, _ in sorted_genes[:top_k]]
        
        return ensembled_genes
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # Sort genes by score
    sorted_genes = sorted(gene_scores.items(), key=lambda x: x[1]['score'])
    ensembled_genes = [gene for gene, _ in sorted_genes[:top_k]]
    
    return ensembled_genes


def save_predictions_cache(predictions_data: Dict[str, Any], cache_file: Path):
    """
    Save predictions to cache file.
    
    Args:
        predictions_data: Dictionary containing all predictions and metadata
        cache_file: Path to cache file
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    print(f"\nSaved predictions cache to {cache_file}", flush=True)


def load_predictions_cache(cache_file: Path) -> Dict[str, Any]:
    """
    Load predictions from cache file.
    
    Args:
        cache_file: Path to cache file
        
    Returns:
        Dictionary containing all predictions and metadata
    """
    with open(cache_file, 'r') as f:
        predictions_data = json.load(f)
    print(f"\nLoaded predictions cache from {cache_file}", flush=True)
    return predictions_data


def evaluate_ensemble_model(
    model: RankingModule,
    test_examples: List[Example],
    metrics_evaluator: RankingMetrics,
    n_runs: int = 5,
    ensemble_method: str = 'rank_frequency',
    top_k: int = 100,
    min_appearances: int = 2,
    topk_threshold: int = 20,
    verbose: bool = True,
    cache_file: Path = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Evaluate ensemble model on test set.
    
    Args:
        model: Ranking module
        test_examples: Test examples
        metrics_evaluator: Metrics evaluator
        n_runs: Number of runs per question for ensemble
        ensemble_method: Method for ensembling ('rank_frequency', 'frequency', 'rank_average', 
                        'reciprocal_rank', 'intersection_only', 'topk_threshold')
        top_k: Number of top genes to return in ensemble
        min_appearances: For 'intersection_only', minimum number of runs a gene must appear in (default: 2)
        topk_threshold: For 'topk_threshold', only consider genes ranked in top-N of at least one run (default: 20)
        verbose: Whether to print results
        cache_file: Path to cache file for saving/loading predictions
        use_cache: Whether to use cached predictions if available
        
    Returns:
        Dictionary with averaged metrics and per-example details
    """
    all_results = []
    per_example_details = []
    
    # Check if we should load from cache
    all_predictions = None
    cached_n_runs = 0
    extend_runs = False
    
    if use_cache and cache_file is not None and cache_file.exists():
        cached_data = load_predictions_cache(cache_file)
        cached_n_runs = cached_data.get('n_runs', 0)
        cached_num_examples = cached_data.get('num_examples', 0)
        
        # Check if cache is exactly what we need
        if cached_n_runs == n_runs and cached_num_examples == len(test_examples):
            all_predictions = cached_data['predictions']
            print(f"  Using cached predictions ({n_runs} runs, {len(test_examples)} examples)", flush=True)
        
        # Check if we can extend existing runs (same examples, but want more runs)
        elif cached_num_examples == len(test_examples) and cached_n_runs < n_runs:
            print(f"  Found cache with {cached_n_runs} runs, but you requested {n_runs} runs.", flush=True)
            print(f"  Will extend by generating {n_runs - cached_n_runs} additional runs...", flush=True)
            all_predictions = cached_data['predictions']
            extend_runs = True
        
        # Check if we can use subset of existing runs (want fewer runs than cached)
        elif cached_num_examples == len(test_examples) and cached_n_runs > n_runs:
            print(f"  Found cache with {cached_n_runs} runs, but you requested {n_runs} runs.", flush=True)
            print(f"  Will use first {n_runs} runs from cache...", flush=True)
            # Truncate predictions to use only first n_runs
            all_predictions = []
            for pred_data in cached_data['predictions']:
                all_predictions.append({
                    'question': pred_data['question'],
                    'predictions': pred_data['predictions'][:n_runs]
                })
            cached_n_runs = n_runs  # Update to reflect what we're using
            
        # Cache is incompatible
        else:
            print(f"  Cache exists but incompatible (cached: {cached_n_runs} runs, "
                  f"{cached_num_examples} examples; requested: {n_runs} runs, "
                  f"{len(test_examples)} examples)", flush=True)
            print(f"  Will generate new predictions...", flush=True)
    
    # Generate predictions if needed
    should_generate = (all_predictions is None) or extend_runs
    
    if should_generate:
        if extend_runs:
            # We're extending existing runs
            start_run = cached_n_runs
            runs_to_generate = n_runs - cached_n_runs
            print(f"\nGenerating {runs_to_generate} additional runs (runs {start_run+1} to {n_runs})...", flush=True)
        else:
            # We're generating all runs from scratch
            start_run = 0
            runs_to_generate = n_runs
            all_predictions = []
            print(f"\nGenerating predictions ({n_runs} runs per example)...", flush=True)
        
        for i, example in enumerate(test_examples):
            if verbose:
                if extend_runs:
                    print(f"\nProcessing example {i+1}/{len(test_examples)} (extending runs)...", flush=True)
                else:
                    print(f"\nProcessing example {i+1}/{len(test_examples)}...", flush=True)
            
            # Get existing predictions if extending
            if extend_runs:
                predictions = all_predictions[i]['predictions'].copy()
            else:
                predictions = []
            
            # Run model for new runs only
            for run_idx in range(start_run, n_runs):
                # Add a unique suffix to bypass DSPy's caching mechanism
                # This ensures each run generates a new prediction
                #question_with_suffix = example.question + f"\n\n[Run {run_idx+1}/{n_runs}]"
                prediction = model(question=example.question)
                output_text = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
                genes = parse_genes_from_output(output_text)
                predictions.append(genes)
                
                if verbose and (run_idx - start_run) < 5:  # Show first 5 new runs
                    print(f"  Run {run_idx+1}: {len(genes)} genes", flush=True)
                    #print(genes, flush=True)
            
            # Update or append predictions
            if extend_runs:
                all_predictions[i]['predictions'] = predictions
            else:
                all_predictions.append({
                    'question': example.question,
                    'predictions': predictions
                })
        
        # Save to cache if cache_file is provided
        if cache_file is not None:
            cache_data = {
                'n_runs': n_runs,
                'num_examples': len(test_examples),
                'predictions': all_predictions
            }
            save_predictions_cache(cache_data, cache_file)
            if extend_runs:
                print(f"  Updated cache with {n_runs} total runs (added {runs_to_generate} new runs)", flush=True)
    
    # First, evaluate individual runs to get per-run metrics
    print(f"\nEvaluating individual runs...", flush=True)
    run_results = [[] for _ in range(n_runs)]  # One list per run
    
    for i, example in enumerate(test_examples):
        predictions = all_predictions[i]['predictions']
        
        if verbose and i < 5:
            print(f"\nExample {i+1} - Individual run metrics:", flush=True)
        
        # Evaluate each individual run
        for run_idx in range(n_runs):
            results = metrics_evaluator.evaluate(
                predicted_genes=predictions[run_idx],
                ground_truth_genes=example.genes,
                relevance_scores=example.relevance_scores
            )
            run_results[run_idx].append(results)
            
            # Print adjusted_ndcg@100 for each run (first few examples)
            if verbose and i < 5:
                print(f"  Run {run_idx+1}: ndcg@100 = {results.get('ndcg@100', 0.0):.4f}", flush=True)
    
    # Calculate and print average metrics per run
    print("\n" + "="*60, flush=True)
    print("INDIVIDUAL RUN METRICS (averaged across all examples):", flush=True)
    print("="*60, flush=True)
    
    run_averages = []
    for run_idx in range(n_runs):
        avg_results = {}
        metric_keys = run_results[run_idx][0].keys()
        
        for key in metric_keys:
            if key not in ['num_predicted', 'num_genes', 'predicted_genes', 'ground_truth_genes', 'predicted_values']:
                values = [r[key] for r in run_results[run_idx] if key in r]
                if values:
                    avg_results[key] = np.mean(values)
        
        run_averages.append(avg_results)
        
        print(f"\nRun {run_idx+1}:", flush=True)
        for key in ['adjusted_ndcg@100', 'ndcg@100', 'mrr']:
            if key in avg_results:
                print(f"  {key}: {avg_results[key]:.4f}", flush=True)
    
    # Show that runs are different by displaying range
    print("\n" + "-"*60, flush=True)
    print("VERIFICATION: Runs produce different results", flush=True)
    print("-"*60, flush=True)
    for key in ['adjusted_ndcg@100', 'ndcg@100', 'mrr']:
        values = [run_avg[key] for run_avg in run_averages if key in run_avg]
        if values:
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val
            print(f"{key}: range = {range_val:.4f} (min={min_val:.4f}, max={max_val:.4f})", flush=True)
    
    # Calculate average and std across runs for key metrics
    print("\n" + "-"*60, flush=True)
    print("STATISTICS ACROSS RUNS:", flush=True)
    print("-"*60, flush=True)
    for key in ['adjusted_ndcg@100', 'ndcg@100', 'mrr']:
        values = [run_avg[key] for run_avg in run_averages if key in run_avg]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}", flush=True)
    
    # Now apply ensemble method and evaluate
    print("\n" + "="*60, flush=True)
    print(f"ENSEMBLE RESULTS (method='{ensemble_method}'):", flush=True)
    print("="*60, flush=True)
    
    for i, example in enumerate(test_examples):
        predictions = all_predictions[i]['predictions']
        
        if verbose and i < 5:
            print(f"\nEvaluating example {i+1}/{len(test_examples)}...", flush=True)
        
        # Ensemble predictions
        ensembled_genes = ensemble_predictions(
            predictions,
            method=ensemble_method,
            top_k=top_k,
            min_appearances=min_appearances,
            topk_threshold=topk_threshold
        )
        
        if verbose and i < 5:
            print(f"  Ensembled: {len(ensembled_genes)} genes", flush=True)
        
        # Evaluate ensemble
        results = metrics_evaluator.evaluate(
            predicted_genes=ensembled_genes,
            ground_truth_genes=example.genes,
            relevance_scores=example.relevance_scores
        )
        
        all_results.append(results)
        
        # Store details
        per_example_details.append({
            'question': example.question[:100],  # Truncate for brevity
            'n_predictions': len(predictions),
            'ensembled_genes': ensembled_genes[:10],  # Top 10
            'metrics': results
        })
        
        if verbose and i < 5:  # Show details for first 5 examples
            print(f"  Dataset: {example.dataset_name}", flush=True)
            print(f"  nDCG@100: {results.get('ndcg@100', 0.0):.4f}", flush=True)
            print(f"  adjusted_nDCG@100: {results.get('adjusted_ndcg@100', 0.0):.4f}", flush=True)
            print(f"  MRR: {results.get('mrr', 0.0):.4f}", flush=True)
    
    # Average all metrics
    avg_results = {}
    metric_keys = all_results[0].keys()
    
    for key in metric_keys:
        if key not in ['num_predicted', 'num_genes', 'predicted_genes', 'ground_truth_genes', 'predicted_values']:
            values = [r[key] for r in all_results if key in r]
            if values:
                avg_results[key] = np.mean(values)
    
    if verbose:
        print("\n" + "="*60, flush=True)
        print("AVERAGE ENSEMBLE RESULTS:", flush=True)
        print("="*60, flush=True)
        for key, value in avg_results.items():
            print(f"  {key}: {value:.4f}", flush=True)
    
    # Compare ensemble to individual runs
    if verbose:
        print("\n" + "="*60, flush=True)
        print("ENSEMBLE vs INDIVIDUAL RUNS:", flush=True)
        print("="*60, flush=True)
        for key in ['adjusted_ndcg@100', 'ndcg@100', 'mrr']:
            if key in avg_results:
                run_values = [run_avg[key] for run_avg in run_averages if key in run_avg]
                if run_values:
                    best_run = max(run_values)
                    mean_run = np.mean(run_values)
                    ensemble_val = avg_results[key]
                    improvement_vs_mean = ensemble_val - mean_run
                    improvement_vs_best = ensemble_val - best_run
                    print(f"{key}:", flush=True)
                    print(f"  Best individual run:  {best_run:.4f}", flush=True)
                    print(f"  Mean of runs:         {mean_run:.4f}", flush=True)
                    print(f"  Ensemble:             {ensemble_val:.4f} "
                          f"({improvement_vs_mean:+.4f} vs mean, "
                          f"{improvement_vs_best:+.4f} vs best)", flush=True)
    
    return {
        'average': avg_results,
        'per_example': per_example_details,
        'all_results': all_results,
        'individual_runs': run_averages
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="ensemble-baseline")
def main(cfg: DictConfig):
    """
    Main function for ensemble baseline evaluation.
    
    Args:
        cfg: Hydra configuration
    """
    from promptopt.utils.gnesys_wrapper import GNEsysPredictor, GNEsysLM

    from sklearn.model_selection import train_test_split
    print("="*60, flush=True)
    print("ENSEMBLE BASELINE EVALUATION FOR GENE RANKING", flush=True)
    print("="*60, flush=True)
    
    # Initialize Langfuse (optional monitoring)
    try:
        from langfuse import Langfuse
        import httpx
        import urllib3

        # Suppress SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Create httpx client with SSL verification disabled
        httpx_client = httpx.Client(verify=False)
        langfuse = Langfuse(httpx_client=httpx_client)

        if langfuse.auth_check():
            print("Langfuse client is authenticated and ready!")
            
            from openinference.instrumentation.dspy import DSPyInstrumentor
            DSPyInstrumentor().instrument()
        else:
            print("Langfuse authentication failed. Continuing without monitoring.")
    except Exception as e:
        print(f"Could not initialize Langfuse: {e}")
        print("Continuing without monitoring.")
    
    # Set random seed
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Get ensemble configuration
    n_runs = cfg.ensemble.get('n_runs', 5)
    ensemble_method = cfg.ensemble.get('method', 'rank_frequency')
    top_k = cfg.ensemble.get('top_k', 100)
    min_appearances = cfg.ensemble.get('min_appearances', 2)
    topk_threshold = cfg.ensemble.get('topk_threshold', 20)
    use_cache = cfg.ensemble.get('use_cache', True)
    cache_filename = cfg.ensemble.get('cache_filename', 'predictions_cache.json')
    
    print(f"\nEnsemble Configuration:", flush=True)
    print(f"  Number of runs per question: {n_runs}", flush=True)
    print(f"  Ensemble method: {ensemble_method}", flush=True)
    print(f"  Top K genes: {top_k}", flush=True)
    if ensemble_method == 'intersection_only':
        print(f"  Min appearances: {min_appearances}", flush=True)
    if ensemble_method == 'topk_threshold':
        print(f"  Top-K threshold: {topk_threshold}", flush=True)
    print(f"  Use cache: {use_cache}", flush=True)
    print(f"  Cache filename: {cache_filename}", flush=True)
    
    # Initialize LM based on configuration
    print(f"\nInitializing LM...", flush=True)
    
    if hasattr(cfg, 'gnesys'):
        # Initialize GNEsys as the LM
        print("\nInitializing GNEsys as custom DSPy LM...", flush=True)
        gnesys_predictor = GNEsysPredictor.from_config_path(
            config_path=cfg.gnesys.config_path,
            config_name=cfg.gnesys.config_name,
            overrides=cfg.gnesys.overrides,
            verbose=cfg.gnesys.verbose
        )
        
        # Configure kernel pooling to prevent memory leaks
        reuse_kernels = cfg.gnesys.get('reuse_kernels', True)
        max_kernels = cfg.gnesys.get('max_kernels', 5)
        
        lm = GNEsysLM(
            gnesys_predictor=gnesys_predictor,
            reuse_kernels=reuse_kernels,
            max_kernels=max_kernels
        )
        print(f"  GNEsys LM initialized successfully", flush=True)
        print(f"  Kernel pooling: {'enabled' if reuse_kernels else 'disabled'} (max_kernels={max_kernels})", flush=True)

        # Extract init_prompt from GNEsys to use in RankingSignature
        init_prompt = gnesys_predictor.get_init_prompt()
        print(f"  Extracted init_prompt ({len(init_prompt)} characters)", flush=True)
        
        tool_prompt = gnesys_predictor.get_tool_prompt()
        print(f"  Tool prompt will be appended by GNEsys ({len(tool_prompt)} characters)", flush=True)
        
        # Create RankingSignature with the GNEsys init_prompt
        RankingSignatureClass = create_ranking_signature(init_prompt)
        
    else:
        # Use standard DSPy LM
        provider = cfg.dspy_lm.get('provider', 'openai')
        
        if provider == 'azure':
            import os
            api_key = os.environ.get('AZURE_API_KEY')
            api_base = os.environ.get('AZURE_API_BASE')
            
            if not api_key or not api_base:
                raise ValueError(
                    "Azure OpenAI requires AZURE_API_KEY and AZURE_API_BASE environment variables."
                )
            
            print(f"  Using Azure OpenAI endpoint: {api_base}", flush=True)
            print(f"  Deployment: {cfg.dspy_lm.model}", flush=True)
            
            lm = dspy.LM(
                model=f"azure/{cfg.dspy_lm.model}",
                max_tokens=cfg.dspy_lm.max_tokens,
                temperature=cfg.dspy_lm.temperature,
            )
        else:
            # Standard OpenAI configuration
            lm = dspy.LM(
                model=cfg.dspy_lm.model,
                max_tokens=cfg.dspy_lm.max_tokens,
                temperature=cfg.dspy_lm.temperature
            )
        
        RankingSignatureClass = create_ranking_signature()
    
    # Configure DSPy to use the LM
    dspy.configure(lm=lm)

    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
    )

    
    # Load dataset
    print(f"\nLoading dataset...", flush=True)
    dataset = ScreensQADSPY(
        mounted_dir=cfg.dataset.mounted_dir,
        cache_directory=cfg.dataset.cache_directory,
        split_type=cfg.dataset.split_type,
        fold=cfg.dataset.fold,
        max_examples=cfg.dataset.max_examples,
        relevance_score_name=cfg.dataset.relevance_score_name,
        prompt_template_key=cfg.dataset.prompt_template_key,
    )
    
    train_examples, val_examples, test_examples = dataset.get_train_test_split()
    
    # Convert to DSPy format
    test_dspy = create_dspy_examples(test_examples)
    
    print(f"Test set: {len(test_dspy)} examples", flush=True)
    print()
    
    # Create ranking module
    print("\nCreating ranking module...", flush=True)
    model = RankingModule(signature_class=RankingSignatureClass)
    
    # Create metrics evaluator
    print("\nCreating metrics evaluator...", flush=True)
    metrics_evaluator = RankingMetrics(
        k_values=cfg.evaluation.k_values,
        use_thresholded_scoring=cfg.evaluation.use_thresholded_scoring
    )
    
    # Evaluate ensemble baseline on test set
    print("\n" + "="*60, flush=True)
    print("EVALUATING ENSEMBLE BASELINE ON TEST SET", flush=True)
    print("="*60, flush=True)
    
    # Setup cache file path
    output_dir = Path(cfg.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / cache_filename
    
    ensemble_results = evaluate_ensemble_model(
        model=model,
        test_examples=test_dspy,
        metrics_evaluator=metrics_evaluator,
        n_runs=n_runs,
        ensemble_method=ensemble_method,
        top_k=top_k,
        min_appearances=min_appearances,
        topk_threshold=topk_threshold,
        verbose=True,
        cache_file=cache_file,
        use_cache=use_cache
    )
    
    # Save results
    results_file = output_dir / "ensemble_baseline_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'ensemble_config': {
                'n_runs': n_runs,
                'method': ensemble_method,
                'top_k': top_k
            },
            'individual_runs': {
                'per_run_averages': ensemble_results['individual_runs'],
                'statistics': {
                    key: {
                        'mean': float(np.mean([run[key] for run in ensemble_results['individual_runs'] if key in run])),
                        'std': float(np.std([run[key] for run in ensemble_results['individual_runs'] if key in run])),
                        'min': float(np.min([run[key] for run in ensemble_results['individual_runs'] if key in run])),
                        'max': float(np.max([run[key] for run in ensemble_results['individual_runs'] if key in run]))
                    }
                    for key in ['adjusted_ndcg@100', 'ndcg@100', 'mrr', 'adjusted_ndcg@5', 'adjusted_ndcg@10']
                    if any(key in run for run in ensemble_results['individual_runs'])
                }
            },
            'ensemble_results': {
                'average_metrics': ensemble_results['average'],
                'num_examples': len(test_dspy)
            },
            'per_example_details': ensemble_results['per_example'],
            'config': OmegaConf.to_container(cfg, resolve=True)
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("EVALUATION COMPLETE!", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()

