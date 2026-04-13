"""
Main script for running GEPA prompt optimization on the ScreenBench ranking task.

This script:
1. Loads the ScreenBench ranking dataset
2. Wraps GNEsys in a DSPy-compatible module
3. Runs GEPA optimization to improve prompts
4. Evaluates the optimized model on test data
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
from typing import List, Dict, Any
import numpy as np

import dspy
from dspy import GEPA, Example

from screensqa.dataset.dataset import ScreensQADSPY
from screensqa.benchmark.ranking_metrics import RankingMetrics
from promptopt.utils.gnesys_wrapper import GNEsysPredictor, GNEsysLM
from screensqa.dataset.dataset import ScreensQADSPY
from screensqa.benchmark.ranking_metrics import RankingMetrics

from sklearn.model_selection import train_test_split

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
    
    Args:
        dataset_examples: List of examples from ScreenRankingDataset
        
    Returns:
        List of dspy.Example objects
    """
    dspy_examples = []
    
    for ex in dataset_examples:
        # Create DSPy example with question and ground truth answer
        dspy_ex = dspy.Example(
            question=ex['question'],
            answer=ex['answer']
        ).with_inputs("question")
        
        # Store additional metadata for evaluation
        dspy_ex.genes = ex['genes']
        dspy_ex.relevance_scores = ex['relevance_scores']
        dspy_ex.alpha = ex['alpha']
        dspy_ex.ranking_method = ex['ranking_method']
        dspy_ex.dataset_name = ex['dataset_name']
        dspy_ex.description = ex['description']
        dspy_ex.phenotype = ex['phenotype']
        dspy_ex.num_genes = ex['num_genes']
        dspy_ex.split = ex['split']
        dspy_ex.reverse = ex['reverse']

        
        dspy_examples.append(dspy_ex)
    
    return dspy_examples


def create_ranking_metric(
    k_values: List[int] = None,
    use_thresholded_scoring: bool = True,
    require_tool_calls: bool = False,
    provide_feedback: bool = False,
    gnesys_lm = None
):
    """
    Create a DSPy-compatible metric function for ranking evaluation.
    
    Args:
        k_values: List of k values for metrics
        use_thresholded_scoring: Whether to use thresholded scoring
        require_tool_calls: If True, return score=0 when no tools are used
        provide_feedback: If True, return feedback text along with score
        gnesys_lm: GNEsysLM instance to access tool_calls information
        
    Returns:
        Metric function compatible with DSPy
    """
    metrics_evaluator = RankingMetrics(
        k_values=k_values,
        use_thresholded_scoring=use_thresholded_scoring
    )
    
    def ranking_metric(
        gold: Example, 
        pred: dspy.Prediction, 
        trace=None, 
        pred_name=None, 
        pred_trace=None
    ):
        """
        Evaluate ranking prediction.
        
        GEPA-compatible metric that accepts 5 arguments:
        - gold: The gold/ground truth example
        - pred: The prediction
        - trace: Optional full execution trace
        - pred_name: Optional name of the predictor being optimized
        - pred_trace: Optional trace of the specific predictor
        
        Returns adjusted_nDCG@100 as the primary metric for optimization.
        If provide_feedback=True, returns dspy.Prediction(score=..., feedback=...)
        Otherwise returns float score.
        """
        # Check tool usage if required (before extracting answer text)
        tool_calls = 0
        if require_tool_calls and gnesys_lm is not None:
            # Look up tool calls using the Prediction object's reasoning field
            # This is thread-safe because we hash the reasoning in GNEsysLM.__call__
            tool_calls = gnesys_lm.get_tool_calls_for_prediction(pred)
            print('tool_calls', tool_calls, flush=True)

            if tool_calls == 0:
                # No tools used - penalize with score of 0
                if provide_feedback:
                    feedback = (
                        "You did not use any tools to answer this question, so you received a score of 0.0. "
                        "For gene ranking tasks, you should use available databases and tools "
                        "to gather evidence about genes relevant to the described phenotype.\n\n"
                        "Please use tools to gather evidence before making predictions. "
                        "Ground your answer in data from the available tools and databases."
                    )
                    return dspy.Prediction(score=0.0, feedback=feedback)
                else:
                    return 0.0
        
        # Extract predicted genes from output for evaluation
        output_text = pred.answer if hasattr(pred, 'answer') else str(pred)
        
        # Evaluate
        results = metrics_evaluator.evaluate_from_output(
            output=output_text,
            ground_truth_genes=gold.genes,
            relevance_scores=gold.relevance_scores
        )
        
        # Get primary score (nDCG@100)
        score = results.get('adjusted_ndcg@100', 0.0)
        
        # Get tool_calls for feedback (if not already retrieved above)
        if provide_feedback and tool_calls == 0 and gnesys_lm is not None:
            tool_calls = gnesys_lm.get_tool_calls_for_prediction(pred)
        
        # Return with or without feedback
        if provide_feedback:
            # Generate feedback based on performance
            if score >= 0.8:
                feedback = (
                    f"Excellent work! Your ranking achieved an adjusted nDCG@100 of {score:.4f}. "
                    f"You successfully used {tool_calls} tool call(s) to gather relevant information. "
                    "Your approach of using tools to ground predictions in data is working well. "
                    "Keep using multiple sources of evidence for comprehensive answers."
                )
            elif score >= 0.5:
                feedback = (
                    f"Good progress! Your ranking achieved an adjusted nDCG@100 of {score:.4f}. "
                    f"You used {tool_calls} tool call(s), which shows you're gathering evidence. "
                    "To improve further, consider:\n"
                    "- Using more diverse data sources\n"
                    "- Prioritizing genes with stronger evidence from multiple tools\n"
                    "- Combining information from different databases for more comprehensive rankings"
                )
            elif score >= 0.2:
                feedback = (
                    f"Your ranking achieved an adjusted nDCG@100 of {score:.4f}. "
                    f"While you used {tool_calls} tool call(s), the predictions could be more accurate. "
                    "Consider:\n"
                    "- Using more specific queries to find relevant genes\n"
                    "- Cross-referencing information from multiple tools\n"
                    "- Paying attention to statistical significance (p-values) when ranking genes\n"
                    "- Using the phenotype description to guide your tool queries"
                )
            else:
                feedback = (
                    f"Your ranking achieved an adjusted nDCG@100 of {score:.4f}. "
                    f"You used {tool_calls} tool call(s), but the predictions need significant improvement. "
                    "Key recommendations:\n"
                    "- Carefully analyze the phenotype description to identify relevant biological processes\n"
                    "- Use tools to search for genes involved in those processes\n"
                    "- Prioritize genes with strong statistical evidence (low p-values, high effect sizes)\n"
                    "- Consider using pathway analysis to find functionally related genes\n"
                    "- Always verify your findings with multiple data sources before ranking"
                )
            
            # Add information about other metrics
            feedback += f"\n\nAdditional metrics: MRR={results.get('mrr', 0.0):.4f}, "
            feedback += f"adjusted_nDCG@5={results.get('adjusted_ndcg@5', 0.0):.4f}, "
            feedback += f"adjustednDCG@10={results.get('adjusted_ndcg@10', 0.0):.4f}, "
            feedback += f"nDCG@5={results.get('ndcg@5', 0.0):.4f}, "
            feedback += f"nDCG@10={results.get('ndcg@10', 0.0):.4f}, "
            feedback += f"nDCG@100={results.get('ndcg@100', 0.0):.4f}"
            
            return dspy.Prediction(score=score, feedback=feedback)
        else:
            return score
    
    return ranking_metric


def evaluate_model(
    model: RankingModule,
    test_examples: List[Example],
    metrics_evaluator: RankingMetrics,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained ranking module
        test_examples: Test examples
        metrics_evaluator: Metrics evaluator
        verbose: Whether to print results
        
    Returns:
        Dictionary of averaged metrics
    """
    all_results = []
    
    for i, example in enumerate(test_examples):
        # Make prediction
        prediction = model(question=example.question)
        
        # Extract output text
        output_text = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
        
        # Evaluate
        results = metrics_evaluator.evaluate_from_output(
            output=output_text,
            ground_truth_genes=example.genes,
            relevance_scores=example.relevance_scores
        )
        
        all_results.append(results)
        
        if verbose and i < 20:
            print(f"\nExample {i+1}:", flush=True)
            print(f"  Dataset: {example.genes[:5]}... ({len(example.genes)} genes)", flush=True)
            print(f"  nDCG@100: {results.get('ndcg@100', 0.0):.4f}", flush=True)
            print(f"  adjusted_nDCG@100: {results.get('adjusted_ndcg@100', 0.0):.4f}", flush=True)
            print(f"  MRR: {results.get('mrr', 0.0):.4f}", flush=True)
            print(f"  Number of Genes Predicted: {results.get('num_predicted', 0.0):.4f}", flush=True)
            print(f"  Predicted Genes: {results.get('predicted_genes', [])}", flush=True)
            print(f"  Predicted Relevances: {results.get('predicted_values', [])}", flush=True)
            print(f"  Ground Truth Genes: {results.get('ground_truth_genes', [])[:100]}", flush=True)
    
    # Average all metrics
    avg_results = {}
    metric_keys = all_results[0].keys()
    print('metric_keys', metric_keys, flush=True)
    
    for key in metric_keys:
        if key not in ['num_predicted', 'num_genes', 'predicted_genes', 'ground_truth_genes', 'predicted_values']:
            values = [r[key] for r in all_results if key in r]
            if values:
                avg_results[key] = np.mean(values)
    
    if verbose:
        print("\n" + "="*60, flush=True)
        print("AVERAGE TEST RESULTS:", flush=True)
        print("="*60, flush=True)
        for key, value in avg_results.items():
            print(f"  {key}: {value:.4f}", flush=True)
    
    return avg_results


@hydra.main(version_base="1.3", config_path="../configs/gepa", config_name="gemini-3-flash")
def main(cfg: DictConfig):
    """
    Main function for GEPA optimization.
    
    Args:
        cfg: Hydra configuration
    """
    print("="*60, flush=True)
    print("GEPA PROMPT OPTIMIZATION FOR GENE RANKING", flush=True)
    print("="*60, flush=True)
    
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
    else:
        print("Authentication failed. Please check your credentials and host.")
        
    ### Connect Langfuse

    from openinference.instrumentation.dspy import DSPyInstrumentor

    DSPyInstrumentor().instrument()

    # Set random seed
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Initialize DSPy with the LM for GEPA reflection
    print(f"\nInitializing GEPA reflection LM with {cfg.dspy_lm.model}...", flush=True)
    
    # Configure for Azure or standard OpenAI
    provider = cfg.dspy_lm.get('provider', 'openai')
    
    if provider == 'azure':
        import os
        # Azure OpenAI configuration
        # DSPy automatically uses AZURE_API_KEY, AZURE_API_BASE, and AZURE_API_VERSION from environment
        api_key = os.environ.get('AZURE_API_KEY')
        api_base = os.environ.get('AZURE_API_BASE')
        
        if not api_key or not api_base:
            raise ValueError(
                "Azure OpenAI requires AZURE_API_KEY and AZURE_API_BASE environment variables.\n"
                "Set them in your .env file or export them:\n"
                "  export AZURE_API_KEY='your-key'\n"
                "  export AZURE_API_BASE='https://your-resource.openai.azure.com'\n"
                "  export AZURE_API_VERSION='2024-02-15-preview'  # Optional"
            )
        
        print(f"  Using Azure OpenAI endpoint: {api_base}", flush=True)
        print(f"  Deployment: {cfg.dspy_lm.model}", flush=True)
        
        # DSPy handles Azure configuration automatically via environment variables
        reflection_lm = dspy.LM(
            model=f"azure/{cfg.dspy_lm.model}",
            max_tokens=cfg.dspy_lm.max_tokens,
            temperature=cfg.dspy_lm.temperature,
            reasoning_effort=cfg.dspy_lm.reasoning_effort,
        )
    else:
        # Standard OpenAI configuration
        reflection_lm = dspy.LM(
            model=cfg.dspy_lm.model,
            max_tokens=cfg.dspy_lm.max_tokens,
            temperature=cfg.dspy_lm.temperature
        )
    
    if hasattr(cfg, 'genesys'):
        # Initialize gnesys as the base reflection LM
        print("\nInitializing GNEsys as custom DSPy LM...", flush=True)
        gnesys_predictor = GNEsysPredictor.from_config_path(
            config_path=cfg.gnesys.config_path,
            config_name=cfg.gnesys.config_name,
            overrides=cfg.gnesys.overrides,
            verbose=cfg.gnesys.verbose
        )
        # Configure kernel pooling to prevent memory leaks
        # By default, reuse up to 5 kernels instead of creating new ones for each prediction
        reuse_kernels = cfg.gnesys.get('reuse_kernels', True)
        max_kernels = cfg.gnesys.get('max_kernels', 5)
        
        gnesys_lm = GNEsysLM(
            gnesys_predictor=gnesys_predictor,
            reuse_kernels=reuse_kernels,
            max_kernels=max_kernels
        )
        print(f"  GNEsys LM initialized successfully", flush=True)
        print(f"  Kernel pooling: {'enabled' if reuse_kernels else 'disabled'} (max_kernels={max_kernels})", flush=True)

        # Extract init_prompt from GNEsys to use in RankingSignature
        # (tool_prompt will be appended by GNEsys automatically)
        init_prompt = gnesys_predictor.get_init_prompt()
        print(f"  Extracted init_prompt ({len(init_prompt)} characters)", flush=True)
        
        tool_prompt = gnesys_predictor.get_tool_prompt()
        print(f"  Tool prompt will be appended by GNEsys ({len(tool_prompt)} characters)", flush=True)
        
        # Create RankingSignature with the GNEsys init_prompt
        # Tool instructions will be appended by GNEsys when processing the system message
        RankingSignatureClass = create_ranking_signature(init_prompt)
        
    else:
        gnesys_lm = reflection_lm

        RankingSignatureClass = create_ranking_signature()

    # Configure DSPy to use GNEsys LM for the ranking module
    dspy.configure(lm=gnesys_lm)
    
    # Load dataset
    print(f"\nLoading dataset...", flush=True)
    dataset = ScreensQADSPY(
        mounted_dir=cfg.dataset.mounted_dir,
        cache_directory=cfg.dataset.cache_directory,
        split_type=cfg.dataset.split_type,
        fold=cfg.dataset.fold,
        max_examples=cfg.dataset.max_examples,  # Load all examples
        relevance_score_name=cfg.dataset.relevance_score_name,
        prompt_template_key=cfg.dataset.prompt_template_key,
    )

    #print(f"Dataset: {dataset}", flush=True)
    #print(dir(dataset), flush=True)
    #print(len(), flush=True)
    #zz
    

    train_examples, val_examples, test_examples = dataset.get_train_test_split()
    # Three-way split for larger datasets
    import random
    random.seed(cfg.dataset.seed)
    np.random.seed(cfg.dataset.seed)
    
    #print(len(train_examples), len(val_examples), len(test_examples), flush=True)
    #zz
    
    #train_examples, val_examples = train_test_split(train_examples, test_size=0.1, random_state=cfg.dataset.seed)
    
    # If test is empty, use val for test as well
    if len(test_examples) == 0:
        test_examples = val_examples
    
    # Convert to DSPy format
    train_dspy = create_dspy_examples(train_examples)
    val_dspy = create_dspy_examples(val_examples)
    test_dspy = create_dspy_examples(test_examples)
    
    print(f"Train set: {len(train_dspy)} examples (used for GEPA optimization)", flush=True)
    print(f"Val set: {len(val_dspy)} examples (used for GEPA Pareto frontier)", flush=True)
    print(f"Test set: {len(test_dspy)} examples (held out for final evaluation)", flush=True)
    print()
    
    # Create ranking module
    # The module uses ChainOfThought with the configured LM (GNEsys in this case)
    print("\nCreating ranking module with GNEsys backend...", flush=True)
    model = RankingModule(signature_class=RankingSignatureClass)
    
    # Create metric
    print("\nCreating ranking metric...", flush=True)
    metric = create_ranking_metric(
        k_values=cfg.evaluation.k_values,
        use_thresholded_scoring=cfg.evaluation.use_thresholded_scoring,
        require_tool_calls=cfg.evaluation.get('require_tool_calls', False),
        provide_feedback=cfg.evaluation.get('provide_feedback', False),
        gnesys_lm=gnesys_lm
    )
    
    # Evaluate random baseline
    print("\n" + "="*60, flush=True)
    print("RANDOM BASELINE EVALUATION", flush=True)
    print("="*60, flush=True)
    
    # Collect union of all genes across train, val, and test sets
    all_genes_set = set()
    for example in train_dspy + val_dspy + test_dspy:
        all_genes_set.update(example.genes)
    all_genes_list = list(all_genes_set)
    print(f"Total unique genes across all splits: {len(all_genes_list)}", flush=True)
    
    # Create random baseline: randomly select 100 genes
    import random
    def get_random_genes(n):
        random_genes = random.sample(all_genes_list, min(n, len(all_genes_list)))
        random_answer = ", ".join(random_genes)
        #print(f"Random baseline selects {len(random_genes)} genes")
        return random_genes
    

    #random_answer = get_random_genes(100)
    #print(f"Random baseline answer: {random_answer}")

    print("\nEvaluating random baseline on validation set...", flush=True)
    metrics_evaluator = RankingMetrics(
        k_values=cfg.evaluation.k_values,
        use_thresholded_scoring=cfg.evaluation.use_thresholded_scoring
    )
    
    random_val_results = []
    for i, example in enumerate(val_dspy):
        # Evaluate
        results = metrics_evaluator.evaluate(
            predicted_genes=get_random_genes(100),
            ground_truth_genes=example.genes,
            relevance_scores=example.relevance_scores
        )
        random_val_results.append(results)
    
        #print(example.genes)
        #print(example.rank_scores)
        #zz

    # Average results for val set
    random_val_avg = {}
    metric_keys = random_val_results[0].keys()
    for key in metric_keys:
        if key not in ['num_predicted', 'num_genes', 'predicted_genes', 'ground_truth_genes', 'predicted_values']:
            values = [r[key] for r in random_val_results if key in r]
            if values:
                random_val_avg[key] = np.mean(values)
    
    print("\nRandom Baseline - Validation Set Results:", flush=True)
    print("-" * 60, flush=True)
    for key, value in random_val_avg.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate random baseline on test set
    print("\nEvaluating random baseline on test set...", flush=True)
    random_test_results = []
    for i, example in enumerate(test_dspy):
        # Evaluate
        results = metrics_evaluator.evaluate(
            predicted_genes=get_random_genes(100),
            ground_truth_genes=example.genes,
            relevance_scores=example.relevance_scores
        )
        random_test_results.append(results)
    
    # Average results for test set
    random_test_avg = {}
    metric_keys = random_test_results[0].keys()
    for key in metric_keys:
        if key not in ['num_predicted', 'num_genes', 'predicted_genes', 'ground_truth_genes', 'predicted_values']:
            values = [r[key] for r in random_test_results if key in r]
            if values:
                random_test_avg[key] = np.mean(values)
    
    print("\nRandom Baseline - Test Set Results:", flush=True)
    print("-" * 60, flush=True)
    for key, value in random_test_avg.items():
        print(f"  {key}: {value:.4f}")
    print("="*60, flush=True)
    
    
    # Evaluate baseline on val set (before optimization)
    # Note: We use val set here to show improvement during optimization
    # The final test set evaluation comes later
    print("\n" + "="*60)
    print("BASELINE EVALUATION ON VAL SET (before optimization)")
    print("="*60, flush=True)
    metrics_evaluator = RankingMetrics(
        k_values=cfg.evaluation.k_values,
        use_thresholded_scoring=cfg.evaluation.use_thresholded_scoring
    )
    baseline_val_results = evaluate_model(model, val_dspy, metrics_evaluator, verbose=True)
    print(baseline_val_results, flush=True)
    #zz

    # Run GEPA optimization
    print("\n" + "="*60)
    print("RUNNING GEPA OPTIMIZATION")
    print("="*60)
    print(f"Max metric calls: {cfg.gepa.max_metric_calls}")
    print(f"Reflection minibatch size: {cfg.gepa.reflection_minibatch_size}")
    print(f"Max merge invocations: {cfg.gepa.max_merge_invocations}", flush=True)
    
    # Initialize GEPA with correct parameters from documentation
    # See: https://dspy.ai/api/optimizers/GEPA/overview/
    # Note: GEPA uses reflection_lm for optimization decisions,
    # while the model being optimized (RankingModule) uses GNEsys
    gepa_optimizer = GEPA(
        metric=metric,
        max_metric_calls=cfg.gepa.max_metric_calls,  # Budget for optimization
        reflection_minibatch_size=cfg.gepa.reflection_minibatch_size,  # Number of examples per reflection
        reflection_lm=reflection_lm,  # Use Azure LM for reflection/optimization
        track_stats=True,  # Track optimization statistics
        log_dir=cfg.output.save_dir,  # Save logs to output directory
        use_merge=cfg.gepa.use_merge,  # Enable candidate merging
        max_merge_invocations=cfg.gepa.max_merge_invocations,  # Limit merge operations
        seed=cfg.seed
    )
    
    # Optimize the model using GEPA
    # Following best practices from https://dspy.ai/tutorials/gepa_aime/
    # - trainset: Used for optimization (reflection and mutation)
    # - valset: Used for Pareto frontier evaluation
    # - test set is NOT passed to GEPA and is held out for final evaluation
    print("\nOptimizing with GEPA...")
    print(f"  Train set size: {len(train_dspy)}")
    print(f"  Val set size: {len(val_dspy)}", flush=True)
    optimized_model = gepa_optimizer.compile(
        model,
        trainset=train_dspy,
        valset=val_dspy
    )
    
    # Evaluate optimized model on val set (to show optimization improvement)
    print("\n" + "="*60)
    print("OPTIMIZED MODEL EVALUATION ON VAL SET")
    print("="*60, flush=True)
    optimized_val_results = evaluate_model(
        optimized_model,
        val_dspy,
        metrics_evaluator,
        verbose=True
    )
    
    # Compare val set results: Baseline vs Optimized
    print("\n" + "="*60)
    print("COMPARISON ON VAL SET: Baseline vs Optimized")
    print("="*60)
    for key in baseline_val_results.keys():
        baseline_val = baseline_val_results.get(key, 0.0)
        optimized_val = optimized_val_results.get(key, 0.0)
        improvement = optimized_val - baseline_val
        print(f"  {key:20s}: {baseline_val:.4f} → {optimized_val:.4f} ({improvement:+.4f})")
    
    # Final evaluation on held-out test set
    print("\n" + "="*60, flush=True)
    print("FINAL EVALUATION ON TEST SET (held out)")
    print("="*60)
    print("Evaluating baseline model on test set...")
    baseline_test_results = evaluate_model(model, test_dspy, metrics_evaluator, verbose=True)
    
    print("\nEvaluating optimized model on test set...")
    optimized_test_results = evaluate_model(optimized_model, test_dspy, metrics_evaluator, verbose=True)
    
    # Compare test set results: Baseline vs Optimized
    print("\n" + "="*60)
    print("COMPARISON ON TEST SET: Baseline vs Optimized")
    print("="*60)
    for key in baseline_test_results.keys():
        baseline_val = baseline_test_results.get(key, 0.0)
        optimized_val = optimized_test_results.get(key, 0.0)
        improvement = optimized_val - baseline_val
        print(f"  {key:20s}: {baseline_val:.4f} → {optimized_val:.4f} ({improvement:+.4f})")
    
    # Save results
    output_dir = Path(cfg.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "gepa_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'random_baseline': {
                'val_set': random_val_avg,
                'test_set': random_test_avg,
            },
            'val_set': {
                'baseline': baseline_val_results,
                'optimized': optimized_val_results,
            },
            'test_set': {
                'baseline': baseline_test_results,
                'optimized': optimized_test_results,
            },
            'config': OmegaConf.to_container(cfg, resolve=True)
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Save optimized model
    model_file = output_dir / "optimized_model.pkl"
    optimized_model.save(str(model_file))
    print(f"Optimized model saved to {model_file}")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()


