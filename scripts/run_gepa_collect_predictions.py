"""
GEPA prompt optimization + test evaluation script.

This script:
1. Loads the BioGRIDDSPY dataset (train / val / test splits)
2. Runs GEPA prompt optimization using train+val
3. Saves the optimized prompt and model
4. Evaluates the optimized model on the test set
5. Saves test predictions and reasonings in the same format as collect_llm_predictions.py

Usage:
  python scripts/run_gepa_collect_predictions.py \
      --config-path ../configs/gepa --config-name gemini-3-flash
"""

import rootutils
rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import dotenv
dotenv.load_dotenv('.env')

import os
import json
import hydra
import random
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any
import numpy as np
import logging

import dspy
from dspy import GEPA, Example

from screensqa.dataset.dataset import BioGRIDDSPY
from screensqa.benchmark.ranking_metrics import RankingMetrics

from scripts.collect_llm_predictions import (
    create_ranking_signature,
    RankingModule,
    create_dspy_examples,
    collect_predictions,
    BiomniLM,
)
from promptopt.utils.gnesys_wrapper import GNEsysPredictor, GNEsysLM

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARNING)


def create_ranking_metric(
    k_values: List[int] = None,
    use_thresholded_scoring: bool = True,
):
    """Create a GEPA-compatible metric that returns adjusted_nDCG@100."""
    metrics_evaluator = RankingMetrics(
        k_values=k_values,
        use_thresholded_scoring=use_thresholded_scoring,
    )

    def ranking_metric(
        gold: Example,
        pred: dspy.Prediction,
        trace=None,
        pred_name=None,
        pred_trace=None,
    ):
        output_text = pred.answer if hasattr(pred, 'answer') else str(pred)
        results = metrics_evaluator.evaluate_from_output(
            output=output_text,
            ground_truth_genes=gold.genes,
            relevance_scores=gold.relevance_scores,
        )
        return results.get('adjusted_ndcg@100', 0.0)

    return ranking_metric


def extract_optimized_prompt(optimized_model: RankingModule) -> str:
    """Extract the optimized prompt text from the model's predictor signature.

    ChainOfThought wraps a Predict module at `self.predict`, which holds the
    extended signature (with the reasoning field prepended).
    """
    predict = optimized_model.predictor.predict
    sig = predict.signature
    return sig.__doc__ or ""


def plot_gepa_results(detailed_results, output_dir: Path):
    """Generate visualizations from GEPA's detailed_results (DspyGEPAResult).

    Produces:
      1. Optimization curve: best val score vs metric calls consumed
      2. Per-candidate bar chart of aggregate val scores
      3. Heatmap of per-val-instance best candidate assignments
      4. Candidate lineage / family tree
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = detailed_results.val_aggregate_scores
    eval_counts = detailed_results.discovery_eval_counts
    subscores = detailed_results.val_subscores
    parents = detailed_results.parents
    n_candidates = len(scores)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # ── 1. Optimization curve ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    running_best = []
    best_so_far = -float('inf')
    for s in scores:
        best_so_far = max(best_so_far, s)
        running_best.append(best_so_far)

    ax.plot(eval_counts, running_best, 'o-', color='#2196F3',
            markersize=4, linewidth=1.8, label='Best val score so far')
    ax.scatter(eval_counts, scores, color='#90CAF9', s=20, alpha=0.6,
               zorder=2, label='Individual candidate')

    best_idx = int(np.argmax(scores))
    ax.scatter([eval_counts[best_idx]], [scores[best_idx]],
               color='#F44336', s=80, zorder=5, marker='*',
               label=f'Best (candidate {best_idx}, {scores[best_idx]:.4f})')

    ax.set_xlabel('Metric calls consumed', fontsize=11)
    ax.set_ylabel('Val aggregate score', fontsize=11)
    ax.set_title('GEPA Optimization Curve', fontsize=13, fontweight='semibold')
    ax.legend(fontsize=9, frameon=True, framealpha=0.9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / 'gepa_optimization_curve.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved gepa_optimization_curve.png")

    # ── 2. Per-candidate bar chart ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_candidates * 0.6), 5))
    colors = ['#F44336' if i == best_idx else '#64B5F6' for i in range(n_candidates)]
    ax.bar(range(n_candidates), scores, color=colors, edgecolor='white',
           linewidth=0.5, alpha=0.85)
    for i, s in enumerate(scores):
        ax.text(i, s + 0.002, f'{s:.3f}', ha='center', va='bottom',
                fontsize=7, rotation=90, color='#444')
    ax.set_xlabel('Candidate index', fontsize=11)
    ax.set_ylabel('Val aggregate score', fontsize=11)
    ax.set_title('Per-Candidate Val Scores', fontsize=13, fontweight='semibold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / 'gepa_candidate_scores.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved gepa_candidate_scores.png")

    # ── 3. Per-val-instance score heatmap ───────────────────────────────
    if subscores and len(subscores) > 0:
        score_matrix = np.array(subscores)  # (n_candidates, n_val_instances)
        n_val = score_matrix.shape[1]

        fig, ax = plt.subplots(figsize=(max(10, n_val * 0.06), max(4, n_candidates * 0.4)))
        im = ax.imshow(score_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel(f'Val instance (n={n_val})', fontsize=10)
        ax.set_ylabel('Candidate', fontsize=10)
        ax.set_title('Per-Instance Scores by Candidate', fontsize=13, fontweight='semibold')
        ax.set_yticks(range(n_candidates))
        fig.colorbar(im, ax=ax, label='Score', shrink=0.8)
        fig.tight_layout()
        fig.savefig(output_dir / 'gepa_score_heatmap.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved gepa_score_heatmap.png")

    # ── 4. Candidate lineage ────────────────────────────────────────────
    if parents and len(parents) > 1:
        fig, ax = plt.subplots(figsize=(max(8, n_candidates * 0.6), 5))
        for child_idx, parent_list in enumerate(parents):
            if parent_list is None:
                continue
            for p in parent_list:
                if p is not None:
                    ax.annotate(
                        '', xy=(child_idx, scores[child_idx]),
                        xytext=(p, scores[p]),
                        arrowprops=dict(arrowstyle='->', color='#999', lw=0.8, alpha=0.6),
                    )
        ax.scatter(range(n_candidates), scores, c=eval_counts, cmap='viridis',
                   s=60, zorder=5, edgecolors='white', linewidths=0.5)
        ax.scatter([best_idx], [scores[best_idx]], color='#F44336',
                   s=120, zorder=6, marker='*', edgecolors='white', linewidths=0.5)
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(
                norm=plt.Normalize(vmin=min(eval_counts), vmax=max(eval_counts)),
                cmap='viridis'),
            ax=ax, label='Metric calls at discovery', shrink=0.8)
        ax.set_xlabel('Candidate index', fontsize=11)
        ax.set_ylabel('Val aggregate score', fontsize=11)
        ax.set_title('Candidate Lineage', fontsize=13, fontweight='semibold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(output_dir / 'gepa_candidate_lineage.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved gepa_candidate_lineage.png")

    # ── 5. Pareto front coverage ────────────────────────────────────────
    best_per_instance = detailed_results.per_val_instance_best_candidates
    if best_per_instance:
        counts = np.zeros(n_candidates, dtype=int)
        for entry in best_per_instance:
            indices = entry if isinstance(entry, (set, frozenset, list, tuple)) else [entry]
            for idx in indices:
                counts[idx] += 1

        fig, ax = plt.subplots(figsize=(max(8, n_candidates * 0.6), 5))
        colors_cov = ['#F44336' if i == best_idx else '#81C784' for i in range(n_candidates)]
        ax.bar(range(n_candidates), counts, color=colors_cov,
               edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.set_xlabel('Candidate index', fontsize=11)
        ax.set_ylabel('# val instances where this candidate is best', fontsize=11)
        ax.set_title('Pareto Front Coverage', fontsize=13, fontweight='semibold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(output_dir / 'gepa_pareto_coverage.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved gepa_pareto_coverage.png")


def init_task_lm(cfg: DictConfig):
    """Initialize the task LM based on provider config (mirrors collect_llm_predictions.py)."""
    provider = cfg.lm.provider

    if provider == 'gnesys':
        print("  Initializing GNEsys...")
        gnesys_predictor = GNEsysPredictor.from_config_path(
            config_path=cfg.gnesys.config_path,
            config_name=cfg.gnesys.config_name,
            overrides=cfg.gnesys.overrides,
            verbose=cfg.gnesys.verbose,
        )
        lm = GNEsysLM(
            gnesys_predictor=gnesys_predictor,
            reuse_kernels=cfg.gnesys.get('reuse_kernels', True),
            max_kernels=cfg.gnesys.get('max_kernels', 5),
        )
        init_prompt = gnesys_predictor.get_init_prompt()
        sig_class = create_ranking_signature(init_prompt)
        return lm, sig_class

    if provider == 'azure':
        api_key = os.environ.get('AZURE_API_KEY') or os.environ.get('AZURE_OPENAI_API_KEY')
        api_base = os.environ.get('AZURE_API_BASE') or os.environ.get('AZURE_OPENAI_ENDPOINT')
        if not api_key:
            raise ValueError("Azure OpenAI requires AZURE_API_KEY or AZURE_OPENAI_API_KEY")
        print(f"  Using Azure OpenAI: {api_base or '(litellm defaults)'}")
        print(f"  Deployment: {cfg.lm.model}")
        lm = dspy.LM(
            model=f"azure/{cfg.lm.model}",
            max_tokens=cfg.lm.get('max_tokens', 16000),
            temperature=cfg.lm.get('temperature', 1.0),
        )

    elif provider == 'anthropic':
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic requires ANTHROPIC_API_KEY")
        print(f"  Using Anthropic Claude: {cfg.lm.model}")
        lm = dspy.LM(
            model=f"anthropic/{cfg.lm.model}",
            max_tokens=cfg.lm.get('max_tokens', 16000),
            temperature=cfg.lm.get('temperature', 1.0),
        )

    elif provider == 'gemini':
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini requires GEMINI_API_KEY")
        print(f"  Using Google Gemini: {cfg.lm.model}")
        lm = dspy.LM(
            model=f"gemini/{cfg.lm.model}",
            max_tokens=cfg.lm.get('max_tokens', 16000),
            temperature=cfg.lm.get('temperature', 1.0),
        )

    elif provider == 'biomni':
        api_key = os.environ.get('AGENECY_API_KEY')
        api_base = cfg.lm.get('api_base') or os.environ.get('BIOMNI_API_BASE', '')
        if not api_key:
            raise ValueError("Biomni requires AGENECY_API_KEY")
        print(f"  Using Biomni: {cfg.lm.model}")
        lm = BiomniLM(
            api_key=api_key,
            api_base=api_base,
            model=cfg.lm.model,
            max_tokens=cfg.lm.get('max_tokens', 16000),
            temperature=cfg.lm.get('temperature', 1.0),
        )

    elif provider == 'local':
        api_base = cfg.lm.get('api_base', 'http://localhost:8000/v1')
        api_key = cfg.lm.get('api_key', 'token-abc123')
        print(f"  Using Local LLM: {api_base}")
        print(f"  Model: {cfg.lm.model}")
        extra_kwargs = {}
        if cfg.lm.get('top_p') is not None:
            extra_kwargs['top_p'] = cfg.lm.top_p
        if cfg.lm.get('chat_template_kwargs') is not None:
            extra_kwargs['extra_body'] = extra_kwargs.get('extra_body', {})
            extra_kwargs['extra_body']['chat_template_kwargs'] = dict(cfg.lm.chat_template_kwargs)
        lm = dspy.LM(
            model=f"openai/{cfg.lm.model}",
            api_base=api_base,
            api_key=api_key,
            max_tokens=cfg.lm.get('max_tokens', 16000),
            temperature=cfg.lm.get('temperature', 1.0),
            **extra_kwargs,
        )

    else:
        print(f"  Using OpenAI: {cfg.lm.model}")
        lm = dspy.LM(
            model=cfg.lm.model,
            max_tokens=cfg.lm.get('max_tokens', 16000),
            temperature=cfg.lm.get('temperature', 1.0),
        )

    return lm, create_ranking_signature()


def init_reflection_lm(cfg: DictConfig) -> dspy.LM:
    """Initialize the reflection LM used by GEPA for prompt mutation."""
    provider = cfg.dspy_lm.get('provider', 'azure')

    if provider == 'azure':
        api_key = os.environ.get('AZURE_API_KEY') or os.environ.get('AZURE_OPENAI_API_KEY')
        api_base = os.environ.get('AZURE_API_BASE') or os.environ.get('AZURE_OPENAI_ENDPOINT')
        if not api_key or not api_base:
            raise ValueError(
                "GEPA reflection LM requires AZURE_API_KEY and AZURE_API_BASE env vars"
            )
        print(f"  Reflection LM: Azure {cfg.dspy_lm.model} @ {api_base}")
        return dspy.LM(
            model=f"azure/{cfg.dspy_lm.model}",
            max_tokens=cfg.dspy_lm.get('max_tokens', 16000),
            temperature=cfg.dspy_lm.get('temperature', 1.0),
            reasoning_effort=cfg.dspy_lm.get('reasoning_effort', None),
        )

    elif provider == 'gemini':
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEPA reflection LM requires GEMINI_API_KEY env var")
        print(f"  Reflection LM: Gemini {cfg.dspy_lm.model}")
        return dspy.LM(
            model=f"gemini/{cfg.dspy_lm.model}",
            max_tokens=cfg.dspy_lm.get('max_tokens', 16000),
            temperature=cfg.dspy_lm.get('temperature', 1.0),
        )

    elif provider == 'anthropic':
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("GEPA reflection LM requires ANTHROPIC_API_KEY env var")
        print(f"  Reflection LM: Anthropic {cfg.dspy_lm.model}")
        return dspy.LM(
            model=f"anthropic/{cfg.dspy_lm.model}",
            max_tokens=cfg.dspy_lm.get('max_tokens', 16000),
            temperature=cfg.dspy_lm.get('temperature', 1.0),
        )

    else:
        print(f"  Reflection LM: {cfg.dspy_lm.model}")
        return dspy.LM(
            model=cfg.dspy_lm.model,
            max_tokens=cfg.dspy_lm.get('max_tokens', 16000),
            temperature=cfg.dspy_lm.get('temperature', 1.0),
        )


@hydra.main(version_base="1.3", config_path="../configs/gepa", config_name="gemini-3-flash")
def main(cfg: DictConfig):
    print("=" * 60)
    print("GEPA PROMPT OPTIMIZATION + TEST EVALUATION")
    print("=" * 60)

    model_name = cfg.get('model_name', 'unnamed_model')
    print(f"\nTask model: {model_name}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── 1. Initialize LMs ───────────────────────────────────────────────
    print("\nInitializing task LM...")
    task_lm, RankingSignatureClass = init_task_lm(cfg)

    print("\nInitializing GEPA reflection LM...")
    reflection_lm = init_reflection_lm(cfg)

    dspy.configure(lm=task_lm)

    # ── 2. Load dataset ─────────────────────────────────────────────────
    print("\nLoading BioGRIDDSPY dataset...")
    dataset = BioGRIDDSPY(
        dataset_path=cfg.dataset.dataset_path,
        split_type=cfg.dataset.split_type,
        fold=cfg.dataset.fold,
    )
    train_raw, val_raw, test_raw = dataset.get_train_test_split()

    train_dspy = create_dspy_examples(train_raw)
    val_dspy = create_dspy_examples(val_raw)
    test_dspy = create_dspy_examples(test_raw)

    print(f"  Train: {len(train_dspy)} examples (GEPA optimization)")
    print(f"  Val:   {len(val_dspy)} examples (GEPA Pareto frontier)")
    print(f"  Test:  {len(test_dspy)} examples (held out for evaluation)")

    # ── 3. Create module & metric ────────────────────────────────────────
    output_dir = Path(cfg.output.save_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    load_saved = cfg.get('load_saved_model', False)
    model_file = output_dir / "optimized_model.pkl"

    if load_saved and model_file.exists():
        # ── 4a. Load previously saved optimized model ────────────────────
        print("\n" + "=" * 60)
        print("LOADING SAVED OPTIMIZED MODEL")
        print("=" * 60)
        print(f"  Loading from {model_file}")
        optimized_model = RankingModule(signature_class=RankingSignatureClass)
        optimized_model.load(str(model_file), allow_pickle=True)
        optimized_prompt = extract_optimized_prompt(optimized_model)
        print(f"  Optimized prompt ({len(optimized_prompt)} chars)")
    else:
        # ── 4b. Run GEPA optimization ────────────────────────────────────
        model = RankingModule(signature_class=RankingSignatureClass)

        metric = create_ranking_metric(
            k_values=list(cfg.evaluation.k_values),
            use_thresholded_scoring=cfg.evaluation.use_thresholded_scoring,
        )

        print("\n" + "=" * 60)
        print("RUNNING GEPA OPTIMIZATION")
        print("=" * 60)
        print(f"  max_metric_calls: {cfg.gepa.max_metric_calls}")
        print(f"  reflection_minibatch_size: {cfg.gepa.reflection_minibatch_size}")
        print(f"  use_merge: {cfg.gepa.use_merge}")
        print(f"  max_merge_invocations: {cfg.gepa.max_merge_invocations}")

        gepa_optimizer = GEPA(
            metric=metric,
            max_metric_calls=cfg.gepa.max_metric_calls,
            reflection_minibatch_size=cfg.gepa.reflection_minibatch_size,
            reflection_lm=reflection_lm,
            track_stats=True,
            log_dir=str(output_dir),
            use_merge=cfg.gepa.use_merge,
            max_merge_invocations=cfg.gepa.max_merge_invocations,
            seed=cfg.seed,
        )

        optimized_model = gepa_optimizer.compile(
            model,
            trainset=train_dspy,
            valset=val_dspy,
        )

        # ── 5. Save optimized prompt & model ─────────────────────────────
        print("\n" + "=" * 60)
        print("SAVING OPTIMIZED PROMPT & MODEL")
        print("=" * 60)

        optimized_prompt = extract_optimized_prompt(optimized_model)
        prompt_file = output_dir / "optimized_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(optimized_prompt)
        print(f"  Optimized prompt ({len(optimized_prompt)} chars) saved to {prompt_file}")

        optimized_model.save(str(model_file))
        print(f"  Optimized model saved to {model_file}")

        # ── 5b. Save detailed_results & generate visualizations ──────────
        detailed = getattr(optimized_model, 'detailed_results', None)
        if detailed is not None:
            results_file = output_dir / "gepa_detailed_results.json"
            raw_best = detailed.per_val_instance_best_candidates
            if raw_best:
                serialized_best = [
                    list(s) if isinstance(s, (set, frozenset, list, tuple)) else [s]
                    for s in raw_best
                ]
            else:
                serialized_best = []
            serializable = dict(
                parents=detailed.parents,
                val_aggregate_scores=detailed.val_aggregate_scores,
                val_subscores=detailed.val_subscores,
                per_val_instance_best_candidates=serialized_best,
                discovery_eval_counts=detailed.discovery_eval_counts,
                total_metric_calls=detailed.total_metric_calls,
                num_full_val_evals=detailed.num_full_val_evals,
                log_dir=detailed.log_dir,
                seed=detailed.seed,
                best_idx=detailed.best_idx,
            )
            with open(results_file, 'w') as f:
                json.dump(serializable, f, indent=2, default=str)
            print(f"  Detailed results saved to {results_file}")

            print("\n" + "=" * 60)
            print("GENERATING GEPA VISUALIZATIONS")
            print("=" * 60)
            try:
                plot_gepa_results(detailed, output_dir / "plots")
            except Exception as e:
                print(f"  Warning: could not generate some plots: {e}")
        else:
            print("  No detailed_results available; skipping visualizations.")

    # ── 6. Collect predictions with optimized model ────────────────────
    lm_config = OmegaConf.to_container(cfg.lm, resolve=True)
    pred_dir = output_dir / "llm_predictions" / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    save_frequency = cfg.collection.get('save_frequency', 5)
    num_workers = cfg.collection.get('num_workers', 1)

    def _save_split_predictions(split_name, predictions, output_file):
        num_trunc = sum(1 for p in predictions if p.get('any_truncated', False))
        total_trunc = sum(sum(p.get('truncated', [])) for p in predictions)
        data = {
            'model_name': model_name,
            'lm_config': lm_config,
            'n_runs': cfg.collection.n_runs,
            'split': split_name,
            'num_examples': len(predictions),
            'num_examples_with_truncation': num_trunc,
            'total_truncated_runs': total_trunc,
            'optimized_prompt': optimized_prompt,
            'predictions': predictions,
            'status': 'complete',
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n  Saved {len(predictions)} {split_name} predictions to {output_file}")

    # ── 6a. Train predictions ────────────────────────────────────────────
    if cfg.collection.get('collect_train', False) and len(train_dspy) > 0:
        print("\n" + "=" * 60)
        print("COLLECTING OPTIMIZED MODEL PREDICTIONS ON TRAIN SET")
        print("=" * 60)
        train_output_file = pred_dir / "train_predictions.json"
        train_predictions = collect_predictions(
            model=optimized_model,
            examples=train_dspy,
            n_runs=cfg.collection.n_runs,
            split_name="TRAIN",
            output_file=train_output_file,
            model_name=model_name,
            lm_config=lm_config,
            save_frequency=save_frequency,
            num_workers=num_workers,
            retry_failed=cfg.collection.get('retry_failed', False),
            verbose=True,
        )
        _save_split_predictions('train', train_predictions, train_output_file)

    # ── 6b. Val predictions (optimized model on val set) ─────────────────
    if cfg.collection.get('collect_val', False) and len(val_dspy) > 0:
        print("\n" + "=" * 60)
        print("EVALUATING OPTIMIZED MODEL ON VAL SET")
        print("=" * 60)
        val_output_file = pred_dir / "val_predictions.json"
        val_predictions = collect_predictions(
            model=optimized_model,
            examples=val_dspy,
            n_runs=cfg.collection.n_runs,
            split_name="VAL",
            output_file=val_output_file,
            model_name=model_name,
            lm_config=lm_config,
            save_frequency=save_frequency,
            num_workers=num_workers,
            retry_failed=cfg.collection.get('retry_failed', False),
            verbose=True,
        )
        _save_split_predictions('val', val_predictions, val_output_file)

    # ── 6c. Test predictions ─────────────────────────────────────────────
    if cfg.collection.get('collect_test', True) and len(test_dspy) > 0:
        print("\n" + "=" * 60)
        print("EVALUATING OPTIMIZED MODEL ON TEST SET")
        print("=" * 60)
        test_output_file = pred_dir / "test_predictions.json"
        test_predictions = collect_predictions(
            model=optimized_model,
            examples=test_dspy,
            n_runs=cfg.collection.n_runs,
            split_name="TEST",
            output_file=test_output_file,
            model_name=model_name,
            lm_config=lm_config,
            save_frequency=save_frequency,
            num_workers=num_workers,
            retry_failed=cfg.collection.get('retry_failed', False),
            verbose=True,
        )
        _save_split_predictions('test', test_predictions, test_output_file)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


def plot_from_saved_results(results_json_path: str, output_dir: str | None = None):
    """Generate plots from a saved gepa_detailed_results.json file.

    Usage:
      python scripts/run_gepa_collect_predictions.py --plot-only output/gepa/gemini-3-flash/gepa_detailed_results.json
    """
    from types import SimpleNamespace

    results_path = Path(results_json_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    result_obj = SimpleNamespace(
        val_aggregate_scores=data['val_aggregate_scores'],
        discovery_eval_counts=data['discovery_eval_counts'],
        val_subscores=data.get('val_subscores', []),
        parents=data.get('parents', []),
        per_val_instance_best_candidates=[
            set(s) for s in data.get('per_val_instance_best_candidates', [])
        ],
    )

    if output_dir is None:
        output_dir = results_path.parent / "plots"
    else:
        output_dir = Path(output_dir)

    print(f"Generating plots from {results_path} → {output_dir}")
    plot_gepa_results(result_obj, output_dir)
    print("Done.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--plot-only":
        out_dir = sys.argv[3] if len(sys.argv) > 3 else None
        plot_from_saved_results(sys.argv[2], out_dir)
    else:
        main()
