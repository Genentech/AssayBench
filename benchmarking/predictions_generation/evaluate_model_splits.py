"""
Script to evaluate LLM model performance across train, validation, and test splits.

This script:
1. Loads predictions from each LLM model for all available splits
2. Computes metrics ONCE per example (cached to disk as per-model files)
3. Re-aggregates metrics for different split configurations without recomputation
4. Creates visualizations showing mean, min, and max values per model per split
5. Saves each model's metrics cache to its own file — adding a new model only writes that file
6. Optionally uses gene mapper to normalize gene symbols before evaluation

Usage:
  # Single split_type/fold
  python scripts/evaluate_model_splits.py \
    models='["gpt-5-mini", "qwen3-235b-a22b-2507"]' \
    metric=adjusted_ndcg@100
    
  # Multiple split_types and folds (fast - metrics computed only once!)
  python scripts/evaluate_model_splits.py \
    models='["gpt-5-mini", "qwen3-235b-a22b-2507"]' \
    dataset.split_types='["year", "author", "cell_line", "phenotype"]' \
    dataset.folds='[0, 1, 2]'
    
  # Force recompute metrics for ALL models (ignore cache)
  python scripts/evaluate_model_splits.py \
    models='["gpt-5-mini"]' \
    force_recompute=true
    
  # Recompute specific models only (keeps cache for others)
  python scripts/evaluate_model_splits.py \
    models='["gpt-5-mini", "qwen3-4b-2507"]' \
    recompute_models='["gpt-5-mini"]'
    
  # Use gene mapper to normalize gene symbols (compares mapped vs unmapped)
  python scripts/evaluate_model_splits.py \
    models='["gpt-5-mini"]' \
    use_gene_mapper=true
"""

import rootutils
rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import json
import os
import hydra
from pathlib import Path
from omegaconf import DictConfig, ListConfig
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from screensqa.benchmark.ranking_metrics import RankingMetrics
from screensqa.utils.gene_mapper import GeneMapper
from screensqa.utils.biogrid_maps import stratify_metrics_by_dataset_name

from scripts.shared_utils import (
    deterministic_hash,
    convert_for_json,
    remove_final_answer_tags,
    load_all_model_predictions,
    load_all_ground_truth,
    load_additional_ground_truth,
    create_example_global_id,
    build_global_to_year_fold0_mapping,
    get_split_assignments,
    load_metrics_cache,
    save_model_metrics_cache,
    compute_all_metrics,
    aggregate_metrics_for_split,
    SKIP_METRIC_FIELDS,
)

# Backward-compatible alias used throughout this file
_convert_for_json = convert_for_json


def _parse_gene_list_from_text(text: str) -> List[str]:
    """Parse comma-separated HGNC gene symbols from an answer string."""
    genes = []
    for token in text.split(','):
        token = token.strip()
        if token and len(token) <= 15:
            genes.append(token)
    return genes


def load_gepa_model_predictions(
    gepa_model_dir: Path,
    model_name: str,
    ground_truth_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Load predictions produced by a GEPA optimization run.

    Looks for standard-format prediction files first (``llm_predictions/<model>/``),
    then falls back to ``generated_best_outputs_valset/`` for val predictions.

    Note: ``generated_best_outputs_valset/`` contains outputs from the initial
    program (before optimization).  Prefer ``val_predictions.json`` produced by
    running the optimized model on the val set, which is the accurate post-GEPA
    evaluation.

    Returns the same dict format as ``load_all_model_predictions``.
    """
    predictions_map: Dict[str, Dict[str, Any]] = {}
    llm_pred_dir = gepa_model_dir / "llm_predictions" / model_name

    # ── Standard-format predictions (val and test) ──────────────────────
    for split in ['train', 'val', 'test']:
        pred_file = llm_pred_dir / f"{split}_predictions.json"
        if not pred_file.exists():
            continue
        with open(pred_file, 'r') as f:
            data = json.load(f)
        n_runs = data.get('n_runs', 1)
        for idx, pred in enumerate(data['predictions']):
            cleaned = [remove_final_answer_tags(p) for p in pred['predictions']]
            predictions_map[f"{split}:{idx}"] = {
                'predictions': cleaned,
                'question': pred['question'],
                'n_runs': n_runs,
                'original_split': split,
                'original_index': idx,
            }

    # ── Fallback: val from generated_best_outputs_valset (pre-optimization) ──
    has_val = any(k.startswith("val:") for k in predictions_map)
    valset_dir = gepa_model_dir / "generated_best_outputs_valset"
    if not has_val and valset_dir.exists():
        print(f"  NOTE: No val_predictions.json found; falling back to "
              f"generated_best_outputs_valset/ (pre-optimization outputs)")
        task_dirs = sorted(
            (d for d in valset_dir.iterdir() if d.is_dir()),
            key=lambda p: int(p.name.split('_')[1]),
        )
        for task_idx, task_dir in enumerate(task_dirs):
            json_files = sorted(task_dir.glob("iter_*_prog_*.json"))
            if not json_files:
                continue
            best_file = json_files[-1]
            with open(best_file, 'r') as f:
                entry = json.load(f)
            genes = _parse_gene_list_from_text(entry.get('answer', ''))

            example_key = f"val:{task_idx}"
            gt = ground_truth_map.get(example_key, {})
            predictions_map[example_key] = {
                'predictions': [genes],
                'question': gt.get('question', ''),
                'n_runs': 1,
                'original_split': 'val',
                'original_index': task_idx,
            }

    return predictions_map



# Functions below (load_all_ground_truth, compute_all_metrics, create_example_global_id,
# build_global_to_year_fold0_mapping, get_split_assignments, aggregate_metrics_for_split,
# load_metrics_cache, save_model_metrics_cache, load_all_model_predictions,
# remove_final_answer_tags, deterministic_hash, _convert_for_json) are now imported
# from scripts.shared_utils above.


def _format_metric_label(metric: str) -> str:
    """Format a metric name into a readable axis label."""
    return metric.replace('_', ' ').replace('@', ' @')


def _get_model_sort_key(
    model: str,
    model_results: Dict[str, Dict[str, Dict]],
    metric: str,
    sort_split: str = 'test',
) -> float:
    """Return mean metric on *sort_split* (descending); fallback to val then train."""
    for s in [sort_split, 'val', 'train']:
        data = model_results.get(model, {}).get(s)
        if data is not None:
            values = data['per_run_means'].get(metric, [])
            if values:
                return -np.nanmean(values)  # negative so sorted() is descending
    return 0.0


def build_phenotype_mapping(
    ground_truth_map: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """Map each example key to its BioGRID phenotype category.

    Returns a dict ``{example_key: phenotype_label}`` for every key whose
    ``dataset_name`` can be resolved via the BioGRID screen index.  Keys
    without a dataset_name or that fail lookup are silently skipped.
    """
    ds_names = {
        v['dataset_name']
        for v in ground_truth_map.values()
        if v.get('dataset_name')
    }
    if not ds_names:
        return {}

    try:
        pheno_df = stratify_metrics_by_dataset_name(
            {ds: 0 for ds in ds_names}
        )
    except Exception as e:
        print(f"  Warning: phenotype stratification failed ({e}); skipping")
        return {}

    ds_to_phenotype: Dict[str, str] = dict(
        zip(pheno_df['dataset_name'], pheno_df['phenotype'])
    )

    key_to_phenotype: Dict[str, str] = {}
    for key, gt in ground_truth_map.items():
        ds = gt.get('dataset_name', '')
        if ds in ds_to_phenotype:
            key_to_phenotype[key] = ds_to_phenotype[ds]
    return key_to_phenotype


def plot_phenotype_stratified_performance(
    model_metrics_cache: Dict[str, Dict[str, Dict[str, Any]]],
    key_to_phenotype: Dict[str, str],
    split_keys: List[str],
    output_file: Path,
    metric: str = 'adjusted_ndcg@100',
    split_label: str = 'test',
    title: str = None,
):
    """Grouped bar chart: one group per phenotype, one bar per model.

    Only considers examples in *split_keys* so the plot reflects a single
    train/val/test split.
    """
    phenotype_keys: Dict[str, List[str]] = {}
    for k in split_keys:
        pheno = key_to_phenotype.get(k)
        if pheno:
            phenotype_keys.setdefault(pheno, []).append(k)
    if not phenotype_keys:
        return

    model_names = sorted(model_metrics_cache.keys())
    phenotypes_sorted = sorted(
        phenotype_keys.keys(),
        key=lambda p: len(phenotype_keys[p]),
        reverse=True,
    )
    short_labels = [p.split('/')[0].strip() for p in phenotypes_sorted]
    n_pheno = len(phenotypes_sorted)
    n_models = len(model_names)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    bar_width = 0.8 / max(n_models, 1)
    fig, ax = plt.subplots(figsize=(max(12, 1.8 * n_pheno * min(n_models, 6)), 7))
    cmap = plt.cm.get_cmap('tab20', max(n_models, 1))
    x = np.arange(n_pheno)

    for i, model in enumerate(model_names):
        mc = model_metrics_cache[model]
        means = []
        for pheno in phenotypes_sorted:
            valid = [k for k in phenotype_keys[pheno] if k in mc]
            if valid:
                agg = aggregate_metrics_for_split(mc, valid, metric)
                if agg:
                    v = agg['per_run_means'].get(metric, [])
                    means.append(np.nanmean(v) if v else 0)
                else:
                    means.append(0)
            else:
                means.append(0)

        positions = x + i * bar_width
        bars = ax.bar(
            positions, means, bar_width,
            label=model, color=cmap(i),
            alpha=0.85, edgecolor='white', linewidth=0.5,
            zorder=3,
        )
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, val + 0.003,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=6, color='#444', rotation=90,
                )

    counts = [len(phenotype_keys[p]) for p in phenotypes_sorted]
    tick_labels = [f'{sl}\n(n={c})' for sl, c in zip(short_labels, counts)]
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(tick_labels, fontsize=8.5, ha='center')
    ax.set_ylabel(_format_metric_label(metric), fontsize=11, labelpad=8)
    ax.legend(
        fontsize=7, frameon=True, framealpha=0.9,
        edgecolor='#ccc', loc='upper right',
        ncol=2 if n_models > 6 else 1,
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.set_title(
        title or f'Performance by Phenotype — {split_label}',
        fontsize=13, fontweight='semibold', pad=10,
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved phenotype plot to {output_file}")
    plt.close(fig)


def plot_model_split_performance(
    model_results: Dict[str, Dict[str, Dict]],
    output_file: Path,
    metric: str = 'adjusted_ndcg@100',
    title: str = None
):
    """
    Plot mean, min, max performance for each model across splits.
    
    Left panel:  grouped bar chart (train / val / test per model).
    Right panel: line plot showing performance degradation across splits.
    """
    splits = ['train', 'val', 'test']

    # Filter to splits that actually have data
    available_splits = [
        s for s in splits
        if any(
            s in model_results[m] and model_results[m][s] is not None
            for m in model_results
        )
    ]
    if not available_splits:
        print("No data available to plot!")
        return

    # Sort models by test performance (best first) for easier reading
    model_names = sorted(
        model_results.keys(),
        key=lambda m: _get_model_sort_key(m, model_results, metric),
    )

    n_models = len(model_names)
    n_splits = len(available_splits)

    # ── Style ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    split_palette = {
        'train': '#4CAF50',   # muted green
        'val':   '#2196F3',   # material blue
        'test':  '#F44336',   # material red
    }
    split_hatch = {'train': '', 'val': '///', 'test': '...'}

    # Distinct markers for up to 15 models
    _markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'p', '*', 'h',
                '<', '>', 'd', '8', 'H']

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(16, 1.2 * n_models + 8), 7),
        gridspec_kw={'width_ratios': [3, 2]},
    )

    # ── Left panel: grouped bar chart ──────────────────────────────────
    ax = axes[0]
    bar_width = 0.8 / n_splits
    x = np.arange(n_models)

    for i, split in enumerate(available_splits):
        means, lo_err, hi_err = [], [], []
        for model in model_names:
            data = model_results[model].get(split)
            if data is not None:
                values = data['per_run_means'][metric]
                m = np.nanmean(values)
                means.append(m)
                lo_err.append(m - np.nanmin(values))
                hi_err.append(np.nanmax(values) - m)
            else:
                means.append(0); lo_err.append(0); hi_err.append(0)

        means = np.array(means)
        positions = x + i * bar_width

        ax.bar(
            positions, means, bar_width,
            label=split.capitalize(),
            color=split_palette.get(split, '#9E9E9E'),
            hatch=split_hatch.get(split, ''),
            alpha=0.85,
            edgecolor='white',
            linewidth=0.6,
            zorder=3,
        )
        ax.errorbar(
            positions, means,
            yerr=[lo_err, hi_err],
            fmt='none', color='#333333',
            capsize=2.5, capthick=1, linewidth=1,
            zorder=4,
        )

        # Annotate only val and test bars (train is usually high & obvious)
        for j, m_val in enumerate(means):
            if m_val > 0 and split != 'train':
                ax.text(
                    positions[j], m_val + max(hi_err[j], 0) + 0.005,
                    f'{m_val:.3f}',
                    ha='center', va='bottom', fontsize=6.5,
                    color='#444444', rotation=90,
                )

    ax.set_ylabel(_format_metric_label(metric), fontsize=11, labelpad=8)
    ax.set_xticks(x + bar_width * (n_splits - 1) / 2)
    ax.set_xticklabels(model_names, rotation=40, ha='right', fontsize=8.5)
    ax.legend(
        fontsize=9, frameon=True, framealpha=0.9,
        edgecolor='#cccccc', loc='upper right',
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.set_title(
        f'{_format_metric_label(metric)}  (Mean ± Min/Max)',
        fontsize=12, fontweight='semibold', pad=10,
    )

    # ── Right panel: line chart ────────────────────────────────────────
    ax = axes[1]
    cmap = plt.cm.get_cmap('tab20', n_models)

    for i, model in enumerate(model_names):
        vals, errs = [], []
        for split in available_splits:
            data = model_results[model].get(split)
            if data is not None:
                v = data['per_run_means'][metric]
                vals.append(np.nanmean(v))
                errs.append(np.nanstd(v))
            else:
                vals.append(np.nan)
                errs.append(0)

        ax.errorbar(
            range(n_splits), vals, yerr=errs,
            marker=_markers[i % len(_markers)],
            markersize=7, linewidth=1.8,
            label=model, color=cmap(i),
            capsize=3, capthick=1,
            markeredgecolor='white', markeredgewidth=0.6,
            zorder=3,
        )

    ax.set_xticks(range(n_splits))
    ax.set_xticklabels([s.capitalize() for s in available_splits], fontsize=10)
    ax.set_ylabel(_format_metric_label(metric), fontsize=11, labelpad=8)
    ax.legend(
        fontsize=7.5, frameon=True, framealpha=0.92,
        edgecolor='#cccccc', loc='best',
        ncol=1 if n_models <= 8 else 2,
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.set_title(
        'Performance Across Splits  (Mean ± Std)',
        fontsize=12, fontweight='semibold', pad=10,
    )

    # ── Suptitle & save ────────────────────────────────────────────────
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {output_file}")
    plt.close(fig)


def plot_mapped_vs_unmapped_comparison(
    unmapped_results: Dict[str, Dict[str, Dict]],
    mapped_results: Dict[str, Dict[str, Dict]],
    output_file: Path,
    metric: str = 'adjusted_ndcg@100',
    title: str = None
):
    """
    Create a comparison figure showing mapped vs unmapped metrics side by side.

    Layout: one panel per split (paired bars) + a delta panel on the right.
    """
    splits = ['train', 'val', 'test']

    # Sort models by test performance (mapped, descending)
    model_names = sorted(
        unmapped_results.keys(),
        key=lambda m: _get_model_sort_key(m, mapped_results, metric),
    )

    available_splits = [
        s for s in splits
        if any(
            s in unmapped_results[m] and unmapped_results[m][s] is not None
            for m in model_names
        )
        and any(
            s in mapped_results[m] and mapped_results[m][s] is not None
            for m in model_names
        )
    ]
    if not available_splits:
        print("No data available to plot comparison!")
        return

    n_models = len(model_names)
    n_splits = len(available_splits)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    unmapped_color = '#EF5350'   # material red 400
    mapped_color   = '#66BB6A'   # material green 400

    fig, axes = plt.subplots(
        1, n_splits + 1,
        figsize=(5.5 * (n_splits + 1), 7),
    )
    if n_splits == 0:
        return
    if n_splits == 1:
        axes = [axes[0], axes[1]]

    bar_width = 0.36
    x = np.arange(n_models)

    # ── Per-split panels ───────────────────────────────────────────────
    for idx, split in enumerate(available_splits):
        ax = axes[idx]
        um_means, mm_means, um_stds, mm_stds = [], [], [], []

        for model in model_names:
            for res, m_list, s_list in [
                (unmapped_results, um_means, um_stds),
                (mapped_results,   mm_means, mm_stds),
            ]:
                data = res.get(model, {}).get(split)
                if data is not None:
                    v = data['per_run_means'].get(metric, [])
                    m_list.append(np.nanmean(v) if v else 0)
                    s_list.append(np.nanstd(v) if v else 0)
                else:
                    m_list.append(0); s_list.append(0)

        ax.bar(
            x - bar_width / 2, um_means, bar_width,
            label='Unmapped', color=unmapped_color, alpha=0.85,
            edgecolor='white', linewidth=0.6,
            yerr=um_stds, capsize=2.5, error_kw={'linewidth': 1, 'color': '#555'},
            zorder=3,
        )
        ax.bar(
            x + bar_width / 2, mm_means, bar_width,
            label='Mapped', color=mapped_color, alpha=0.85,
            hatch='///', edgecolor='white', linewidth=0.6,
            yerr=mm_stds, capsize=2.5, error_kw={'linewidth': 1, 'color': '#555'},
            zorder=3,
        )

        # Light value labels
        for j, (um, mm) in enumerate(zip(um_means, mm_means)):
            for val, xoff in [(um, -bar_width / 2), (mm, bar_width / 2)]:
                if val > 0:
                    ax.text(
                        x[j] + xoff, val + 0.008, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=6.5,
                        color='#444', rotation=90,
                    )

        ax.set_ylabel(_format_metric_label(metric), fontsize=10, labelpad=6)
        ax.set_title(f'{split.capitalize()} Split', fontsize=12, fontweight='semibold', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=40, ha='right', fontsize=8)
        ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='#ccc', loc='upper right')
        ax.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)

    # ── Delta panel (mapped - unmapped) ────────────────────────────────
    ax = axes[-1]
    target_split = 'test' if 'test' in available_splits else available_splits[-1]
    diffs = []

    for model in model_names:
        um_val, mm_val = 0.0, 0.0
        for res, ref in [(unmapped_results, 'um'), (mapped_results, 'mm')]:
            data = res.get(model, {}).get(target_split)
            if data is not None:
                v = data['per_run_means'].get(metric, [])
                val = np.nanmean(v) if v else 0.0
                if ref == 'um':
                    um_val = val
                else:
                    mm_val = val
        diffs.append(mm_val - um_val)

    bar_colors = [mapped_color if d >= 0 else unmapped_color for d in diffs]

    ax.bar(
        x, diffs, 0.55,
        color=bar_colors, alpha=0.85,
        edgecolor='white', linewidth=0.6,
        zorder=3,
    )
    ax.axhline(0, color='#333', linewidth=0.8, zorder=2)

    for j, d in enumerate(diffs):
        va = 'bottom' if d >= 0 else 'top'
        offset = 0.002 if d >= 0 else -0.002
        ax.text(
            x[j], d + offset, f'{d:+.4f}',
            ha='center', va=va, fontsize=7.5, fontweight='semibold',
            color='#333',
        )

    ax.set_ylabel(f'Δ {_format_metric_label(metric)}', fontsize=10, labelpad=6)
    ax.set_title(
        f'Mapping Δ ({target_split.capitalize()})',
        fontsize=12, fontweight='semibold', pad=8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=40, ha='right', fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved mapped vs unmapped comparison to {output_file}")
    plt.close(fig)


def _extract_qwen35_info(model_name: str) -> Optional[Tuple[float, str]]:
    """Extract total parameter count and architecture type from a Qwen3.5 name.

    Returns ``(total_params_B, arch)`` where *arch* is ``'MoE'`` or ``'Dense'``.
    Returns None if the pattern is unrecognised.

    Examples::

        qwen3.5-27b        → (27.0, 'Dense')
        qwen3.5-397b-a17b  → (397.0, 'MoE')
        qwen3.5-0.8b       → (0.8, 'Dense')
    """
    import re
    base_match = re.search(r'qwen3\.5-(\d+(?:\.\d+)?)b', model_name, re.IGNORECASE)
    if not base_match:
        return None
    total_params = float(base_match.group(1))
    is_moe = bool(re.search(r'-a\d+(?:\.\d+)?b', model_name, re.IGNORECASE))
    return (total_params, 'MoE' if is_moe else 'Dense')


def plot_qwen35_scaling(
    model_results: Dict[str, Dict[str, Dict]],
    output_file: Path,
    metric: str = 'adjusted_ndcg@100',
    title: str = None,
):
    """
    Plot Qwen3.5 model family performance vs **total** parameter count,
    with separate lines for Dense and MoE architectures.

    Left panel:  scaling curves (one line per split×architecture combination).
    Right panel: generalisation gap bars grouped by architecture.
    """
    # Filter to qwen3.5 models and extract sizes + arch
    qwen_models: List[Tuple[str, float, str]] = []
    for model in model_results:
        if 'qwen3.5' not in model.lower():
            continue
        info = _extract_qwen35_info(model)
        if info is not None:
            qwen_models.append((model, info[0], info[1]))

    if len(qwen_models) < 2:
        print(f"Skipping Qwen3.5 scaling plot — need ≥2 models, found {len(qwen_models)}")
        return

    qwen_models.sort(key=lambda t: t[1])

    dense_models = [(m, s) for m, s, a in qwen_models if a == 'Dense']
    moe_models   = [(m, s) for m, s, a in qwen_models if a == 'MoE']
    all_model_names = [m for m, _, _ in qwen_models]

    splits = ['train', 'val', 'test']
    available_splits = [
        s for s in splits
        if any(
            s in model_results[m] and model_results[m][s] is not None
            for m in all_model_names
        )
    ]
    if not available_splits:
        print("No data available for Qwen3.5 scaling plot!")
        return

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    split_palette = {
        'train': '#4CAF50',
        'val':   '#2196F3',
        'test':  '#F44336',
    }
    split_marker = {'train': 's', 'val': 'D', 'test': 'o'}
    arch_linestyle = {'Dense': '-', 'MoE': '--'}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={'width_ratios': [3, 2]})

    # ── Left panel: performance vs total param count ───────────────────
    ax = axes[0]

    for split in available_splits:
        color = split_palette.get(split, '#9E9E9E')
        marker = split_marker.get(split, 'o')

        for arch_label, arch_group in [('Dense', dense_models), ('MoE', moe_models)]:
            if not arch_group:
                continue
            names = [m for m, _ in arch_group]
            group_sizes = np.array([s for _, s in arch_group])
            means, stds = [], []
            for model in names:
                data = model_results[model].get(split)
                if data is not None:
                    v = data['per_run_means'].get(metric, [])
                    means.append(np.nanmean(v) if v else np.nan)
                    stds.append(np.nanstd(v) if v else 0)
                else:
                    means.append(np.nan)
                    stds.append(0)

            means = np.array(means)
            stds = np.array(stds)
            ls = arch_linestyle[arch_label]

            ax.errorbar(
                group_sizes, means, yerr=stds,
                marker=marker, markersize=8, linewidth=2,
                linestyle=ls,
                label=f'{split.capitalize()} ({arch_label})',
                color=color,
                capsize=4, capthick=1.2,
                markeredgecolor='white', markeredgewidth=0.8,
                alpha=1.0 if arch_label == 'Dense' else 0.75,
                zorder=3,
            )
            for xi, yi in zip(group_sizes, means):
                if not np.isnan(yi):
                    ax.annotate(
                        f'{yi:.3f}', (xi, yi),
                        textcoords='offset points', xytext=(0, 10),
                        fontsize=6.5, ha='center', color=color,
                    )

    all_sizes = np.array([s for _, s, _ in qwen_models])
    ax.set_xscale('log')
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([f'{s:g}B' for s in all_sizes], fontsize=8.5, rotation=30, ha='right')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}B'))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel('Total Parameters (B)', fontsize=11, labelpad=8)
    ax.set_ylabel(_format_metric_label(metric), fontsize=11, labelpad=8)
    ax.legend(
        fontsize=7.5, frameon=True, framealpha=0.92, edgecolor='#ccc',
        loc='best', ncol=2,
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.set_title(
        'Qwen3.5 Scaling  (solid = Dense, dashed = MoE)',
        fontsize=11, fontweight='semibold', pad=10,
    )

    # ── Right panel: generalisation gap grouped by arch ─────────────────
    ax = axes[1]
    dense_color = '#5C6BC0'  # indigo
    moe_color   = '#FF8A65'  # deep orange

    gap_data: List[Tuple[str, float, float, str]] = []  # (label, size, gap, arch)
    for model, size, arch in qwen_models:
        train_data = model_results[model].get('train')
        test_data  = model_results[model].get('test')
        train_mean = np.nanmean(train_data['per_run_means'].get(metric, [0])) if train_data else np.nan
        test_mean  = np.nanmean(test_data['per_run_means'].get(metric, [0])) if test_data else np.nan
        gap_data.append((f'{size:g}B', size, train_mean - test_mean, arch))

    x = np.arange(len(gap_data))
    bar_colors = [dense_color if a == 'Dense' else moe_color for _, _, _, a in gap_data]
    hatches = ['' if a == 'Dense' else '///' for _, _, _, a in gap_data]
    gaps = np.array([g for _, _, g, _ in gap_data])

    bars = ax.bar(
        x, gaps, 0.6, color=bar_colors, alpha=0.85,
        edgecolor='white', linewidth=0.6, zorder=3,
    )
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.axhline(0, color='#333', linewidth=0.8, zorder=2)

    for j, (_, _, g, _) in enumerate(gap_data):
        if not np.isnan(g):
            va = 'bottom' if g >= 0 else 'top'
            off = 0.002 if g >= 0 else -0.002
            ax.text(x[j], g + off, f'{g:+.4f}', ha='center', va=va,
                    fontsize=7.5, fontweight='semibold', color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in gap_data], fontsize=8.5, rotation=30, ha='right')
    ax.set_xlabel('Total Parameters (B)', fontsize=11, labelpad=8)
    ax.set_ylabel(f'Δ {_format_metric_label(metric)}  (train − test)', fontsize=10, labelpad=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title('Generalisation Gap', fontsize=12, fontweight='semibold', pad=10)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=dense_color, label='Dense'),
                 Patch(facecolor=moe_color, hatch='///', label='MoE')],
        fontsize=9, frameon=True, framealpha=0.9, edgecolor='#ccc', loc='best',
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved Qwen3.5 scaling plot to {output_file}")
    plt.close(fig)


def create_summary_table(
    model_results: Dict[str, Dict[str, Dict]],
    metric: str = 'adjusted_ndcg@100',
    split_type: str = None,
    fold: int = None
) -> str:
    """Create a formatted summary table of results."""
    model_names = list(model_results.keys())
    splits = ['train', 'val', 'test']
    
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("MODEL PERFORMANCE SUMMARY")
    if split_type is not None and fold is not None:
        lines.append(f"Split Type: {split_type}, Fold: {fold}")
    lines.append("=" * 100)
    lines.append(f"\nMetric: {metric}")
    lines.append("-" * 100)
    
    header = f"{'Model':<30} | {'Split':<8} | {'Mean':>10} | {'Min':>10} | {'Max':>10} | {'Std':>10} | {'N Runs':>8}"
    lines.append(header)
    lines.append("-" * 100)
    
    for model in model_names:
        for split in splits:
            if split in model_results[model] and model_results[model][split] is not None:
                data = model_results[model][split]
                values = data['per_run_means'][metric]
                mean_val = np.nanmean(values)
                min_val = np.nanmin(values)
                max_val = np.nanmax(values)
                std_val = np.nanstd(values)
                n_runs = data['n_runs']
                
                row = f"{model:<30} | {split:<8} | {mean_val:>10.4f} | {min_val:>10.4f} | {max_val:>10.4f} | {std_val:>10.4f} | {n_runs:>8}"
                lines.append(row)
            else:
                row = f"{model:<30} | {split:<8} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>8}"
                lines.append(row)
        lines.append("-" * 100)
    
    return "\n".join(lines)


def save_detailed_results(
    model_results: Dict[str, Dict[str, Dict]],
    output_file: Path,
    metric: str = 'adjusted_ndcg@100',
    split_type: str = None,
    fold: int = None
):
    """Save detailed results to JSON file."""
    summary = {
        'split_type': split_type,
        'fold': fold,
        'metric': metric,
        'models': {}
    }
    
    for model, splits_data in model_results.items():
        summary['models'][model] = {}
        for split, data in splits_data.items():
            if data is not None:
                values = data['per_run_means'].get(metric, [])
                if values:
                    summary['models'][model][split] = {
                        'mean': float(np.nanmean(values)),
                        'min': float(np.nanmin(values)),
                        'max': float(np.nanmax(values)),
                        'std': float(np.nanstd(values)),
                        'n_runs': data['n_runs'],
                        'n_examples': data['n_examples'],
                        'per_run_means': _convert_for_json(values),
                        'all_metrics': {
                            k: {
                                'mean': float(np.nanmean(v)),
                                'std': float(np.nanstd(v))
                            }
                            for k, v in data['per_run_means'].items()
                        }
                    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved detailed results to {output_file}")


def evaluate_split_configuration(
    model_metrics_cache: Dict[str, Dict[str, Dict[str, Any]]],
    ground_truth_map: Dict[str, Dict[str, Any]],
    global_to_year_fold0_key: Dict[str, str],
    dataset_path: str,
    split_type: str,
    fold: int,
    metric: str
) -> Optional[Dict[str, Dict[str, Dict]]]:
    """
    Evaluate a specific split configuration using cached metrics (no recomputation).
    
    Uses global IDs to map examples from requested split to their year fold 0 keys.
    
    Returns:
        Dict[model_name][split] = aggregated results
    """
    print(f"\n  Getting split assignments for {split_type} fold {fold}...")
    
    try:
        # Get split assignments (maps to year fold 0 keys for cache lookup)
        split_assignments = get_split_assignments(
            dataset_path,
            global_to_year_fold0_key,
            split_type,
            fold
        )
        
        print(f"    train: {len(split_assignments['train'])} examples")
        print(f"    val: {len(split_assignments['val'])} examples")
        print(f"    test: {len(split_assignments['test'])} examples")
        
    except KeyError as e:
        print(f"\n  WARNING: Split type '{split_type}' with fold {fold} not found in dataset.")
        print(f"  Error: {e}")
        return None
    
    # Aggregate metrics for each model and split
    model_results = {}
    
    for model_name, metrics_cache in model_metrics_cache.items():
        model_results[model_name] = {}
        
        for split_name, split_keys in split_assignments.items():
            # Filter to keys that have cached metrics
            valid_keys = [k for k in split_keys if k in metrics_cache]
            
            if not valid_keys:
                model_results[model_name][split_name] = None
                continue
            
            results = aggregate_metrics_for_split(metrics_cache, valid_keys, metric)
            model_results[model_name][split_name] = results
            
            if results:
                values = results['per_run_means'].get(metric, [])
                if values:
                    print(f"    {model_name} {split_name}: {np.nanmean(values):.4f} (n={results['n_examples']})")
    
    return model_results


def create_aggregate_summary(
    all_results: List[Dict[str, Any]],
    metric: str,
    output_file: Path
):
    """Create an aggregate summary across all split_types and folds."""
    
    lines = []
    lines.append("\n" + "=" * 120)
    lines.append("AGGREGATE SUMMARY ACROSS ALL SPLIT TYPES AND FOLDS")
    lines.append("=" * 120)
    lines.append(f"\nMetric: {metric}")
    lines.append("-" * 120)
    
    header = f"{'Split Type':<12} | {'Fold':<6} | {'Model':<25} | {'Train Mean':>12} | {'Val Mean':>12} | {'Test Mean':>12}"
    lines.append(header)
    lines.append("-" * 120)
    
    for result in all_results:
        split_type = result['split_type']
        fold = result['fold']
        model_results = result['model_results']
        
        for model_name, splits_data in model_results.items():
            train_mean = "N/A"
            val_mean = "N/A"
            test_mean = "N/A"
            
            if 'train' in splits_data and splits_data['train'] is not None:
                values = splits_data['train']['per_run_means'].get(metric, [])
                if values:
                    train_mean = f"{np.nanmean(values):.4f}"
            
            if 'val' in splits_data and splits_data['val'] is not None:
                values = splits_data['val']['per_run_means'].get(metric, [])
                if values:
                    val_mean = f"{np.nanmean(values):.4f}"
            
            if 'test' in splits_data and splits_data['test'] is not None:
                values = splits_data['test']['per_run_means'].get(metric, [])
                if values:
                    test_mean = f"{np.nanmean(values):.4f}"
            
            row = f"{split_type:<12} | {fold:<6} | {model_name:<25} | {train_mean:>12} | {val_mean:>12} | {test_mean:>12}"
            lines.append(row)
        
        lines.append("-" * 120)
    
    summary_text = "\n".join(lines)
    
    with open(output_file, 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\nSaved aggregate summary to {output_file}")
    
    return summary_text


def plot_per_screen_performance(
    per_screen_results: Dict[str, Dict[str, Dict]],
    output_file: Path,
    metric: str = 'adjusted_ndcg@100',
    split_name: str = '',
):
    """
    Plot per-screen performance for an additional split.

    ``per_screen_results`` maps ``model_name`` -> ``screen_name`` -> aggregated
    results dict (as returned by ``aggregate_metrics_for_split``).

    Creates a grouped horizontal bar chart: screens on the y-axis, one bar per
    model, ordered by average performance across models.
    """
    model_names = list(per_screen_results.keys())
    if not model_names:
        return

    all_screens = set()
    for model_data in per_screen_results.values():
        all_screens.update(model_data.keys())
    if not all_screens:
        return

    # Compute average metric per screen across models for sorting
    screen_avg: Dict[str, float] = {}
    for screen in all_screens:
        vals = []
        for model in model_names:
            data = per_screen_results[model].get(screen)
            if data is not None:
                v = data['per_run_means'].get(metric, [])
                if v:
                    vals.append(np.nanmean(v))
        screen_avg[screen] = np.nanmean(vals) if vals else 0.0

    sorted_screens = sorted(all_screens, key=lambda s: screen_avg[s])
    n_screens = len(sorted_screens)
    n_models = len(model_names)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    bar_height = 0.8 / max(n_models, 1)
    fig_height = max(4, 0.5 * n_screens + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    cmap = plt.cm.get_cmap('tab20', max(n_models, 1))
    y = np.arange(n_screens)

    for i, model in enumerate(model_names):
        means = []
        for screen in sorted_screens:
            data = per_screen_results[model].get(screen)
            if data is not None:
                v = data['per_run_means'].get(metric, [])
                means.append(np.nanmean(v) if v else 0)
            else:
                means.append(0)

        positions = y + i * bar_height
        bars = ax.barh(
            positions, means, bar_height,
            label=model, color=cmap(i),
            alpha=0.85, edgecolor='white', linewidth=0.5,
            zorder=3,
        )
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(
                    val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', ha='left', fontsize=7,
                    color='#444',
                )

    ax.set_yticks(y + bar_height * (n_models - 1) / 2)
    ax.set_yticklabels(sorted_screens, fontsize=8)
    ax.set_xlabel(_format_metric_label(metric), fontsize=10, labelpad=6)
    ax.set_title(
        f'Per-Screen Performance — {split_name}',
        fontsize=12, fontweight='semibold', pad=10,
    )
    ax.legend(
        fontsize=7.5, frameon=True, framealpha=0.9,
        edgecolor='#ccc', loc='lower right',
    )
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved per-screen plot to {output_file}")
    plt.close(fig)


def create_per_screen_summary_table(
    per_screen_results: Dict[str, Dict[str, Dict]],
    metric: str,
    split_name: str,
) -> str:
    """Create a formatted per-screen summary table."""
    model_names = sorted(per_screen_results.keys())
    all_screens = set()
    for model_data in per_screen_results.values():
        all_screens.update(model_data.keys())
    sorted_screens = sorted(all_screens)

    col_width = 12
    model_header = " | ".join(f"{m[:col_width]:>{col_width}}" for m in model_names)
    lines = [
        "",
        "=" * (30 + (col_width + 3) * len(model_names)),
        f"PER-SCREEN RESULTS — {split_name}  (metric: {metric})",
        "=" * (30 + (col_width + 3) * len(model_names)),
        f"{'Screen':<30} | {model_header}",
        "-" * (30 + (col_width + 3) * len(model_names)),
    ]

    for screen in sorted_screens:
        vals = []
        for model in model_names:
            data = per_screen_results[model].get(screen)
            if data is not None:
                v = data['per_run_means'].get(metric, [])
                vals.append(f"{np.nanmean(v):.4f}" if v else "N/A")
            else:
                vals.append("N/A")
        row = " | ".join(f"{v:>{col_width}}" for v in vals)
        lines.append(f"{screen:<30} | {row}")

    lines.append("-" * (30 + (col_width + 3) * len(model_names)))
    return "\n".join(lines)


def plot_additional_splits_summary(
    split_aggregates: Dict[str, Dict[str, Dict]],
    output_file: Path,
    metric: str = 'adjusted_ndcg@100',
):
    """
    Grouped bar chart showing average metric per additional split.

    ``split_aggregates`` maps ``split_label`` -> ``model_name`` ->
    aggregated results dict.  Each split_label becomes a group of bars
    (one bar per model).
    """
    split_labels = list(split_aggregates.keys())
    if not split_labels:
        return

    model_names_set: set = set()
    for models in split_aggregates.values():
        model_names_set.update(models.keys())
    model_names = sorted(model_names_set)
    if not model_names:
        return

    n_splits = len(split_labels)
    n_models = len(model_names)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    bar_width = 0.8 / max(n_models, 1)
    fig, ax = plt.subplots(figsize=(max(10, 1.5 * n_splits * n_models), 7))
    cmap = plt.cm.get_cmap('tab20', max(n_models, 1))
    x = np.arange(n_splits)

    for i, model in enumerate(model_names):
        means, stds = [], []
        for sl in split_labels:
            data = split_aggregates[sl].get(model)
            if data is not None:
                v = data['per_run_means'].get(metric, [])
                means.append(np.nanmean(v) if v else 0)
                stds.append(np.nanstd(v) if v else 0)
            else:
                means.append(0)
                stds.append(0)

        positions = x + i * bar_width
        ax.bar(
            positions, means, bar_width,
            label=model, color=cmap(i),
            alpha=0.85, edgecolor='white', linewidth=0.5,
            yerr=stds, capsize=2.5,
            error_kw={'linewidth': 1, 'color': '#555'},
            zorder=3,
        )
        for j, (m_val, pos) in enumerate(zip(means, positions)):
            if m_val > 0:
                ax.text(
                    pos, m_val + stds[j] + 0.005,
                    f'{m_val:.3f}', ha='center', va='bottom',
                    fontsize=6.5, color='#444', rotation=90,
                )

    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(split_labels, fontsize=10)
    ax.set_ylabel(_format_metric_label(metric), fontsize=11, labelpad=8)
    ax.legend(
        fontsize=7.5, frameon=True, framealpha=0.9,
        edgecolor='#ccc', loc='upper right',
        ncol=2 if n_models > 6 else 1,
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.set_title(
        f'Average Performance by Split  ({_format_metric_label(metric)})',
        fontsize=12, fontweight='semibold', pad=10,
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved additional splits summary to {output_file}")
    plt.close(fig)


# Dataset-name exact matches and prefixes for internal/proprietary screens in
# ongoing_screens.  Each maps to a human-readable label used in summary charts.
# Order matters: exact matches are checked first, then prefix matches.
_INTERNAL_SCREEN_EXACT = {
    'jake_treg_suppresion_simplified': 'Treg simplified (internal)',
    'jake_treg_suppresion': 'Treg full (internal)',
}
_INTERNAL_SCREEN_PREFIXES = {
    'NGS7126_': 'NGS7126 CDKi (internal)',
}

# Zhu2025 condition prefixes for sub-split grouping.
_ZHU2025_CONDITION_PREFIXES = ['Rest_', 'Stim8hr_', 'Stim48hr_']


@hydra.main(version_base="1.3", config_path="../configs", config_name="evaluate-model-splits")
def main(cfg: DictConfig):
    """
    Main function for evaluating model performance across splits.
    Computes metrics ONCE and re-uses for different split configurations.
    """
    print("=" * 60)
    print("MODEL SPLIT EVALUATION")
    print("=" * 60)
    
    # Get model names (standard prediction collection)
    if isinstance(cfg.models, ListConfig):
        model_names = list(cfg.models)
    else:
        model_names = [cfg.models] if isinstance(cfg.models, str) else []
    
    # Get GEPA model names
    gepa_model_names = []
    if hasattr(cfg, 'gepa_models') and cfg.gepa_models is not None:
        if isinstance(cfg.gepa_models, ListConfig):
            gepa_model_names = list(cfg.gepa_models)
        elif isinstance(cfg.gepa_models, str):
            gepa_model_names = [cfg.gepa_models]
    gepa_dir = Path(cfg.get('gepa_dir', './output/gepa/')) if gepa_model_names else None

    # Prefix GEPA model names so they appear distinctly in plots/tables
    gepa_display_names = {m: f"gepa/{m}" for m in gepa_model_names}

    # Get baseline model names
    baseline_model_names = []
    if hasattr(cfg, 'baseline_models') and cfg.baseline_models is not None:
        if isinstance(cfg.baseline_models, ListConfig):
            baseline_model_names = list(cfg.baseline_models)
        elif isinstance(cfg.baseline_models, str):
            baseline_model_names = [cfg.baseline_models]
    baseline_dir = Path(cfg.get('baseline_dir', './output/multi_model_ensemble/baseline_predictions/')) if baseline_model_names else None

    baseline_display_names = {m: f"baseline/{m}" for m in baseline_model_names}

    # Get few-shot model names
    fewshot_model_names = []
    if hasattr(cfg, 'fewshot_models') and cfg.fewshot_models is not None:
        if isinstance(cfg.fewshot_models, ListConfig):
            fewshot_model_names = list(cfg.fewshot_models)
        elif isinstance(cfg.fewshot_models, str):
            fewshot_model_names = [cfg.fewshot_models]
    fewshot_dir = Path(cfg.get('fewshot_dir', './output/multi_model_ensemble/fewshot_predictions/')) if fewshot_model_names else None

    fewshot_display_names = {m: f"fewshot/{m}" for m in fewshot_model_names}

    all_model_names = (
        model_names
        + [gepa_display_names[m] for m in gepa_model_names]
        + [baseline_display_names[m] for m in baseline_model_names]
        + [fewshot_display_names[m] for m in fewshot_model_names]
    )
    
    if not all_model_names:
        raise ValueError("Must specify at least one model via models=[...], gepa_models=[...], baseline_models=[...], or fewshot_models=[...]")
    
    print(f"\nModels to evaluate: {model_names}")
    if gepa_model_names:
        print(f"GEPA models to evaluate: {gepa_model_names} (from {gepa_dir})")
    if baseline_model_names:
        print(f"Baseline models to evaluate: {baseline_model_names} (from {baseline_dir})")
    if fewshot_model_names:
        print(f"Few-shot models to evaluate: {fewshot_model_names} (from {fewshot_dir})")
    print(f"Metric: {cfg.metric}")
    
    # Get split_types and folds to evaluate
    if hasattr(cfg.dataset, 'split_types') and cfg.dataset.split_types is not None:
        if isinstance(cfg.dataset.split_types, ListConfig):
            split_types = list(cfg.dataset.split_types)
        else:
            split_types = [cfg.dataset.split_types]
    else:
        split_types = [cfg.dataset.split_type]
    
    if hasattr(cfg.dataset, 'folds') and cfg.dataset.folds is not None:
        if isinstance(cfg.dataset.folds, ListConfig):
            folds = list(cfg.dataset.folds)
        else:
            folds = [cfg.dataset.folds]
    else:
        folds = [cfg.dataset.fold]
    
    print(f"Split types to evaluate: {split_types}")
    print(f"Folds to evaluate: {folds}")

    # Build list of all metrics to visualize (primary + additional)
    metrics_to_plot = [cfg.metric]
    if cfg.get('additional_metrics', None):
        for m in cfg.additional_metrics:
            if m not in metrics_to_plot:
                metrics_to_plot.append(m)
    print(f"Metrics to plot: {metrics_to_plot}")

    n_workers = cfg.get('n_workers', min(os.cpu_count() or 1, 8))
    print(f"Parallel workers for metric computation: {n_workers}")
    
    # Check if gene mapper should be used
    use_gene_mapper = cfg.get('use_gene_mapper', False)
    print(f"Use gene mapper: {use_gene_mapper}")
    
    # Initialize metrics evaluators
    # Always create an unmapped evaluator
    metrics_evaluator_unmapped = RankingMetrics(
        k_values=cfg.evaluation.k_values,
        use_thresholded_scoring=cfg.evaluation.get('use_thresholded_scoring', True),
        use_gene_mapper=False
    )
    
    # Create mapped evaluator if requested
    metrics_evaluator_mapped = None
    gene_mapper = None
    if use_gene_mapper:
        gene_mapper = GeneMapper()
        metrics_evaluator_mapped = RankingMetrics(
            k_values=cfg.evaluation.k_values,
            use_thresholded_scoring=cfg.evaluation.get('use_thresholded_scoring', True),
            use_gene_mapper=True,
            gene_mapper=gene_mapper
        )
    
    # Base directory for predictions
    base_dir = Path(cfg.output.save_dir)
    
    # =========================================================================
    # STEP 1: Load all ground truth (once)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: LOADING GROUND TRUTH")
    print("=" * 60)
    
    ground_truth_map, raw_dataset = load_all_ground_truth(cfg.dataset.dataset_path)
    print(f"Loaded {len(ground_truth_map)} ground truth examples")
    
    # Build global ID mapping for cross-split lookups
    global_to_year_fold0_key = build_global_to_year_fold0_mapping(ground_truth_map)
    print(f"Built global ID mapping for {len(global_to_year_fold0_key)} examples")

    # Build phenotype mapping (example_key -> phenotype) for stratified plots
    print("  Building phenotype mapping from BioGRID metadata...")
    key_to_phenotype = build_phenotype_mapping(ground_truth_map)
    phenotype_set = set(key_to_phenotype.values())
    print(f"  Mapped {len(key_to_phenotype)} examples to {len(phenotype_set)} phenotype categories")
    
    # =========================================================================
    # STEP 2: Load/compute metrics (cached to disk)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: LOADING/COMPUTING METRICS (CACHED TO DISK)")
    print("=" * 60)
    
    # Load existing cache
    cache_dir = base_dir / "model_split_evaluation"
    force_recompute = cfg.get('force_recompute', False)
    
    # Get list of specific models to recompute (if any)
    recompute_models = []
    if hasattr(cfg, 'recompute_models') and cfg.recompute_models is not None:
        if isinstance(cfg.recompute_models, ListConfig):
            recompute_models = list(cfg.recompute_models)
        elif isinstance(cfg.recompute_models, str):
            recompute_models = [cfg.recompute_models]
    
    # Load unmapped cache (always load — per-model files are independent)
    model_metrics_cache_unmapped = load_metrics_cache(cache_dir, use_mapped=False)
    if force_recompute:
        print("Force recompute enabled - will recompute metrics for all requested models")
    if recompute_models:
        print(f"Will recompute metrics for: {recompute_models}")
    
    # Load mapped cache if gene mapper is enabled
    model_metrics_cache_mapped = {}
    if use_gene_mapper:
        model_metrics_cache_mapped = load_metrics_cache(cache_dir, use_mapped=True)
    
    for model_name in model_names:
        # Check if we should recompute this specific model
        should_recompute = model_name in recompute_models
        
        # Load predictions (needed for both mapped and unmapped if not cached)
        needs_unmapped = model_name not in model_metrics_cache_unmapped or force_recompute or should_recompute
        needs_mapped = use_gene_mapper and (model_name not in model_metrics_cache_mapped or force_recompute or should_recompute)
        
        if not needs_unmapped and not needs_mapped:
            cached_count = len(model_metrics_cache_unmapped.get(model_name, {}))
            print(f"\n{model_name}: Using cached metrics ({cached_count} examples)")
            continue
        
        # Load predictions
        predictions_map = load_all_model_predictions(base_dir, model_name)
        
        # Count runs per split
        runs_by_split = {}
        for key, pred in predictions_map.items():
            split = pred['original_split']
            if split not in runs_by_split:
                runs_by_split[split] = {'count': 0, 'n_runs': pred['n_runs']}
            runs_by_split[split]['count'] += 1
        
        total_runs = sum(p['n_runs'] for p in predictions_map.values())
        split_info = ", ".join([f"{s}: {d['count']}×{d['n_runs']}" for s, d in sorted(runs_by_split.items())])
        print(f"\n{model_name}: Loaded {len(predictions_map)} examples ({split_info}) = {total_runs} total predictions")
        
        if not predictions_map:
            print(f"  WARNING: No predictions found for {model_name}")
            model_metrics_cache_unmapped[model_name] = {}
            if use_gene_mapper:
                model_metrics_cache_mapped[model_name] = {}
            continue
        
        # Compute UNMAPPED metrics
        if needs_unmapped:
            if should_recompute:
                print(f"  Computing UNMAPPED metrics (requested via recompute_models)...")
            else:
                print(f"  Computing UNMAPPED metrics (not in cache)...")
            
            metrics_cache = compute_all_metrics(
                predictions_map,
                ground_truth_map,
                metrics_evaluator_unmapped
            , n_workers=n_workers)
            model_metrics_cache_unmapped[model_name] = metrics_cache
            save_model_metrics_cache(cache_dir, model_name, metrics_cache, use_mapped=False)
            print(f"  Computed and cached UNMAPPED metrics for {len(metrics_cache)} examples")
        
        # Compute MAPPED metrics
        if needs_mapped:
            if should_recompute:
                print(f"  Computing MAPPED metrics (requested via recompute_models)...")
            else:
                print(f"  Computing MAPPED metrics (not in cache)...")
            
            metrics_cache = compute_all_metrics(
                predictions_map,
                ground_truth_map,
                metrics_evaluator_mapped
            , n_workers=n_workers)
            model_metrics_cache_mapped[model_name] = metrics_cache
            save_model_metrics_cache(cache_dir, model_name, metrics_cache, use_mapped=True)
            print(f"  Computed and cached MAPPED metrics for {len(metrics_cache)} examples")
    
    # ── GEPA models ─────────────────────────────────────────────────────
    for gepa_model in gepa_model_names:
        display_name = gepa_display_names[gepa_model]
        gepa_model_dir = gepa_dir / gepa_model

        should_recompute = display_name in recompute_models or gepa_model in recompute_models
        needs_unmapped = display_name not in model_metrics_cache_unmapped or force_recompute or should_recompute
        needs_mapped = use_gene_mapper and (display_name not in model_metrics_cache_mapped or force_recompute or should_recompute)

        if not needs_unmapped and not needs_mapped:
            cached_count = len(model_metrics_cache_unmapped.get(display_name, {}))
            print(f"\n{display_name}: Using cached metrics ({cached_count} examples)")
            continue

        predictions_map = load_gepa_model_predictions(gepa_model_dir, gepa_model, ground_truth_map)

        runs_by_split = {}
        for key, pred in predictions_map.items():
            split = pred['original_split']
            if split not in runs_by_split:
                runs_by_split[split] = {'count': 0, 'n_runs': pred['n_runs']}
            runs_by_split[split]['count'] += 1

        total_runs = sum(p['n_runs'] for p in predictions_map.values())
        split_info = ", ".join([f"{s}: {d['count']}x{d['n_runs']}" for s, d in sorted(runs_by_split.items())])
        print(f"\n{display_name}: Loaded {len(predictions_map)} examples ({split_info}) = {total_runs} total predictions")

        if not predictions_map:
            print(f"  WARNING: No predictions found for GEPA model {gepa_model} at {gepa_model_dir}")
            model_metrics_cache_unmapped[display_name] = {}
            if use_gene_mapper:
                model_metrics_cache_mapped[display_name] = {}
            continue

        if needs_unmapped:
            print(f"  Computing UNMAPPED metrics for GEPA model...")
            metrics_cache = compute_all_metrics(predictions_map, ground_truth_map, metrics_evaluator_unmapped, n_workers=n_workers)
            model_metrics_cache_unmapped[display_name] = metrics_cache
            save_model_metrics_cache(cache_dir, display_name, metrics_cache, use_mapped=False)
            print(f"  Computed and cached UNMAPPED metrics for {len(metrics_cache)} examples")

        if needs_mapped:
            print(f"  Computing MAPPED metrics for GEPA model...")
            metrics_cache = compute_all_metrics(predictions_map, ground_truth_map, metrics_evaluator_mapped, n_workers=n_workers)
            model_metrics_cache_mapped[display_name] = metrics_cache
            save_model_metrics_cache(cache_dir, display_name, metrics_cache, use_mapped=True)
            print(f"  Computed and cached MAPPED metrics for {len(metrics_cache)} examples")

    # ── Baseline models ──────────────────────────────────────────────────
    for bl_model in baseline_model_names:
        display_name = baseline_display_names[bl_model]
        bl_model_dir = baseline_dir / bl_model

        should_recompute = display_name in recompute_models or bl_model in recompute_models
        needs_unmapped = display_name not in model_metrics_cache_unmapped or force_recompute or should_recompute
        needs_mapped = use_gene_mapper and (display_name not in model_metrics_cache_mapped or force_recompute or should_recompute)

        if not needs_unmapped and not needs_mapped:
            cached_count = len(model_metrics_cache_unmapped.get(display_name, {}))
            print(f"\n{display_name}: Using cached metrics ({cached_count} examples)")
            continue

        # Baseline predictions use the same JSON format as LLM predictions
        # but are stored under baseline_predictions/ instead of llm_predictions/
        predictions_map = load_all_model_predictions(
            base_dir, bl_model, predictions_subdir="baseline_predictions"
        )

        runs_by_split = {}
        for key, pred in predictions_map.items():
            split = pred['original_split']
            if split not in runs_by_split:
                runs_by_split[split] = {'count': 0, 'n_runs': pred['n_runs']}
            runs_by_split[split]['count'] += 1

        total_runs = sum(p['n_runs'] for p in predictions_map.values())
        split_info = ", ".join([f"{s}: {d['count']}x{d['n_runs']}" for s, d in sorted(runs_by_split.items())])
        print(f"\n{display_name}: Loaded {len(predictions_map)} examples ({split_info}) = {total_runs} total predictions")

        if not predictions_map:
            print(f"  WARNING: No predictions found for baseline {bl_model} at {bl_model_dir}")
            model_metrics_cache_unmapped[display_name] = {}
            if use_gene_mapper:
                model_metrics_cache_mapped[display_name] = {}
            continue

        if needs_unmapped:
            print(f"  Computing UNMAPPED metrics for baseline model...")
            metrics_cache = compute_all_metrics(predictions_map, ground_truth_map, metrics_evaluator_unmapped, n_workers=n_workers)
            model_metrics_cache_unmapped[display_name] = metrics_cache
            save_model_metrics_cache(cache_dir, display_name, metrics_cache, use_mapped=False)
            print(f"  Computed and cached UNMAPPED metrics for {len(metrics_cache)} examples")

        if needs_mapped:
            print(f"  Computing MAPPED metrics for baseline model...")
            metrics_cache = compute_all_metrics(predictions_map, ground_truth_map, metrics_evaluator_mapped, n_workers=n_workers)
            model_metrics_cache_mapped[display_name] = metrics_cache
            save_model_metrics_cache(cache_dir, display_name, metrics_cache, use_mapped=True)
            print(f"  Computed and cached MAPPED metrics for {len(metrics_cache)} examples")

    # ── Few-shot models ─────────────────────────────────────────────────
    for fs_model in fewshot_model_names:
        display_name = fewshot_display_names[fs_model]

        should_recompute = display_name in recompute_models or fs_model in recompute_models
        needs_unmapped = display_name not in model_metrics_cache_unmapped or force_recompute or should_recompute
        needs_mapped = use_gene_mapper and (display_name not in model_metrics_cache_mapped or force_recompute or should_recompute)

        if not needs_unmapped and not needs_mapped:
            cached_count = len(model_metrics_cache_unmapped.get(display_name, {}))
            print(f"\n{display_name}: Using cached metrics ({cached_count} examples)")
            continue

        predictions_map = load_all_model_predictions(
            fewshot_dir, fs_model, predictions_subdir=""
        )

        runs_by_split = {}
        for key, pred in predictions_map.items():
            split = pred['original_split']
            if split not in runs_by_split:
                runs_by_split[split] = {'count': 0, 'n_runs': pred['n_runs']}
            runs_by_split[split]['count'] += 1

        total_runs = sum(p['n_runs'] for p in predictions_map.values())
        split_info = ", ".join([f"{s}: {d['count']}x{d['n_runs']}" for s, d in sorted(runs_by_split.items())])
        print(f"\n{display_name}: Loaded {len(predictions_map)} examples ({split_info}) = {total_runs} total predictions")

        if not predictions_map:
            print(f"  WARNING: No predictions found for few-shot model {fs_model} at {fewshot_dir}")
            model_metrics_cache_unmapped[display_name] = {}
            if use_gene_mapper:
                model_metrics_cache_mapped[display_name] = {}
            continue

        if needs_unmapped:
            print(f"  Computing UNMAPPED metrics for few-shot model...")
            metrics_cache = compute_all_metrics(predictions_map, ground_truth_map, metrics_evaluator_unmapped, n_workers=n_workers)
            model_metrics_cache_unmapped[display_name] = metrics_cache
            save_model_metrics_cache(cache_dir, display_name, metrics_cache, use_mapped=False)
            print(f"  Computed and cached UNMAPPED metrics for {len(metrics_cache)} examples")

        if needs_mapped:
            print(f"  Computing MAPPED metrics for few-shot model...")
            metrics_cache = compute_all_metrics(predictions_map, ground_truth_map, metrics_evaluator_mapped, n_workers=n_workers)
            model_metrics_cache_mapped[display_name] = metrics_cache
            save_model_metrics_cache(cache_dir, display_name, metrics_cache, use_mapped=True)
            print(f"  Computed and cached MAPPED metrics for {len(metrics_cache)} examples")

    # Use unmapped as the primary cache for backward compatibility
    model_metrics_cache = model_metrics_cache_unmapped
    
    # =========================================================================
    # STEP 3: Evaluate each split configuration (fast - just aggregation)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATING SPLIT CONFIGURATIONS (FAST - NO RECOMPUTATION)")
    print("=" * 60)
    
    all_results = []
    all_results_mapped = []  # Store mapped results for comparison
    
    for split_type in split_types:
        for fold in folds:
            print(f"\n{'='*50}")
            print(f"Evaluating: {split_type} fold {fold}")
            print(f"{'='*50}")
            
            # Evaluate UNMAPPED metrics
            print("\n--- UNMAPPED METRICS ---")
            model_results_unmapped = evaluate_split_configuration(
                model_metrics_cache_unmapped,
                ground_truth_map,
                global_to_year_fold0_key,
                cfg.dataset.dataset_path,
                split_type,
                fold,
                cfg.metric
            )
            
            if model_results_unmapped is None:
                continue
            
            # Create output directory
            output_dir = base_dir / "model_split_evaluation" / f"{split_type}_fold{fold}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Print summary table for unmapped
            summary_table = create_summary_table(model_results_unmapped, cfg.metric, split_type, fold)
            print(summary_table)
            
            # Save unmapped summary
            summary_file = output_dir / "summary.txt"
            with open(summary_file, 'w') as f:
                f.write(summary_table)
            
            # Create unmapped visualizations for each metric
            for plot_metric in metrics_to_plot:
                safe_metric = plot_metric.replace('@', '_at_')
                plot_model_split_performance(
                    model_results_unmapped,
                    output_dir / f"model_split_performance_{safe_metric}.png",
                    metric=plot_metric,
                    title=f"Model Performance Across Splits ({split_type} fold {fold}) — {plot_metric}"
                )
                plot_qwen35_scaling(
                    model_results_unmapped,
                    output_dir / f"qwen35_scaling_{safe_metric}.png",
                    metric=plot_metric,
                    title=f"Qwen3.5 Scaling Trends ({split_type} fold {fold}) — {plot_metric}"
                )

            # Save unmapped detailed results
            save_detailed_results(
                model_results_unmapped,
                output_dir / "detailed_results.json",
                metric=cfg.metric,
                split_type=split_type,
                fold=fold
            )
            
            # Phenotype-stratified plots for val and test
            if key_to_phenotype:
                try:
                    split_assignments = get_split_assignments(
                        cfg.dataset.dataset_path,
                        global_to_year_fold0_key,
                        split_type, fold,
                    )
                    for plot_metric in metrics_to_plot:
                        safe_metric = plot_metric.replace('@', '_at_')
                        for sp in ['val', 'test']:
                            sp_keys = split_assignments.get(sp, [])
                            if sp_keys:
                                plot_phenotype_stratified_performance(
                                    model_metrics_cache_unmapped,
                                    key_to_phenotype,
                                    sp_keys,
                                    output_dir / f"phenotype_{sp}_{safe_metric}.png",
                                    metric=plot_metric,
                                    split_label=f"{sp} ({split_type} fold {fold})",
                                    title=f"Performance by Phenotype — {sp.capitalize()} ({split_type} fold {fold}) — {plot_metric}",
                                )
                except Exception as e:
                    print(f"  Warning: phenotype plots failed ({e})")

            all_results.append({
                'split_type': split_type,
                'fold': fold,
                'model_results': model_results_unmapped,
                'output_dir': str(output_dir)
            })
            
            # Evaluate MAPPED metrics if gene mapper is enabled
            if use_gene_mapper:
                print("\n--- MAPPED METRICS (Gene Mapper) ---")
                model_results_mapped = evaluate_split_configuration(
                    model_metrics_cache_mapped,
                    ground_truth_map,
                    global_to_year_fold0_key,
                    cfg.dataset.dataset_path,
                    split_type,
                    fold,
                    cfg.metric
                )
                
                if model_results_mapped is not None:
                    # Print summary table for mapped
                    summary_table_mapped = create_summary_table(
                        model_results_mapped, cfg.metric, split_type, fold
                    )
                    print(summary_table_mapped)
                    
                    # Save mapped summary
                    summary_file_mapped = output_dir / "summary_mapped.txt"
                    with open(summary_file_mapped, 'w') as f:
                        f.write(summary_table_mapped)
                    
                    # Create mapped visualizations for each metric
                    for plot_metric in metrics_to_plot:
                        safe_metric = plot_metric.replace('@', '_at_')
                        plot_model_split_performance(
                            model_results_mapped,
                            output_dir / f"model_split_performance_mapped_{safe_metric}.png",
                            metric=plot_metric,
                            title=f"Model Performance (Mapped) Across Splits ({split_type} fold {fold}) — {plot_metric}"
                        )
                        plot_qwen35_scaling(
                            model_results_mapped,
                            output_dir / f"qwen35_scaling_mapped_{safe_metric}.png",
                            metric=plot_metric,
                            title=f"Qwen3.5 Scaling Trends — Mapped ({split_type} fold {fold}) — {plot_metric}"
                        )

                    # Save mapped detailed results
                    save_detailed_results(
                        model_results_mapped,
                        output_dir / "detailed_results_mapped.json",
                        metric=cfg.metric,
                        split_type=split_type,
                        fold=fold
                    )
                    
                    # Create COMPARISON visualizations (mapped vs unmapped)
                    print("\n--- COMPARISON: Mapped vs Unmapped ---")
                    for plot_metric in metrics_to_plot:
                        safe_metric = plot_metric.replace('@', '_at_')
                        plot_mapped_vs_unmapped_comparison(
                            model_results_unmapped,
                            model_results_mapped,
                            output_dir / f"mapped_vs_unmapped_comparison_{safe_metric}.png",
                            metric=plot_metric,
                            title=f"Mapped vs Unmapped Comparison ({split_type} fold {fold}) — {plot_metric}"
                        )
                    
                    all_results_mapped.append({
                        'split_type': split_type,
                        'fold': fold,
                        'model_results': model_results_mapped,
                        'output_dir': str(output_dir)
                    })
    
    # =========================================================================
    # STEP 4: Evaluate additional splits with per-screen breakdown
    # =========================================================================
    additional_splits_cfg = cfg.get('additional_splits', None)
    additional_split_names = list(additional_splits_cfg.keys()) if additional_splits_cfg else []

    if additional_split_names:
        print("\n" + "=" * 60)
        print("STEP 4: EVALUATING ADDITIONAL SPLITS (PER-SCREEN)")
        print("=" * 60)

        addl_output_dir = base_dir / "model_split_evaluation" / "additional_splits"
        addl_output_dir.mkdir(parents=True, exist_ok=True)

        # Collect per-split aggregates for the summary chart.
        # Maps label -> model -> aggregated results.
        summary_aggregates: Dict[str, Dict[str, Dict]] = {}

        for addl_split_name, addl_cfg in additional_splits_cfg.items():
            addl_paths = list(addl_cfg.get('paths', []))
            if not addl_paths:
                continue

            print(f"\n{'='*50}")
            print(f"Additional split: {addl_split_name}")
            print(f"{'='*50}")

            # Load ground truth
            addl_gt = load_additional_ground_truth(
                split_name=addl_split_name,
                dataset_paths=addl_paths,
                use_existing_prompt=addl_cfg.get('use_existing_prompt', False),
                display_library_genes=addl_cfg.get('display_library_genes', False),
            )
            print(f"  Ground truth: {len(addl_gt)} examples")

            # Build per-screen index (dataset_name -> list of keys)
            screen_to_keys: Dict[str, List[str]] = {}
            for key, gt in addl_gt.items():
                sname = gt.get('dataset_name', 'unknown')
                screen_to_keys.setdefault(sname, []).append(key)

            # For ongoing_screens, group into internal screens and public
            # Zhu2025 sub-conditions for separate reporting.
            is_ongoing = (addl_split_name == 'ongoing_screens')
            # label -> list of example keys
            ongoing_groups: Dict[str, List[str]] = {}
            if is_ongoing:
                for key, gt in addl_gt.items():
                    ds_name = gt.get('dataset_name', '')
                    matched = False
                    if ds_name in _INTERNAL_SCREEN_EXACT:
                        ongoing_groups.setdefault(_INTERNAL_SCREEN_EXACT[ds_name], []).append(key)
                        matched = True
                    else:
                        for prefix, label in _INTERNAL_SCREEN_PREFIXES.items():
                            if ds_name.startswith(prefix):
                                ongoing_groups.setdefault(label, []).append(key)
                                matched = True
                                break
                    if not matched:
                        # Public Zhu2025 — sub-group by condition
                        cond_label = None
                        for cp in _ZHU2025_CONDITION_PREFIXES:
                            if ds_name.startswith(cp):
                                cond_label = f'Zhu2025 {cp.rstrip("_")}'
                                break
                        if cond_label is None:
                            cond_label = 'Zhu2025 (other)'
                        ongoing_groups.setdefault(cond_label, []).append(key)

                for label, keys in sorted(ongoing_groups.items()):
                    print(f"    {label}: {len(keys)} examples")

            # Per-model metrics caches — needed for sub-aggregates
            model_metrics_caches: Dict[str, Dict] = {}

            # Evaluate each model
            per_screen_results: Dict[str, Dict[str, Dict]] = {}
            aggregate_results: Dict[str, Dict] = {}

            for display_name in all_model_names:
                # Determine where to load predictions from
                if display_name.startswith("gepa/"):
                    raw_model = display_name.replace("gepa/", "", 1)
                    pred_map = load_all_model_predictions(
                        gepa_dir / raw_model, raw_model,
                        predictions_subdir="llm_predictions",
                        additional_splits=[addl_split_name],
                    )
                elif display_name.startswith("baseline/"):
                    raw_model = display_name.replace("baseline/", "", 1)
                    pred_map = load_all_model_predictions(
                        base_dir, raw_model,
                        predictions_subdir="baseline_predictions",
                        additional_splits=[addl_split_name],
                    )
                elif display_name.startswith("fewshot/"):
                    raw_model = display_name.replace("fewshot/", "", 1)
                    pred_map = load_all_model_predictions(
                        fewshot_dir if fewshot_dir else base_dir, raw_model,
                        predictions_subdir="",
                        additional_splits=[addl_split_name],
                    )
                else:
                    pred_map = load_all_model_predictions(
                        base_dir, display_name,
                        additional_splits=[addl_split_name],
                    )

                # Keep only the additional split keys
                addl_pred_map = {
                    k: v for k, v in pred_map.items()
                    if k.startswith(f"{addl_split_name}:")
                }
                if not addl_pred_map:
                    print(f"  {display_name}: no predictions for {addl_split_name}")
                    continue

                print(f"  {display_name}: {len(addl_pred_map)} predictions loaded")

                # Compute metrics
                metrics_cache = compute_all_metrics(
                    addl_pred_map, addl_gt, metrics_evaluator_unmapped
                , n_workers=n_workers)
                model_metrics_caches[display_name] = metrics_cache

                # Aggregate over the whole split
                all_keys = list(metrics_cache.keys())
                agg = aggregate_metrics_for_split(metrics_cache, all_keys, cfg.metric)
                if agg:
                    aggregate_results[display_name] = agg
                    vals = agg['per_run_means'].get(cfg.metric, [])
                    if vals:
                        print(f"    Overall {cfg.metric}: {np.nanmean(vals):.4f} (n={agg['n_examples']})")

                # Per-screen breakdown
                per_screen_results[display_name] = {}
                for screen_name, screen_keys in screen_to_keys.items():
                    valid_keys = [k for k in screen_keys if k in metrics_cache]
                    if valid_keys:
                        screen_agg = aggregate_metrics_for_split(metrics_cache, valid_keys, cfg.metric)
                        if screen_agg:
                            per_screen_results[display_name][screen_name] = screen_agg

            # Build summary aggregates for chart
            if is_ongoing and ongoing_groups:
                for group_label, group_keys in sorted(ongoing_groups.items()):
                    group_agg: Dict[str, Dict] = {}
                    for display_name, mc in model_metrics_caches.items():
                        valid = [k for k in group_keys if k in mc]
                        if valid:
                            a = aggregate_metrics_for_split(mc, valid, cfg.metric)
                            if a:
                                group_agg[display_name] = a
                    if group_agg:
                        summary_aggregates[group_label] = group_agg
            elif not is_ongoing:
                if aggregate_results:
                    summary_aggregates[addl_split_name] = aggregate_results

            # Collect all metrics to plot (primary + additional)
            addl_metrics_to_plot = [cfg.metric]
            if cfg.get('additional_metrics', None):
                for m in cfg.additional_metrics:
                    if m not in addl_metrics_to_plot:
                        addl_metrics_to_plot.append(m)

            # Print per-screen table
            if per_screen_results:
                table = create_per_screen_summary_table(
                    per_screen_results, cfg.metric, addl_split_name
                )
                print(table)

                table_file = addl_output_dir / f"per_screen_{addl_split_name}.txt"
                with open(table_file, 'w') as f:
                    f.write(table)

                # Plot per-screen performance for each metric
                for plot_metric in addl_metrics_to_plot:
                    plot_per_screen_performance(
                        per_screen_results,
                        addl_output_dir / f"per_screen_{addl_split_name}_{plot_metric.replace('@', '_at_')}.png",
                        metric=plot_metric,
                        split_name=addl_split_name,
                    )

                # Save detailed JSON
                detail_json = {
                    'split_name': addl_split_name,
                    'metric': cfg.metric,
                    'n_screens': len(screen_to_keys),
                    'n_examples': len(addl_gt),
                    'aggregate': {},
                    'per_screen': {},
                }
                for model, agg in aggregate_results.items():
                    vals = agg['per_run_means'].get(cfg.metric, [])
                    detail_json['aggregate'][model] = {
                        'mean': float(np.nanmean(vals)) if vals else None,
                        'std': float(np.nanstd(vals)) if vals else None,
                        'n_examples': agg['n_examples'],
                    }
                for model, screens in per_screen_results.items():
                    detail_json['per_screen'][model] = {}
                    for sn, sd in screens.items():
                        vals = sd['per_run_means'].get(cfg.metric, [])
                        detail_json['per_screen'][model][sn] = {
                            'mean': float(np.nanmean(vals)) if vals else None,
                            'n_examples': sd['n_examples'],
                        }

                detail_file = addl_output_dir / f"per_screen_{addl_split_name}_results.json"
                with open(detail_file, 'w') as f:
                    json.dump(_convert_for_json(detail_json), f, indent=2)
                print(f"Saved detailed per-screen results to {detail_file}")

        # Summary bar chart across all additional splits (for each metric)
        if summary_aggregates:
            all_addl_metrics = [cfg.metric]
            if cfg.get('additional_metrics', None):
                for m in cfg.additional_metrics:
                    if m not in all_addl_metrics:
                        all_addl_metrics.append(m)
            for plot_metric in all_addl_metrics:
                plot_additional_splits_summary(
                    summary_aggregates,
                    addl_output_dir / f"additional_splits_summary_{plot_metric.replace('@', '_at_')}.png",
                    metric=plot_metric,
                )

    # Create aggregate summary if multiple configurations
    if len(all_results) > 1:
        aggregate_output_dir = base_dir / "model_split_evaluation"
        aggregate_output_dir.mkdir(parents=True, exist_ok=True)
        
        create_aggregate_summary(
            all_results,
            cfg.metric,
            aggregate_output_dir / "aggregate_summary.txt"
        )
        
        # Save aggregate JSON
        aggregate_data = {
            'models': model_names,
            'metric': cfg.metric,
            'split_types': split_types,
            'folds': folds,
            'use_gene_mapper': use_gene_mapper,
            'results': []
        }
        
        for result in all_results:
            result_summary = {
                'split_type': result['split_type'],
                'fold': result['fold'],
                'models': {}
            }
            
            for model_name, splits_data in result['model_results'].items():
                result_summary['models'][model_name] = {}
                for split, data in splits_data.items():
                    if data is not None:
                        values = data['per_run_means'].get(cfg.metric, [])
                        if values:
                            result_summary['models'][model_name][split] = {
                                'mean': float(np.nanmean(values)),
                                'min': float(np.nanmin(values)),
                                'max': float(np.nanmax(values)),
                                'std': float(np.nanstd(values)),
                                'n_runs': data['n_runs'],
                                'n_examples': data['n_examples']
                            }
            
            aggregate_data['results'].append(result_summary)
        
        aggregate_json_file = aggregate_output_dir / "aggregate_results.json"
        with open(aggregate_json_file, 'w') as f:
            json.dump(_convert_for_json(aggregate_data), f, indent=2)
        print(f"Saved aggregate results to {aggregate_json_file}")
        
        # Save mapped aggregate summary if gene mapper was used
        if use_gene_mapper and len(all_results_mapped) > 1:
            create_aggregate_summary(
                all_results_mapped,
                cfg.metric,
                aggregate_output_dir / "aggregate_summary_mapped.txt"
            )
            
            aggregate_data_mapped = {
                'models': model_names,
                'metric': cfg.metric,
                'split_types': split_types,
                'folds': folds,
                'gene_mapped': True,
                'results': []
            }
            
            for result in all_results_mapped:
                result_summary = {
                    'split_type': result['split_type'],
                    'fold': result['fold'],
                    'models': {}
                }
                
                for model_name, splits_data in result['model_results'].items():
                    result_summary['models'][model_name] = {}
                    for split, data in splits_data.items():
                        if data is not None:
                            values = data['per_run_means'].get(cfg.metric, [])
                            if values:
                                result_summary['models'][model_name][split] = {
                                    'mean': float(np.nanmean(values)),
                                    'min': float(np.nanmin(values)),
                                    'max': float(np.nanmax(values)),
                                    'std': float(np.nanstd(values)),
                                    'n_runs': data['n_runs'],
                                    'n_examples': data['n_examples']
                                }
                
                aggregate_data_mapped['results'].append(result_summary)
            
            aggregate_json_file_mapped = aggregate_output_dir / "aggregate_results_mapped.json"
            with open(aggregate_json_file_mapped, 'w') as f:
                json.dump(_convert_for_json(aggregate_data_mapped), f, indent=2)
            print(f"Saved mapped aggregate results to {aggregate_json_file_mapped}")
    
    print("\n" + "=" * 60)
    print("ALL EVALUATIONS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
