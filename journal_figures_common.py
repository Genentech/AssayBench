from __future__ import annotations

import json
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


SCRIPT_DIR = Path("/cv/home/edwarc24/code/PromptOptBioGrid/promptoptbase/scripts")
BASE_DIR = SCRIPT_DIR.parent
EVAL_DIR = BASE_DIR / "output" / "multi_model_ensemble" / "model_split_evaluation"
METRICS_CACHE_DIR = EVAL_DIR / "metrics_cache"
DATASET_PATH = "/cv/data/braid/gnesys/datasets/screensQA/biogrid_v0.4_combined"
RESULTS_DIR = Path("/cv/data/braid/gnesys/datasets/screensQA/results")
OUTPUT_DIR = Path(__file__).resolve().parent / "journal_figures"
CACHE_DIR = Path(__file__).resolve().parent / "journal_figures_cache"
DEFAULT_CACHE_PATH = CACHE_DIR / "figure_data_cache.pkl"
DEFAULT_RESULTS_CACHE_PATH = CACHE_DIR / "results_cache.pkl"
LLM_PRED_DIR = BASE_DIR / "output" / "multi_model_ensemble" / "llm_predictions"
ADDITIONAL_SPLITS_DIR = EVAL_DIR / "additional_splits"
NOVEL_JSON = ADDITIONAL_SPLITS_DIR / "per_screen_novel_public_dataset_results.json"
BAYESIAN_DIR = BASE_DIR / "ensemble_learn" / "bayesian_results"
MEMORIZATION_DIR = BASE_DIR / "output" / "multi_model_ensemble" / "memorization_analysis"
QWEN35_SCALING_MAPPED_PNG = (
    EVAL_DIR / "year_fold0" / "qwen35_scaling_mapped_adjusted_ndcg_at_100.png"
)
LEGACY_ORACLE_CACHE_DIR = EVAL_DIR / "oracle_rerank_cache"
ORACLE_CACHE_DIR = CACHE_DIR / "oracle_rerank_cache"
ORACLE_BACKBONES = ["gemini-3-pro", "gpt-5.4", "claude-opus-4.5"]

NOVEL_DATASET_PATHS = [
    "/cv/data/braid/gnesys/datasets/screensQA/novel_public_2026_dataset/",
]
NOVEL_SPLIT_NAME = "novel_public_dataset"

CLASSIFIER_YEAR_CSV = Path(
    "/cv/data/braid/debroue1/promptoptbase/results/year_split.csv"
)
CLASSIFIER_RANDOM_CSV = Path(
    "/cv/data/braid/debroue1/promptoptbase/results/random_split.csv"
)
CLASSIFIER_RECENT_CSV = Path(
    "/cv/data/braid/debroue1/promptoptbase/results/recent_data_test_metrics_r3ph5o3v.csv"
)

METRICS: Tuple[str, ...] = (
    "adjusted_ndcg@100",
    "precision@100",
    "inverse_precision@100",
    "normalized_precision@100",
    "normalized_inverse_precision@100",
)
METRIC_LABELS = {
    "adjusted_ndcg@100": "AnDCG@100",
    "precision@100": "Precision@100",
    "inverse_precision@100": "OPDR@100",
    "normalized_precision@100": "NPrecision@100",
    "normalized_inverse_precision@100": "NInversePrecision@100",
}

BIOMNI_PREFIX = "biomni"
FEWSHOT_PREFIX = "fewshot/"
LOWER_IS_BETTER = {"inverse_precision@100", "normalized_inverse_precision@100"}

GRPO_STAGING_INTERIM: List[Dict[str, Any]] = [
    {"backbone": "qwen3-30b-instruct-2507", "stage": "Base (zero-shot)", "val": 0.0451, "test": 0.0672, "reward": None},
    {"backbone": "qwen3-30b-instruct-2507", "stage": "SFT", "val": 0.0636, "test": 0.0680, "reward": None},
    {"backbone": "qwen3-30b-instruct-2507", "stage": "GRPO", "val": 0.1114, "test": 0.0806, "reward": "0.47–0.55"},
    {"backbone": "qwen3-30b-instruct-2507", "stage": "SFT + GRPO", "val": 0.0832, "test": 0.0686, "reward": "0.45–0.90"},
    {"backbone": "gpt-oss-120B", "stage": "Base (zero-shot)", "val": 0.1167, "test": 0.1174, "reward": None},
    {"backbone": "gpt-oss-120B", "stage": "SFT", "val": 0.1296, "test": 0.1224, "reward": None},
    {"backbone": "gpt-oss-120B", "stage": "SFT + GRPO best", "val": 0.1361, "test": 0.1293, "reward": "0.47–0.65"},
]

STATISTICAL_BASELINES = {
    "baseline/random",
    "baseline/global-hit-freq",
    "baseline/screen-type-hit-freq",
    "baseline/phenotype-knn-hit-freq",
    "baseline/library-size-prior",
    "baseline/phenotype-hit-freq",
    "baseline/coarse-phenotype-hit-freq",
}
TEXT_BASELINES = {"baseline/bm25", "baseline/gene-name-overlap"}
NETWORK_BASELINES = {"baseline/pagerank", "baseline/degree"}

EXTERNAL_MODELS = {
    "GRPO (qwen3-30b-instruct-2507)": {
        "category": "LLM (trained)",
        "source": "grpo",
        "path": "/cv/data/braid/lix361/openrlhf_output/screensqa_preds_30b_run3",
        "step": "step130",
    },
    "C2S (Gemma-2B LoRA)": {
        "category": "LLM (trained)",
        "source": "c2s",
        "path": str(
            BASE_DIR / "output" / "c2s_screensqa_lora"
            / "c2s_screensqa_lora_2026-03-03-04_43_03"
        ),
    },
}

STAGED_MODEL_PREDICTIONS = {
    "SFT (gpt-oss-120B)": {
        "category": "LLM (trained)",
        "format": "list_predictions",
        "val_path": "/cv/data/braid/lix361/screensqa_gpt_120B_pred/sft/sft_job5897396_val_predictions.json",
        "test_path": "/cv/data/braid/lix361/screensqa_gpt_120B_pred/sft/sft_job5897396_test_predictions.json",
    },
    "SFT + GRPO best (gpt-oss-120B)": {
        "category": "LLM (trained)",
        "format": "list_predictions",
        "val_path": "/cv/data/braid/lix361/screensqa_gpt_120B_pred/sft-grpo/val_predictions.json",
        "test_path": "/cv/data/braid/lix361/screensqa_gpt_120B_pred/sft-grpo/test_predictions.json",
    },
}

CATEGORY_COLORS = {
    "Baseline (statistical)": "#999999",
    "Baseline (text)": "#44AA99",
    "Baseline (network)": "#DDCC77",
    "kNN": "#CC6677",
    "LLM": "#88CCEE",
    "Biomni (agent)": "#44AA44",
    "LLM (trained)": "#332288",
    "Ensemble": "#D95F02",
    "Classifier": "#117733",
    "GEPA": "#AA4499",
    "Oracle rerank": "#E78AC3",
}

ENSEMBLE_RUNS = [
    ("multi_model_knn", BAYESIAN_DIR / "multi_model_knn" / "trials.jsonl"),
    ("top3_equal_weights", BAYESIAN_DIR / "top3_equal_weights" / "trials.jsonl"),
    ("top3_optimize_all", BAYESIAN_DIR / "top3_optimize_all" / "trials.jsonl"),
]

BEST_ENSEMBLE_ALL_METRICS = BAYESIAN_DIR / "best_ensemble_all_metrics.json"
KNN_TEST_DIR = BASE_DIR / "output_latent_biology" / "knn_test"

CACHE_SCHEMA_VERSION = 2
RESULTS_CACHE_SCHEMA_VERSION = 1
_FINAL_ANSWER_RE = re.compile(r"<\s*/?FINAL\s*ANSWER\s*>", re.IGNORECASE)
_COHORT_MARKERS = {"val": "s", "test": "o", "novel": "D"}
_COHORT_COLORS = {"val": "#6BAED6", "test": "#2171B5", "novel": "#CB181D"}
PLOT1_LEGEND_FONTSIZE = 9.0
PLOT1_LEGEND_TITLE_FONTSIZE = 9.0


@dataclass
class MissingDataReport:
    entries: List[Dict[str, Any]] = field(default_factory=list)
    used_grpo_interim: bool = True
    grpo_path: Optional[str] = None

    def add(
        self,
        *,
        model: str,
        split_layout: str,
        cohort: str,
        metric: str,
        status: str,
        reason: str = "",
    ) -> None:
        self.entries.append({
            "model": model,
            "split_layout": split_layout,
            "cohort": cohort,
            "metric": metric,
            "status": status,
            "reason": reason,
        })

    def to_markdown(self) -> str:
        lines = [
            "# Missing data report",
            "",
            f"- GRPO staging: `{'interim table' if self.used_grpo_interim else 'file'}`"
            + (f" ({self.grpo_path})" if self.grpo_path else ""),
            "",
            "| Model | Split layout | Cohort | Metric | Status | Reason |",
            "|---|---|---|---|---|---|",
        ]
        for entry in self.entries:
            lines.append(
                f"| {entry['model']} | {entry['split_layout']} | {entry['cohort']} | "
                f"{entry['metric']} | {entry['status']} | {entry['reason']} |"
            )
        lines.append("")
        lines.append("## Summary counts")
        counts = Counter((entry["status"], entry["reason"][:40]) for entry in self.entries)
        for (status, reason), count in sorted(counts.items()):
            lines.append(f"- {status} / {reason}: {count}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": self.entries,
            "used_grpo_interim": self.used_grpo_interim,
            "grpo_path": self.grpo_path,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MissingDataReport":
        if data is None:
            return cls()
        return cls(
            entries=list(data.get("entries", [])),
            used_grpo_interim=bool(data.get("used_grpo_interim", True)),
            grpo_path=data.get("grpo_path"),
        )


def save_figure_cache(payload: Dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_figure_cache(cache_path: Path) -> Dict[str, Any]:
    with open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    version = payload.get("schema_version")
    if version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported figure cache schema version {version}; "
            f"expected {CACHE_SCHEMA_VERSION}. Regenerate the cache."
        )
    return payload


def save_results_cache(payload: Dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_results_cache(cache_path: Path) -> Dict[str, Any]:
    with open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    version = payload.get("schema_version")
    if version != RESULTS_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported results cache schema version {version}; "
            f"expected {RESULTS_CACHE_SCHEMA_VERSION}. Regenerate the cache."
        )
    return payload


def categorize_model(name: str) -> str:
    lower = name.lower()
    if lower.startswith(BIOMNI_PREFIX):
        return "Biomni (agent)"
    if name in ("Oracle kNN", "Embedding kNN"):
        return "kNN"
    if name.startswith("Oracle"):
        return "Oracle rerank"
    if name in STATISTICAL_BASELINES:
        return "Baseline (statistical)"
    if name in TEXT_BASELINES:
        return "Baseline (text)"
    if name in NETWORK_BASELINES:
        return "Baseline (network)"
    if name.startswith("gepa/"):
        return "GEPA"
    for ext_name, cfg in EXTERNAL_MODELS.items():
        if name == ext_name:
            return cfg["category"]
    for staged_name, cfg in STAGED_MODEL_PREDICTIONS.items():
        if name == staged_name:
            return cfg["category"]
    if name.startswith(("SFT ", "GRPO ", "SFT + GRPO")):
        return "LLM (trained)"
    if name == "Classifier":
        return "Classifier"
    return "LLM"


def short_label(name: str) -> str:
    if name.startswith("baseline/"):
        return name[len("baseline/"):]
    if name.startswith("gepa/"):
        return "GEPA: " + name[len("gepa/"):]
    if name.startswith(FEWSHOT_PREFIX):
        return name[len(FEWSHOT_PREFIX):]
    return name


def is_random_split_eligible(model: str, category: str) -> bool:
    return category in ("LLM", "Biomni (agent)")


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
    })


def plot0_dataset_task(meta_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    vc = meta_df["split"].value_counts()
    axes[0].bar(vc.index.astype(str), vc.values, color="#88CCEE", edgecolor="white")
    axes[0].set_title("Examples per split (year fold 0)", fontweight="semibold")
    axes[0].set_ylabel("Count")

    n_pheno = meta_df["biogrid_screen_rationale"].nunique()
    axes[1].axis("off")
    axes[1].text(
        0.02,
        0.95,
        f"BioGRID ScreensQA v0.4\n"
        f"Total examples: {len(meta_df)}\n"
        f"Distinct screen rationales: {n_pheno}\n"
        f"Task: rank genes by screen relevance (nDCG / precision @K).",
        va="top",
        fontsize=10,
        linespacing=1.45,
    )
    fig.suptitle("Plot 0 — Dataset & task overview", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"plot0_dataset_task.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _dataset_pct_label(pct: float) -> str:
    return f"{pct:.1f}%" if pct >= 3 else ""


def _normalize_dataset_phenotype_series(phenotypes: pd.Series) -> pd.Series:
    phenotypes = pd.Series(phenotypes)
    return (
        phenotypes
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown", "-": "Unknown"})
    )


def _dataset_phenotype_counts(phenotypes: pd.Series) -> pd.Series:
    phenotypes = _normalize_dataset_phenotype_series(phenotypes)
    if len(phenotypes) == 0:
        return pd.Series(dtype="int64")
    return phenotypes.value_counts()


def _build_dataset_phenotype_color_map(
    phenotypes: pd.Series,
    top_n: int = 8,
) -> tuple[list[str], dict[str, Any]]:
    counts = _dataset_phenotype_counts(phenotypes)
    base_categories = counts.index[:top_n].tolist()
    use_other = len(counts) > top_n
    legend_categories = base_categories + (["Other"] if use_other else [])
    cmap = plt.get_cmap("tab20")
    color_map = {
        category: cmap(i % 20) for i, category in enumerate(legend_categories)
    }
    return legend_categories, color_map


def _plot_dataset_phenotype_pies_shared_legend(
    split_phenotype_sets: Dict[str, pd.Series],
    out_path: Path,
    title: str,
    top_n: int = 8,
    figsize: Tuple[float, float] = (18, 6),
    percent_fontsize: int = 10,
    legend_fontsize: int = 14,
    dpi: int = 300,
    legend_categories: Optional[List[str]] = None,
    color_map: Optional[Dict[str, Any]] = None,
) -> None:
    apply_style()
    all_phenotypes_with_multiplicity: List[str] = []
    for phenotypes in split_phenotype_sets.values():
        all_phenotypes_with_multiplicity.extend(
            list(_normalize_dataset_phenotype_series(phenotypes))
        )

    global_counts = _dataset_phenotype_counts(pd.Series(all_phenotypes_with_multiplicity))
    if global_counts.empty:
        return

    if legend_categories is None or color_map is None:
        legend_categories, color_map = _build_dataset_phenotype_color_map(
            pd.Series(all_phenotypes_with_multiplicity),
            top_n=top_n,
        )
    base_categories = [category for category in legend_categories if category != "Other"]
    use_other = "Other" in legend_categories

    n_splits = len(split_phenotype_sets)
    fig, axes = plt.subplots(1, n_splits, figsize=figsize if n_splits <= 3 else (5 * n_splits, 6))
    if n_splits == 1:
        axes = [axes]

    for ax, (split_name, split_phenotypes) in zip(axes, split_phenotype_sets.items()):
        split_phenotypes = _normalize_dataset_phenotype_series(split_phenotypes)
        counts = _dataset_phenotype_counts(split_phenotypes)

        if counts.empty:
            ax.set_title(f"{split_name}\n(no phenotypes)")
            ax.axis("off")
            continue

        if use_other:
            other_count = counts[~counts.index.isin(base_categories)].sum()
            counts = counts[counts.index.isin(base_categories)]
            counts = counts.reindex(base_categories, fill_value=0)
            if other_count > 0:
                counts.loc["Other"] = other_count
        else:
            counts = counts.reindex(base_categories, fill_value=0)

        counts = counts[counts > 0]
        _, _, autotexts = ax.pie(
            counts.values,
            labels=None,
            colors=[color_map[category] for category in counts.index],
            autopct=_dataset_pct_label,
            startangle=90,
        )
        for text in autotexts:
            text.set_fontsize(percent_fontsize)
        ax.set_title(f"{split_name} (n={len(split_phenotypes)})")
        ax.axis("equal")

    handles = [Patch(facecolor=color_map[category], label=category) for category in legend_categories]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(legend_categories), 5),
        frameon=False,
        prop={"size": legend_fontsize},
    )
    for text in legend.get_texts():
        text.set_fontsize(legend_fontsize)

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(out_path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_composition(meta_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    from screensqa.utils.biogrid_maps import _extract_screen_ids_from_dataset_name

    split_order = ["train", "val", "test"]
    split_titles_unique = {
        "train": "train (<=2020) (unique screens)",
        "val": "val (2021) (unique screens)",
        "test": "test (>=2022) (unique screens)",
    }
    split_titles_example = {
        "train": "train (<=2020) (example-level)",
        "val": "val (2021) (example-level)",
        "test": "test (>=2022) (example-level)",
    }
    split_colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}

    df = meta_df.copy()
    df["screen_ids"] = df["dataset_name"].apply(_extract_screen_ids_from_dataset_name).apply(tuple)
    legend_categories, color_map = _build_dataset_phenotype_color_map(
        df["biogrid_phenotype"],
        top_n=8,
    )

    split_phenotypes_unique = {
        split_titles_unique[split]: (
            df[df["split"] == split]
            .drop_duplicates(subset=["screen_ids"])["biogrid_phenotype"]
        )
        for split in split_order
    }
    _plot_dataset_phenotype_pies_shared_legend(
        split_phenotype_sets=split_phenotypes_unique,
        out_path=out_dir / "plot_dataset_phenotype_composition_unique_screens",
        title="Phenotype composition by split (unique screen IDs)",
        top_n=8,
        figsize=(18, 6),
        dpi=dpi,
        legend_categories=legend_categories,
        color_map=color_map,
    )

    split_phenotypes_examples = {
        split_titles_example[split]: df[df["split"] == split]["biogrid_phenotype"]
        for split in split_order
    }
    _plot_dataset_phenotype_pies_shared_legend(
        split_phenotype_sets=split_phenotypes_examples,
        out_path=out_dir / "plot_dataset_phenotype_composition_example_level",
        title="Phenotype composition by split (example-level, multiplicity kept)",
        top_n=8,
        figsize=(18, 6),
        dpi=dpi,
        legend_categories=legend_categories,
        color_map=color_map,
    )

    combined_unique_phenotypes = pd.concat(
        [
            df[df["split"] == split]
            .drop_duplicates(subset=["screen_ids"])["biogrid_phenotype"]
            for split in split_order
        ],
        ignore_index=True,
    )
    _plot_dataset_phenotype_pies_shared_legend(
        split_phenotype_sets={"train+val+test (unique screens)": combined_unique_phenotypes},
        out_path=out_dir / "plot_dataset_phenotype_composition_combined_unique_screens",
        title="Phenotype composition (train+val+test, unique screen IDs)",
        top_n=8,
        figsize=(7, 6),
        dpi=dpi,
        legend_categories=legend_categories,
        color_map=color_map,
    )

    combined_example_phenotypes = pd.concat(
        [df[df["split"] == split]["biogrid_phenotype"] for split in split_order],
        ignore_index=True,
    )
    _plot_dataset_phenotype_pies_shared_legend(
        split_phenotype_sets={"train+val+test (example-level)": combined_example_phenotypes},
        out_path=out_dir / "plot_dataset_phenotype_composition_combined_example_level",
        title="Phenotype composition (train+val+test, example-level, multiplicity kept)",
        top_n=8,
        figsize=(7, 6),
        dpi=dpi,
        legend_categories=legend_categories,
        color_map=color_map,
    )

    phenotype_screen_counts = (
        df[["dataset_name", "biogrid_phenotype", "num_genes"]]
        .sort_values(by="num_genes", ascending=False, na_position="last")
        .drop_duplicates(subset=["dataset_name"], keep="first")
    )
    phenotype_screen_counts["biogrid_phenotype"] = _normalize_dataset_phenotype_series(
        phenotype_screen_counts["biogrid_phenotype"]
    )
    mean_by_phenotype = (
        phenotype_screen_counts.groupby("biogrid_phenotype", as_index=False)["num_genes"]
        .mean()
        .set_index("biogrid_phenotype")
        .reindex([category for category in legend_categories if category != "Other"])
        .reset_index()
    )
    if "Other" in legend_categories:
        other_mask = ~phenotype_screen_counts["biogrid_phenotype"].isin(
            [category for category in legend_categories if category != "Other"]
        )
        if other_mask.any():
            other_mean = phenotype_screen_counts.loc[other_mask, "num_genes"].mean()
            mean_by_phenotype = pd.concat(
                [
                    mean_by_phenotype,
                    pd.DataFrame([{
                        "biogrid_phenotype": "Other",
                        "num_genes": other_mean,
                    }]),
                ],
                ignore_index=True,
            )
    mean_by_phenotype = mean_by_phenotype.dropna(subset=["num_genes"])

    apply_style()
    fig_w = max(8, 1.2 * len(mean_by_phenotype))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    ax.bar(
        mean_by_phenotype["biogrid_phenotype"],
        mean_by_phenotype["num_genes"],
        color=[color_map[category] for category in mean_by_phenotype["biogrid_phenotype"]],
    )
    ax.set_ylabel("Average # relevance genes per screen")
    ax.set_xlabel("Phenotype")
    ax.set_title("Average number of relevance genes by phenotype", fontweight="semibold")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"plot_dataset_relevance_genes_by_phenotype.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot1_axis_label(model: str, max_len: int = 40) -> str:
    label = short_label(model)
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "…"


def _plot1_get_panel_data(
    plot_ready: pd.DataFrame,
    metric: str,
    split_layout: str,
    cohort_order: Sequence[str],
) -> Optional[pd.DataFrame]:
    novel_both = (
        (plot_ready["split_layout"] == "both")
        & (plot_ready["cohort"] == "novel")
    )
    sub = plot_ready[
        (plot_ready["metric"] == metric)
        & ((plot_ready["split_layout"] == split_layout) | novel_both)
    ]
    if sub.empty:
        return None
    pivoted = sub.pivot_table(
        index="model",
        columns="cohort",
        values="value",
        aggfunc="mean",
    )
    for cohort in cohort_order:
        if cohort not in pivoted.columns:
            pivoted[cohort] = np.nan
    pivoted = pivoted[list(cohort_order)]
    ascending = metric in LOWER_IS_BETTER
    return pivoted.sort_values(by="test", ascending=ascending, na_position="last")


def _plot1_dot_panel(
    ax: plt.Axes,
    pivoted: pd.DataFrame,
    cohort_order: Sequence[str],
    title: str,
    show_ylabel: bool = True,
) -> None:
    n_rows = len(pivoted.index)
    y_positions = np.arange(n_rows)

    for cohort in cohort_order:
        values = pivoted[cohort].values
        mask = ~np.isnan(values)
        ax.scatter(
            values[mask],
            y_positions[mask],
            marker=_COHORT_MARKERS[cohort],
            color=_COHORT_COLORS[cohort],
            s=38,
            zorder=3,
            label=cohort.capitalize(),
            edgecolors="white",
            linewidths=0.4,
        )

    for idx in y_positions:
        row_values = [pivoted.iloc[idx][cohort] for cohort in cohort_order if pd.notna(pivoted.iloc[idx][cohort])]
        if len(row_values) >= 2:
            ax.plot(
                [min(row_values), max(row_values)],
                [idx, idx],
                color="#CCCCCC",
                linewidth=0.6,
                zorder=1,
            )

    ax.set_yticks(y_positions)
    if show_ylabel:
        labels = [_plot1_axis_label(model) for model in pivoted.index]
        ax.set_yticklabels(labels, fontsize=7.5)
        for tick_label, model in zip(ax.get_yticklabels(), pivoted.index):
            tick_label.set_color(CATEGORY_COLORS.get(categorize_model(model), "#333333"))
            tick_label.set_fontweight("bold")
    else:
        ax.set_yticklabels([])

    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlim(left=0.0)
    ax.axvline(0, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=9.5, pad=6)

    separator_categories = {"Oracle rerank", "Ensemble"}
    model_categories = [categorize_model(model) for model in pivoted.index]
    for idx in range(n_rows - 1):
        above_special = model_categories[idx] in separator_categories
        below_special = model_categories[idx + 1] in separator_categories
        if above_special != below_special:
            ax.axhline(idx + 0.5, color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=2)


def plot1_whole_dataset(
    plot_ready: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    report: MissingDataReport,
    filename_stem: str = "plot1_whole_dataset",
    figure_title: str = "Plot 1 — Benchmark summary: val / test / novel cohorts",
    year_split_subtitle: str = "year split (all methods)",
    random_split_subtitle: str = "random split (zero-shot LLM + classifier + oracle)",
) -> None:
    apply_style()
    cohort_order = ["val", "test", "novel"]
    splits = ["year", "random"]

    row_heights: List[float] = []
    pivots: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
    for metric in METRICS:
        max_rows = 1
        for split_layout in splits:
            pivoted = _plot1_get_panel_data(plot_ready, metric, split_layout, cohort_order)
            pivots[(metric, split_layout)] = pivoted
            if pivoted is None or pivoted.empty:
                report.add(
                    model="*",
                    split_layout=split_layout,
                    cohort="*",
                    metric=metric,
                    status="missing",
                    reason="no rows for plot",
                )
            else:
                max_rows = max(max_rows, len(pivoted.index))
        row_heights.append(max(max_rows, 1))

    fig = plt.figure(figsize=(16.0, min(55.0, 3.2 + sum(0.26 * h for h in row_heights))))
    gs = GridSpec(
        nrows=len(METRICS),
        ncols=2,
        figure=fig,
        height_ratios=row_heights,
        hspace=0.55,
        wspace=0.55,
        left=0.22,
        right=0.96,
        top=0.95,
        bottom=0.10,
    )

    for row_index, metric in enumerate(METRICS):
        for col_index, split_layout in enumerate(splits):
            ax = fig.add_subplot(gs[row_index, col_index])
            pivoted = pivots[(metric, split_layout)]
            if pivoted is None or pivoted.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
                continue
            subtitle = year_split_subtitle if split_layout == "year" else random_split_subtitle
            _plot1_dot_panel(
                ax,
                pivoted,
                cohort_order,
                title=f"{METRIC_LABELS.get(metric, metric)} — {subtitle}",
                show_ylabel=True,
            )
            if row_index == 0 and col_index == 0:
                ax.legend(loc="lower right", fontsize=7.5, framealpha=0.85, handletextpad=0.3, borderpad=0.3)
                leg = ax.get_legend()
                if leg is not None:
                    leg.set_title(leg.get_title().get_text(), prop={"size": PLOT1_LEGEND_TITLE_FONTSIZE})
                    for text in leg.get_texts():
                        text.set_fontsize(PLOT1_LEGEND_FONTSIZE)

    category_patches = [
        Patch(facecolor=color, label=category)
        for category, color in CATEGORY_COLORS.items()
    ]
    fig.legend(
        handles=category_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(category_patches), 5),
        fontsize=PLOT1_LEGEND_FONTSIZE,
        framealpha=0.9,
        title="Model category",
        title_fontsize=PLOT1_LEGEND_TITLE_FONTSIZE,
    )
    fig.suptitle(figure_title, fontsize=13, fontweight="semibold", y=0.998)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"{filename_stem}.{fmt}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot2_phenotype(
    per_ex_year: pd.DataFrame,
    meta_df: pd.DataFrame,
    per_ex_rand: pd.DataFrame,
    models_year: List[str],
    models_rand: List[str],
    metric: str,
    out_dir: Path,
    dpi: int,
    min_examples: int = 4,
) -> None:
    apply_style()
    import seaborn as sns

    def render_one_split(per_ex: pd.DataFrame, models: List[str], title: str, filename: str) -> None:
        test_df = per_ex[
            (per_ex["split"] == "test")
            & (per_ex["model"].isin(models))
            & (per_ex["metric"] == metric)
        ].copy()
        if test_df.empty:
            return
        merged = test_df.merge(
            meta_df[["example_key", "biogrid_screen_rationale", "num_genes"]],
            on="example_key",
            how="left",
        ).dropna(subset=["biogrid_screen_rationale"])
        counts = merged.groupby("biogrid_screen_rationale")["example_key"].nunique()
        keep = counts[counts >= min_examples].index
        merged = merged[merged["biogrid_screen_rationale"].isin(keep)]
        if merged.empty or len(keep) < 2:
            return
        heat_df = merged.groupby(["biogrid_screen_rationale", "model"])["value"].mean().reset_index()
        mean_genes = merged.groupby("biogrid_screen_rationale")["num_genes"].mean()
        pivoted = heat_df.pivot(index="biogrid_screen_rationale", columns="model", values="value")
        pivoted = pivoted[[column for column in models if column in pivoted.columns]]
        row_counts = counts.reindex(pivoted.index).fillna(0).astype(int)
        row_labels = []
        for rationale in pivoted.index:
            n_examples = int(row_counts.loc[rationale])
            mean_gene_text = f"{float(mean_genes.loc[rationale]):.0f}" if rationale in mean_genes.index and pd.notna(mean_genes.loc[rationale]) else "?"
            row_labels.append(f"{rationale} (n={n_examples}, mean genes={mean_gene_text})")

        fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(pivoted.columns) + 3), max(4, 0.4 * len(pivoted.index) + 1.5)))
        sns.heatmap(
            pivoted.values,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=[short_label(column) for column in pivoted.columns],
            yticklabels=row_labels,
            ax=ax,
            cbar_kws={"label": METRIC_LABELS.get(metric, metric)},
        )
        ax.set_title(
            f"{title}\nPrevalence confound: n varies by phenotype; unadjusted means may reflect label density, not only skill.",
            fontsize=9,
        )
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in ("pdf", "png"):
            fig.savefig(out_dir / f"{filename}.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    render_one_split(
        per_ex_year,
        models_year,
        f"Plot 2 — Phenotype (year split, test, {metric})",
        f"plot2_phenotype_year_{metric.replace('@', '_at_')}",
    )
    render_one_split(
        per_ex_rand,
        models_rand,
        f"Plot 2 — Phenotype (random split, test, {metric})",
        f"plot2_phenotype_random_{metric.replace('@', '_at_')}",
    )


def plot2_coarse_phenotype_bar_year(
    per_ex_year: pd.DataFrame,
    meta_df: pd.DataFrame,
    models: List[str],
    out_dir: Path,
    dpi: int,
    metric: str = "adjusted_ndcg@100",
    filename_stem: str = "plot2_phenotype_bar_plot_year",
    min_examples: int = 1,
) -> None:
    apply_style()
    test_df = per_ex_year[
        (per_ex_year["split"] == "test")
        & (per_ex_year["metric"] == metric)
        & (per_ex_year["model"].isin(models))
    ].copy()
    if test_df.empty:
        return

    merged = test_df.merge(
        meta_df[["example_key", "biogrid_phenotype"]],
        on="example_key",
        how="left",
    ).dropna(subset=["biogrid_phenotype"])
    if merged.empty:
        return

    counts = merged.groupby("biogrid_phenotype")["example_key"].nunique()
    keep = counts[counts >= min_examples].index
    merged = merged[merged["biogrid_phenotype"].isin(keep)]
    if merged.empty:
        return

    grouped = merged.groupby(["biogrid_phenotype", "model"], as_index=False)["value"].mean()
    pivoted = grouped.pivot(index="biogrid_phenotype", columns="model", values="value")
    pivoted = pivoted[[model for model in models if model in pivoted.columns]]
    if pivoted.empty:
        return

    phenotype_order = counts.loc[pivoted.index].sort_values(ascending=False).index.tolist()
    pivoted = pivoted.reindex(phenotype_order)
    display_labels = [
        f"{phenotype}\n(n={int(counts.loc[phenotype])})"
        for phenotype in pivoted.index
    ]

    x = np.arange(len(pivoted.index))
    n_models = len(pivoted.columns)
    width = min(0.75 / max(n_models, 1), 0.22)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_models, 1)))

    fig_w = max(10.0, 1.6 * len(pivoted.index) + 0.5 * n_models)
    fig_h = max(5.5, 3.8 + 0.18 * n_models)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for idx, model in enumerate(pivoted.columns):
        offset = (idx - (n_models - 1) / 2) * width
        values = pivoted[model].values
        ax.bar(
            x + offset,
            values,
            width=width,
            label=short_label(model),
            color=colors[idx],
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=18, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(
        "Plot 2 — Year test performance by coarse phenotype",
        fontweight="semibold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    legend = fig.legend(
        title="Method",
        fontsize=9.0,
        title_fontsize=9.0,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=min(4, max(n_models, 1)),
        framealpha=0.95,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout(rect=(0, 0.14, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"{filename_stem}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot3_ensemble(
    trial_rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    dpi: int,
    report: MissingDataReport,
) -> None:
    apply_style()
    if not trial_rows:
        report.add(
            model="LLM RRF Ensemble",
            split_layout="ensemble",
            cohort="*",
            metric="adjusted_ndcg@100",
            status="missing",
            reason="no Bayesian trials found",
        )
        return
    names = [row["label"] for row in trial_rows]
    vals = [row.get("val_score", float("nan")) for row in trial_rows]
    tests = [row.get("test_score", float("nan")) for row in trial_rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.175, vals, width=0.35, label="Val", color="#88CCEE")
    ax.bar(x + 0.175, tests, width=0.35, label="Test", color="#332288")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("AnDCG@100")
    ax.set_title("Plot 3 — Best Bayesian ensemble trial (val vs test fitness)", fontweight="semibold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"plot3_ensemble_val_vs_test.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot4_memorization_composite(
    panels: Sequence[Dict[str, Any]],
    out_dir: Path,
    dpi: int,
    report: MissingDataReport,
) -> None:
    apply_style()
    images = [(panel["title"], panel["image"]) for panel in panels]
    if len(images) < 2:
        report.add(
            model="plot4",
            split_layout="plot4",
            cohort="*",
            metric="*",
            status="missing",
            reason="fewer than 2 cached image panels",
        )
        return
    n_panels = len(images)
    cols = 2
    rows = (n_panels + 1) // 2
    fig = plt.figure(figsize=(12, 4 * rows))
    for idx, (title, image) in enumerate(images):
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title, fontsize=8)
    fig.suptitle(
        "Plot 4 — Scaling (Qwen3.5) + memorization / distribution panels",
        fontsize=11,
        fontweight="semibold",
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"plot4_memorization_composite.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot7_qwen35_param_billions(model_name: str) -> Optional[float]:
    match = re.match(r"^qwen3\.5-(\d+(?:\.\d+)?)b", model_name)
    if not match:
        return None
    return float(match.group(1))


def plot7_scaling_laws(
    model_scores_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    apply_style()

    rows = model_scores_df[
        (model_scores_df["metric"] == "adjusted_ndcg@100")
        & (model_scores_df["split"].isin(["train", "val", "test"]))
    ].copy()
    rows["param_billions"] = rows["model"].map(_plot7_qwen35_param_billions)
    rows = rows.dropna(subset=["param_billions"]).copy()
    if rows.empty:
        raise ValueError(
            "No cached Qwen3.5 adjusted_ndcg@100 rows were found for train/val/test."
        )

    summary = (
        rows.groupby(["model", "param_billions"], as_index=False)["value"]
        .mean()
        .sort_values(["param_billions", "model"])
        .reset_index(drop=True)
    )
    summary["param_label"] = summary["param_billions"].map(lambda value: f"{value:g}B")
    dense_summary = summary[summary["param_billions"] <= 27.0].copy()
    moe_summary = summary[summary["param_billions"] >= 27.0].copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        dense_summary["param_billions"],
        dense_summary["value"],
        color="#4C78A8",
        linewidth=2.0,
    )
    ax.plot(
        moe_summary["param_billions"],
        moe_summary["value"],
        color="#4C78A8",
        linewidth=2.0,
        linestyle=":",
    )
    ax.scatter(
        summary["param_billions"],
        summary["value"],
        color="#F58518",
        edgecolor="white",
        linewidth=0.9,
        s=62,
        zorder=3,
    )

    for _, row in summary.iterrows():
        ax.annotate(
            row["param_label"],
            (row["param_billions"], row["value"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xticks(summary["param_billions"].tolist())
    ax.set_xticklabels(summary["param_label"].tolist())
    ax.set_xlabel("Model parameters (billions)")
    ax.set_ylabel("Mean AnDCG@100")
    ax.set_title(
        "Plot 7 — Qwen3.5 scaling laws\nmean AnDCG@100 across train + val + test screens",
        fontweight="semibold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(axis="x", linestyle=":", alpha=0.18)
    ax.set_ylim(bottom=max(0.0, float(summary["value"].min()) - 0.01))

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"plot7_scaling_laws.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot5_safe_filename_fragment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return safe or "model"


def _plot5_aggregate_model_scores(
    model_scores_df: pd.DataFrame,
    model_name: str,
    value_column: str,
    phenotype_column: str,
) -> pd.DataFrame:
    model_df = model_scores_df[
        (model_scores_df["model"] == model_name)
        & (model_scores_df["metric"] == "adjusted_ndcg@100")
    ].copy()
    if model_df.empty:
        raise ValueError(
            f"Model '{model_name}' has no cached adjusted_ndcg@100 per-example rows."
        )

    return (
        model_df.groupby("dataset_name", as_index=False)
        .agg({
            "value": "mean",
            "example_key": "first",
            "split": "first",
            "biogrid_phenotype": "first",
        })
        .rename(columns={
            "value": value_column,
            "biogrid_phenotype": phenotype_column,
        })
    )


def plot5_duplicate_transfer_vs_model(
    duplicate_transfer_df: pd.DataFrame,
    model_scores_df: pd.DataFrame,
    model_name: str,
    out_dir: Path,
    dpi: int,
    filename_stem: Optional[str] = None,
) -> None:
    apply_style()
    oracle_model_name = "Oracle kNN"

    duplicate_df = duplicate_transfer_df.copy()
    if duplicate_df.empty:
        raise ValueError("Duplicate-transfer cache is empty. Regenerate the figure cache.")

    model_df = _plot5_aggregate_model_scores(
        model_scores_df=model_scores_df,
        model_name=model_name,
        value_column="model_adjusted_ndcg@100",
        phenotype_column="model_phenotype_from_scores",
    )
    oracle_df = _plot5_aggregate_model_scores(
        model_scores_df=model_scores_df,
        model_name=oracle_model_name,
        value_column="oracle_adjusted_ndcg@100",
        phenotype_column="oracle_phenotype_from_scores",
    )

    merged = duplicate_df.merge(
        model_df,
        on="dataset_name",
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        oracle_df,
        on="dataset_name",
        how="left",
        validate="one_to_one",
    )
    merged = merged.dropna(subset=["model_adjusted_ndcg@100"]).copy()
    if merged.empty:
        raise ValueError(
            f"No overlapping duplicate-screen dataset names were found for '{model_name}'."
        )

    merged["phenotype"] = (
        merged["phenotype"]
        .fillna(merged["model_phenotype_from_scores"])
        .fillna(merged["oracle_phenotype_from_scores"])
    )

    base_summary_metrics = [
        "duplicate_avg_adjusted_ndcg@100",
        "model_adjusted_ndcg@100",
    ]
    preferred_order = [
        "Fitness / Proliferation / Viability",
        "Drug / Chemical / Environmental Response",
        "Host-Pathogen / Infection Response",
        "Molecular Output / Reporter / Pathway Activity",
        "Trafficking / Localization / Structural Phenotypes",
        "Not specified",
    ]
    observed_phenotypes = merged["phenotype"].dropna().astype(str).unique().tolist()
    summary_order = [item for item in preferred_order if item in observed_phenotypes]
    summary_order += sorted(set(observed_phenotypes) - set(summary_order))

    comparison_by_phenotype = merged.groupby("phenotype", as_index=False)[base_summary_metrics].mean()
    comparison_by_phenotype["phenotype"] = pd.Categorical(
        comparison_by_phenotype["phenotype"],
        categories=summary_order,
        ordered=True,
    )
    comparison_by_phenotype = comparison_by_phenotype.sort_values("phenotype").reset_index(drop=True)

    overall_row = pd.DataFrame([{
        "phenotype": "Overall",
        **merged[base_summary_metrics].mean().to_dict(),
    }])
    summary_plot_df = pd.concat(
        [
            comparison_by_phenotype.assign(
                phenotype=lambda df: df["phenotype"].astype(str)
            ),
            overall_row,
        ],
        ignore_index=True,
    )

    oracle_overlap = merged.dropna(subset=["oracle_adjusted_ndcg@100"]).copy()
    oracle_summary = (
        oracle_overlap.groupby("phenotype", as_index=False)["oracle_adjusted_ndcg@100"].mean()
        if not oracle_overlap.empty
        else pd.DataFrame(columns=["phenotype", "oracle_adjusted_ndcg@100"])
    )
    oracle_overall = pd.DataFrame([{
        "phenotype": "Overall",
        "oracle_adjusted_ndcg@100": oracle_overlap["oracle_adjusted_ndcg@100"].mean()
        if not oracle_overlap.empty
        else np.nan,
    }])
    oracle_summary = pd.concat([oracle_summary, oracle_overall], ignore_index=True)
    summary_plot_df = summary_plot_df.merge(
        oracle_summary,
        on="phenotype",
        how="left",
        validate="one_to_one",
    )

    x = np.arange(len(summary_plot_df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        x - width,
        summary_plot_df["duplicate_avg_adjusted_ndcg@100"],
        width=width,
        label="Duplicate transfer avg AnDCG@100",
        color="#4C78A8",
    )
    ax.bar(
        x,
        summary_plot_df["model_adjusted_ndcg@100"],
        width=width,
        label=f"{short_label(model_name)} AnDCG@100",
        color="#F58518",
    )
    ax.bar(
        x + width,
        summary_plot_df["oracle_adjusted_ndcg@100"],
        width=width,
        label=f"{short_label(oracle_model_name)} AnDCG@100",
        color="#54A24B",
    )

    if len(summary_plot_df) > 1:
        ax.axvline(
            len(summary_plot_df) - 1.5,
            color="#444444",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
        )

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Mean AnDCG@100")
    ax.set_title(
        f"Plot 5 — Duplicate-screen transfer vs {short_label(model_name)} and "
        f"{short_label(oracle_model_name)}\naggregated by coarse phenotype",
        fontweight="semibold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label.replace(" / ", "\n") for label in summary_plot_df["phenotype"]],
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    legend = fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        framealpha=0.95,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = filename_stem or (
        "plot5_duplicate_transfer_vs_model__"
        f"{_plot5_safe_filename_fragment(model_name)}__with_oracle_knn"
    )
    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_grpo_staging_table(rows: List[Dict[str, Any]], out_dir: Path, dpi: int) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis("off")
    headers = ["Backbone", "Stage", "Val AnDCG@100", "Test AnDCG@100", "Train reward"]
    table_rows = [
        [
            row["backbone"],
            row["stage"],
            f"{row['val']:.4f}",
            f"{row['test']:.4f}",
            row["reward"] if row["reward"] is not None else "—",
        ]
        for row in rows
    ]
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    table.scale(1.2, 1.8)
    ax.set_title(
        "GRPO / SFT staging (interim table — replace with --grpo-staging file)",
        fontsize=11,
        fontweight="semibold",
        pad=12,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(out_dir / f"plot_grpo_staging_table.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_figure_captions(caption_path: Path, report: MissingDataReport) -> None:
    text = """# Figure captions

## Plot 0 — `plot0_dataset_task`
**Caption.** Overview of the ScreensQA BioGRID benchmark: example counts per temporal split (year fold 0) and short description of the gene-ranking task.

## Plot 1 — `plot1_whole_dataset`
**Caption.** Single figure: **rows** = AnDCG@100, precision@100, inverse precision@100; **columns** = year split (full zoo) vs random split (LLM + Biomni + classifier + oracle rerank only). **Columns** val / test / novel within each heatmap. Row heights scale with model count; shared color scale 0–1. Year vs random columns are **not** directly comparable across all methods.

## Plot 2 — `plot2_phenotype_*`
**Caption.** Phenotype-stratified mean metric on the **test** set (year and random where available). Row labels show **n** per phenotype. **Prevalence confound:** label density and library size differ across phenotypes; segment means may partly reflect **who has more hits to find**, not only model skill or “viability” effects.

## Plot 3 — `plot3_ensemble_val_vs_test`
**Caption.** Best validation-score trial from each Bayesian optimization run (multi_model_knn, top3_equal_weights, top3_optimize_all) with corresponding test fitness — illustrates validation–test gap for learned ensembles.

## Plot 4 — `plot4_memorization_composite`
**Caption.** **(1)** Qwen3.5 scaling laws (mapped AnDCG@100), produced by `evaluate_model_splits.py` / `plot_qwen35_scaling` under `model_split_evaluation/year_fold0/`. **(2–4)** Phenotype-stratified year effects, citation-based memorization proxy, and classifier vs LLM comparison from `analyze_memorization_vs_distribution.py` (`memorization_analysis/`).

## Plot 5 — `plot5_duplicate_transfer_vs_model`
**Caption.** Duplicate-screen transfer benchmark aggregated by coarse phenotype. For each merged duplicate screen in BioGRID v0.4, the plot compares the average cross-duplicate transfer AnDCG@100 computed from the raw v0.5 singleton screens against a chosen model’s cached per-example AnDCG@100 on that same merged screen.

## GRPO — `plot_grpo_staging_table`
**Caption.** Interim table of qwen3-30b-instruct-2507 and gpt-oss-120B training stages (base / SFT / GRPO). Replace with file-driven values via `--grpo-staging` when available.

---
*Missing-data audit:* see `missing_data_report.md`.
"""
    if report.used_grpo_interim:
        text += "\n(GRPO staging used **interim** built-in table.)\n"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(text)


def write_report_and_captions(
    report: MissingDataReport,
    captions_path: Path,
    missing_report_path: Path,
) -> None:
    write_figure_captions(captions_path, report)
    missing_report_path.parent.mkdir(parents=True, exist_ok=True)
    missing_report_path.write_text(report.to_markdown())


def save_payload_metadata_json(payload: Dict[str, Any], path: Path) -> None:
    summary = {
        "schema_version": payload.get("schema_version"),
        "inputs": payload.get("inputs", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
