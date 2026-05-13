"""Build the slim JSON data files used by the AssayBench Pages site.

This script reads the existing results cache produced by
``figures/generate_results_cache.py`` plus the auxiliary CSV/JSON files in
``figures/data/`` and writes the following web-friendly artifacts into
``docs/assets/data/``:

* ``models.json`` — display name / category / color / size hints for every
  model present in the results cache.
* ``leaderboard.json`` — per (model, split_layout, cohort, metric) value;
  feeds the leaderboard table and dot-and-line chart.
* ``phenotype_means.json`` — per (model, biogrid_phenotype, metric) mean on
  the year-split test cohort; feeds the phenotype heatmap.
* ``per_screen.json`` — per (model, dataset_name, metric) value on the year
  split, used by the per-screen drill-down (lazy-loaded by the page).
* ``screens.json`` — per-screen metadata (phenotype, split, num_genes,
  publication_year, citation_count).
* ``bias.json`` — long-form table parsed from ``bias_matrix.csv``.
* ``duplicate_transfer.json`` — long-form table parsed from
  ``plot5_duplicate_transfer.csv``.
* ``scaling.json`` — Qwen3.5 scaling rows + optimization deltas.
* ``memorization.json`` — Gemini 3 Pro per-screen rows joined with citation
  counts plus the OLS coefficients from the regression in ``plot4_…``.
* ``summary.json`` — small bundle (counts, top models, headline numbers)
  used by the landing page.

Usage::

    cd AssayBench
    python docs/build_data.py \
        --cache-path figures/journal_figures_cache/results_cache.pkl

Use ``--cache-path`` to point at any cache file (e.g. one that lives outside
this repository). All paths default to locations inside this repo.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_DIR / "docs"
DATA_OUT_DIR = DOCS_DIR / "assets" / "data"
FIGURES_DIR = REPO_DIR / "figures"

sys.path.insert(0, str(FIGURES_DIR))

# These imports require the ``assaybench`` package to be installed in the
# active environment (``pip install -e .`` from the repo root).
from journal_figures_common import (  # type: ignore  # noqa: E402
    CATEGORY_COLORS,
    CATEGORY_DISPLAY_NAMES,
    METHOD_DISPLAY_NAMES,
    METRIC_LABELS,
    METRICS,
    categorize_model,
    display_category_name,
    display_model_name,
)

# Model size hints for the bias / scaling pages — mirrors plot6_bias.MODEL_META.
MODEL_META: Dict[str, Dict[str, Any]] = {
    "qwen3.5-0.8b":          {"total_params_b": 0.8,  "active_params_b": 0.8,  "is_moe": False},
    "qwen3.5-2b":            {"total_params_b": 2.0,  "active_params_b": 2.0,  "is_moe": False},
    "qwen3-4b-2507":         {"total_params_b": 4.0,  "active_params_b": 4.0,  "is_moe": False},
    "qwen3.5-4b":            {"total_params_b": 4.0,  "active_params_b": 4.0,  "is_moe": False},
    "qwen3.5-9b":            {"total_params_b": 9.0,  "active_params_b": 9.0,  "is_moe": False},
    "gpt-oss-20b":           {"total_params_b": 20.0, "active_params_b": 20.0, "is_moe": False},
    "qwen3.5-27b":           {"total_params_b": 27.0, "active_params_b": 27.0, "is_moe": False},
    "qwen3-30b-a3b-2507":    {"total_params_b": 30.0, "active_params_b": 3.0,  "is_moe": True},
    "olmo-3.1-32b-think":    {"total_params_b": 32.0, "active_params_b": 32.0, "is_moe": False},
    "qwen3.5-35b-a3b":       {"total_params_b": 35.0, "active_params_b": 3.0,  "is_moe": True},
    "gpt-oss-120b":          {"total_params_b": 120.0, "active_params_b": 5.1, "is_moe": True},
    "qwen3-coder-next":      {"total_params_b": 80.0, "active_params_b": 80.0, "is_moe": True},
    "qwen3.5-122b-a10b":     {"total_params_b": 122.0, "active_params_b": 10.0, "is_moe": True},
    "MiniMax-M2.5":          {"total_params_b": 229.0, "active_params_b": 10.0, "is_moe": True},
    "qwen3-235b-a22b-2507":  {"total_params_b": 235.0, "active_params_b": 22.0, "is_moe": True},
    "qwen3.5-397b-a17b":     {"total_params_b": 397.0, "active_params_b": 17.0, "is_moe": True},
    "deepseek-v3.2":         {"total_params_b": 685.0, "active_params_b": 37.0, "is_moe": True},
    "deepseek-v3.2-nothink": {"total_params_b": 685.0, "active_params_b": 37.0, "is_moe": True},
    "GLM-5":                 {"total_params_b": 744.0, "active_params_b": 37.0, "is_moe": True},
    "Kimi-K2.5":             {"total_params_b": 1000.0, "active_params_b": 32.0, "is_moe": True},
}

# Site-specific category overrides. The upstream ``categorize_model`` helper
# lumps C2S (a biology-specific LLM) under the agent category alongside Biomni.
# On the website we want C2S to read as a biology-specific language model, so
# it shares the LLM color/family but with its own display label.
CATEGORY_OVERRIDES: Dict[str, Dict[str, str]] = {
    "C2S (Gemma-2B LoRA)": {
        "category": "Bio LLM",
        "category_display": "Biology-specific LLM",
        "color": "#5BA0B5",
    },
}

# Display-name fallbacks for models that aren't in METHOD_DISPLAY_NAMES. Used
# by ``_resolve_display_name`` below. Mirrors the override dict in
# ``figures/plot6_bias.py``.
EXTRA_DISPLAY_NAMES: Dict[str, str] = {
    "qwen3.5-0.8b":          "Qwen3.5-0.8B",
    "qwen3.5-4b":            "Qwen3.5-4B",
    "qwen3.5-9b":            "Qwen3.5-9B",
    "qwen3.5-27b":           "Qwen3.5-27B",
    "qwen3.5-35b-a3b":       "Qwen3.5-35B-A3B",
    "qwen3.5-122b-a10b":     "Qwen3.5-122B-A10B",
    "qwen3.5-397b-a17b":     "Qwen3.5-397B-A17B",
    "qwen3-4b-2507":         "Qwen3-4B-2507",
    "qwen3-30b-a3b-2507":    "Qwen3-30B-A3B-2507",
    "qwen3-235b-a22b-2507":  "Qwen3-235B-A22B-2507",
    "qwen3-coder-next":      "Qwen3-Coder Next",
    "gpt-oss-20b":           "GPT-OSS-20B",
    "gpt-oss-120b":          "GPT-OSS-120B",
    "deepseek-v3.2":         "DeepSeek v3.2",
    "deepseek-v3.2-nothink": "DeepSeek v3.2 (no-think)",
    "gemini-3.1-pro":        "Gemini 3.1 Pro",
    "gpt-5.2":               "GPT-5.2",
    "gpt-5-mini":            "GPT-5 mini",
    "claude-haiku-4.5":      "Claude Haiku 4.5",
    "claude-sonnet-4.5":     "Claude Sonnet 4.5",
    "claude-opus-4.5":       "Claude Opus 4.5",
    "olmo-3.1-32b-think":    "Olmo 3.1 32B (think)",
}


def _resolve_display_name(name: str) -> str:
    label = display_model_name(name)
    if label != name:
        return label
    return EXTRA_DISPLAY_NAMES.get(name, name)


def log(msg: str) -> None:
    print(f"[build_data {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _round(value: Any, digits: int = 5) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return float(round(numeric, digits))


def _qwen_param_billions(model_name: str) -> Optional[float]:
    match = re.match(r"^qwen3\.5-(\d+(?:\.\d+)?)b", model_name)
    if not match:
        return None
    return float(match.group(1))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _resolve_category(name: str) -> Dict[str, Any]:
    if name in CATEGORY_OVERRIDES:
        override = CATEGORY_OVERRIDES[name]
        return {
            "category": override["category"],
            "category_display": override["category_display"],
            "color": override["color"],
        }
    category = categorize_model(name)
    return {
        "category": category,
        "category_display": display_category_name(category),
        "color": CATEGORY_COLORS.get(category, "#7E7E7E"),
    }


def build_models_json(model_names: Iterable[str]) -> List[Dict[str, Any]]:
    records = []
    for name in sorted(set(model_names)):
        meta = MODEL_META.get(name, {})
        record: Dict[str, Any] = {
            "key": name,
            "display_name": _resolve_display_name(name),
            **_resolve_category(name),
        }
        if meta.get("total_params_b") is not None:
            record["total_params_b"] = float(meta["total_params_b"])
        if meta.get("active_params_b") is not None:
            record["active_params_b"] = float(meta["active_params_b"])
        if meta.get("is_moe") is not None:
            record["is_moe"] = bool(meta["is_moe"])
        records.append(record)
    return records


def build_categories_json() -> List[Dict[str, Any]]:
    rows = [
        {
            "key": key,
            "display": display_category_name(key),
            "color": CATEGORY_COLORS[key],
        }
        for key in CATEGORY_COLORS
    ]
    # Insert any site-specific categories that aren't in CATEGORY_COLORS.
    existing = {row["key"] for row in rows}
    for override in CATEGORY_OVERRIDES.values():
        if override["category"] not in existing:
            rows.append({
                "key": override["category"],
                "display": override["category_display"],
                "color": override["color"],
            })
            existing.add(override["category"])
    return rows


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def build_leaderboard_json(plot1_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in plot1_df.itertuples(index=False):
        rows.append({
            "model": record.model,
            "category": _resolve_category(record.model)["category"],
            "split_layout": record.split_layout,
            "cohort": record.cohort,
            "metric": record.metric,
            "value": _round(record.value),
        })
    return rows


# ---------------------------------------------------------------------------
# Phenotype means / per-screen
# ---------------------------------------------------------------------------

PHENOTYPE_METRICS = ("adjusted_ndcg@100", "precision@100", "fdr@100")
SITE_METRIC_LABELS = {
    "fdr@100": "dFDR@100",
    "normalized_fdr@100": "Normalized dFDR@100",
}


def build_phenotype_means_json(scores_df: pd.DataFrame) -> Dict[str, Any]:
    df = scores_df.copy()
    df = df[df["split"].isin(["train", "val", "test", "novel_public_dataset"])]
    df = df[df["metric"].isin(PHENOTYPE_METRICS)]
    df = df.dropna(subset=["biogrid_phenotype"])

    grouped = (
        df.groupby(["split", "metric", "model", "biogrid_phenotype"], as_index=False)
        .agg(value=("value", "mean"), n=("value", "size"))
    )

    counts = (
        df.dropna(subset=["example_key"])
        .drop_duplicates(subset=["example_key", "biogrid_phenotype", "split"])
        .groupby(["split", "biogrid_phenotype"], as_index=False)
        .agg(n_screens=("example_key", "nunique"))
    )

    rows: List[Dict[str, Any]] = []
    for record in grouped.itertuples(index=False):
        rows.append({
            "split": record.split,
            "metric": record.metric,
            "model": record.model,
            "phenotype": record.biogrid_phenotype,
            "value": _round(record.value),
            "n": int(record.n),
        })
    phenotype_counts: Dict[str, Any] = {}
    for record in counts.itertuples(index=False):
        phenotype_counts.setdefault(record.split, {})[record.biogrid_phenotype] = int(record.n_screens)
    return {"rows": rows, "phenotype_counts": phenotype_counts}


PER_SCREEN_METRICS = ("adjusted_ndcg@100", "precision@100")


def build_per_screen_json(scores_df: pd.DataFrame) -> Dict[str, Any]:
    df = scores_df.copy()
    df = df[df["metric"].isin(PER_SCREEN_METRICS)]
    df = df[df["split"].isin(["train", "val", "test", "novel_public_dataset"])]
    df = df.dropna(subset=["dataset_name"])

    metric_index = {metric: i for i, metric in enumerate(PER_SCREEN_METRICS)}
    model_keys = sorted(df["model"].dropna().unique().tolist())
    model_index = {model: i for i, model in enumerate(model_keys)}
    dataset_keys = sorted(df["dataset_name"].dropna().unique().tolist())
    dataset_index = {ds: i for i, ds in enumerate(dataset_keys)}

    # rows: [model_idx, dataset_idx, metric_idx, value]
    rows: List[List[Any]] = []
    for record in df.itertuples(index=False):
        rows.append([
            model_index[record.model],
            dataset_index[record.dataset_name],
            metric_index[record.metric],
            _round(record.value, 4),
        ])
    return {
        "metrics": list(PER_SCREEN_METRICS),
        "models": model_keys,
        "datasets": dataset_keys,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Screen metadata
# ---------------------------------------------------------------------------

_PUB_YEAR_RE = re.compile(r"\((\d{4})\)")


def _extract_publication_year(author: Any) -> Optional[int]:
    if not author:
        return None
    match = _PUB_YEAR_RE.search(str(author))
    return int(match.group(1)) if match else None


def build_screens_json(
    scores_df: pd.DataFrame,
    citation_count_path: Optional[Path],
) -> List[Dict[str, Any]]:
    from assaybench import AssayBenchDataset  # local import to avoid heavy import on --help

    metadata: Dict[str, Dict[str, Any]] = {}
    log("Loading public AssayBenchDataset for screen metadata ...")
    year_ds = AssayBenchDataset(
        dataset_name="biogrid",
        split_type="year",
        fold=0,
        novel_dataset_name="LaTest",
    )
    train, val, test, novel = year_ds.get_train_test_split()
    for split_name, examples in [
        ("train", train), ("val", val), ("test", test), ("novel_public_dataset", novel),
    ]:
        for ex in examples:
            name = str(ex["dataset_name"])
            if name not in metadata:
                metadata[name] = {
                    "dataset_name": name,
                    "split_year": split_name,
                    "biogrid_phenotype": ex.get("cleaned_phenotype") or "Not specified",
                    "phenotype": ex.get("phenotype") or "Not specified",
                    "screen_rationale": ex.get("screen_rationale") or "",
                    "screen_type": ex.get("screen_type") or "",
                    "library_methodology": ex.get("library_methodology") or "",
                    "cell_type": ex.get("cell_type") or "",
                    "cell_line": ex.get("cell_line") or "",
                    "num_genes": int(ex.get("num_genes") or 0),
                    "author": ex.get("author") or "",
                    "source_id": str(ex.get("source_id") or ""),
                    "publication_year": _extract_publication_year(ex.get("author")),
                }
                # Known correction from plot4_memorization_analysis.py
                if name == "1686":
                    metadata[name]["publication_year"] = 2021

    pmid_to_citations: Dict[str, Any] = {}
    if citation_count_path is not None and citation_count_path.exists():
        with open(citation_count_path) as handle:
            pmid_to_citations = json.load(handle)

    for record in metadata.values():
        sid = record.get("source_id")
        if sid:
            record["citation_count"] = pmid_to_citations.get(sid)
        else:
            record["citation_count"] = None

    # Annotate with random-split assignment if present in the cache.
    if "split_layout" in scores_df.columns:
        random_split_map = (
            scores_df[scores_df["split_layout"] == "random"]
            [["dataset_name", "split"]]
            .drop_duplicates()
        )
        for record in random_split_map.itertuples(index=False):
            name = str(record.dataset_name)
            if name in metadata:
                metadata[name]["split_random"] = record.split

    return [metadata[name] for name in sorted(metadata.keys())]


# ---------------------------------------------------------------------------
# Bias / duplicate transfer
# ---------------------------------------------------------------------------

def build_bias_json(bias_csv: Path) -> Dict[str, Any]:
    bias_df = pd.read_csv(bias_csv, index_col="model_name")
    rows: List[Dict[str, Any]] = []
    for model_name, row in bias_df.iterrows():
        for gene_set in bias_df.columns:
            rows.append({
                "model": model_name,
                "gene_set": gene_set,
                "value": _round(row[gene_set], 5),
            })
    return {
        "gene_sets": list(bias_df.columns),
        "models": list(bias_df.index),
        "rows": rows,
    }


def build_duplicate_transfer_json(csv_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    rows: List[Dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        phenotype = record.get("phenotype")
        if isinstance(phenotype, float) and math.isnan(phenotype):
            phenotype = None
        rows.append({
            "dataset_name": record["dataset_name"],
            "phenotype": phenotype,
            "id1": int(record["id1"]),
            "id2": int(record["id2"]),
            "andcg_1": _round(record.get("adjusted_ndcg@100_1")),
            "andcg_2": _round(record.get("adjusted_ndcg@100_2")),
            "duplicate_avg_andcg": _round(record.get("duplicate_avg_adjusted_ndcg@100")),
        })
    return rows


# ---------------------------------------------------------------------------
# Scaling / memorization
# ---------------------------------------------------------------------------

def build_scaling_json(scores_df: pd.DataFrame) -> Dict[str, Any]:
    df = scores_df.copy()
    df = df[(df["metric"] == "adjusted_ndcg@100") & (df["split"].isin(["train", "val", "test"]))]
    df = df[df["model"].str.match(r"^qwen3\.5-", na=False)]

    if df.empty:
        return {"qwen35": [], "optimization": []}

    df["param_billions"] = df["model"].map(_qwen_param_billions)
    df = df.dropna(subset=["param_billions"])

    grouped = (
        df.groupby(["model", "param_billions", "split"], as_index=False)["value"].mean()
    )
    overall = (
        df.groupby(["model", "param_billions"], as_index=False)["value"].mean()
    )

    by_model: Dict[str, Dict[str, Any]] = {}
    for record in overall.itertuples(index=False):
        meta = MODEL_META.get(record.model, {})
        by_model[record.model] = {
            "model": record.model,
            "display_name": _resolve_display_name(record.model),
            "param_billions": float(record.param_billions),
            "active_params_b": meta.get("active_params_b"),
            "is_moe": meta.get("is_moe", False),
            "mean_andcg100": _round(record.value),
            "per_split": {},
        }
    for record in grouped.itertuples(index=False):
        by_model[record.model]["per_split"][record.split] = _round(record.value)

    qwen_rows = sorted(by_model.values(), key=lambda r: r["param_billions"])

    # Optimization-delta bars: compare base vs SFT/GRPO/few-shot/GEPA variants.
    target_models = {
        "gpt-oss-120b": ["SFT (gpt-oss-120B)", "SFT + GRPO best (gpt-oss-120B)"],
        "qwen3-30b-a3b-2507": ["GRPO (qwen3-30b-instruct-2507)"],
        "gemini-3-flash": ["gepa/gemini-3-flash", "fewshot/gemini-3-flash-fewshot-knn10"],
        "gemini-3-pro": ["gemini-3-pro-fewshot-knn10"],
    }
    opt_rows: List[Dict[str, Any]] = []
    test_scores = (
        scores_df[(scores_df["metric"] == "adjusted_ndcg@100") & (scores_df["split"] == "test")]
        .groupby("model", as_index=False)["value"].mean()
    )
    novel_scores = (
        scores_df[(scores_df["metric"] == "adjusted_ndcg@100") & (scores_df["split"] == "novel_public_dataset")]
        .groupby("model", as_index=False)["value"].mean()
    )
    test_map = dict(zip(test_scores["model"], test_scores["value"]))
    novel_map = dict(zip(novel_scores["model"], novel_scores["value"]))

    for base_model, variants in target_models.items():
        base_test = test_map.get(base_model)
        base_novel = novel_map.get(base_model)
        opt_rows.append({
            "base_model": base_model,
            "variant": base_model,
            "label": _resolve_display_name(base_model),
            "kind": "base",
            "test_andcg100": _round(base_test),
            "novel_andcg100": _round(base_novel),
        })
        for variant in variants:
            v_test = test_map.get(variant)
            v_novel = novel_map.get(variant)
            kind = "SFT" if "SFT" in variant and "GRPO" not in variant else (
                "SFT+GRPO" if "SFT + GRPO" in variant else (
                "GRPO" if "GRPO" in variant else (
                "GEPA" if variant.startswith("gepa/") else (
                "Few-shot" if "fewshot" in variant else "Variant"))))
            opt_rows.append({
                "base_model": base_model,
                "variant": variant,
                "label": _resolve_display_name(variant),
                "kind": kind,
                "test_andcg100": _round(v_test),
                "novel_andcg100": _round(v_novel),
                "delta_test": _round((v_test or 0) - (base_test or 0)) if v_test is not None and base_test is not None else None,
                "delta_novel": _round((v_novel or 0) - (base_novel or 0)) if v_novel is not None and base_novel is not None else None,
            })

    return {"qwen35": qwen_rows, "optimization": opt_rows}


def build_memorization_json(
    scores_df: pd.DataFrame,
    citation_count_path: Optional[Path],
) -> Dict[str, Any]:
    from assaybench import AssayBenchDataset  # local import

    df = scores_df.copy()
    df = df[
        (df["model"] == "gemini-3-pro")
        & (df["metric"] == "adjusted_ndcg@100")
        & df["example_key"].notna()
    ][["example_key", "dataset_name", "biogrid_phenotype", "value"]].rename(
        columns={"value": "metric_value"}
    )
    if df.empty:
        return {"points": [], "coefficients": [], "r_squared": None}

    log("Loading dataset metadata for memorization regression ...")
    ds = AssayBenchDataset(dataset_name="biogrid", split_type="year", fold=0)
    examples = ds.get_list_examples()
    pub_year_map: Dict[str, Optional[int]] = {}
    source_id_map: Dict[str, Optional[str]] = {}
    for ex in examples:
        name = str(ex["dataset_name"])
        if name in pub_year_map:
            continue
        pub_year_map[name] = _extract_publication_year(ex.get("author"))
        sid = str(ex.get("source_id") or "")
        source_id_map[name] = sid if sid not in ("", "Not specified", "nan") else None

    df["publication_year"] = df["dataset_name"].map(pub_year_map)
    df["source_id"] = df["dataset_name"].map(source_id_map)

    pmid_to_citations: Dict[str, Any] = {}
    if citation_count_path is not None and citation_count_path.exists():
        with open(citation_count_path) as handle:
            pmid_to_citations = json.load(handle)
    df["citation_count"] = df["source_id"].map(
        lambda sid: pmid_to_citations.get(sid) if sid else None
    )
    df.loc[df["dataset_name"] == "1686", "publication_year"] = 2021
    df = df.dropna(subset=["publication_year", "citation_count"]).copy()
    df["publication_year"] = df["publication_year"].astype(int)
    df["biogrid_phenotype"] = df["biogrid_phenotype"].fillna("Not specified")
    df["log_citations"] = np.log1p(df["citation_count"].astype(float))

    screen_df = (
        df.groupby(
            ["example_key", "publication_year", "biogrid_phenotype", "citation_count", "log_citations"],
            as_index=False,
        )["metric_value"].mean()
    )

    points: List[Dict[str, Any]] = []
    for record in screen_df.itertuples(index=False):
        points.append({
            "example_key": record.example_key,
            "publication_year": int(record.publication_year),
            "phenotype": record.biogrid_phenotype,
            "citation_count": int(record.citation_count),
            "log_citations": _round(record.log_citations, 4),
            "andcg100": _round(record.metric_value, 4),
        })

    # Regression — keep it dependency-light (NumPy least squares).
    coefficients: List[Dict[str, Any]] = []
    r_squared: Optional[float] = None
    if len(screen_df) > 5:
        year_c = screen_df["publication_year"].values - np.median(screen_df["publication_year"].values)
        log_cit = screen_df["log_citations"].values

        phenotypes = sorted(screen_df["biogrid_phenotype"].unique().tolist())
        reference = phenotypes[0] if phenotypes else None
        design_columns: List[Dict[str, Any]] = [
            {"term": "Intercept", "vector": np.ones(len(screen_df))},
            {"term": "year_c", "vector": year_c.astype(float)},
            {"term": "log_citations", "vector": log_cit.astype(float)},
        ]
        for phenotype in phenotypes[1:]:
            indicator = (screen_df["biogrid_phenotype"] == phenotype).astype(float).values
            design_columns.append({"term": f"phenotype::{phenotype}", "vector": indicator})

        X = np.stack([col["vector"] for col in design_columns], axis=1)
        y = screen_df["metric_value"].values.astype(float)
        beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

        # SEs: sigma^2 * (X^T X)^{-1}
        dof = max(1, len(y) - rank)
        sigma2 = ss_res / dof
        try:
            inv_xt_x = np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.diag(sigma2 * inv_xt_x))
        except np.linalg.LinAlgError:
            se = np.full(len(beta), float("nan"))

        for column, coef, std in zip(design_columns, beta, se):
            term = column["term"]
            if term == "Intercept":
                continue
            if term == "year_c":
                label = "Publication year"
            elif term == "log_citations":
                label = "log(1 + citations)"
            elif term.startswith("phenotype::"):
                label = term.split("::", 1)[1]
            else:
                label = term
            coefficients.append({
                "term": term,
                "label": label,
                "coef": _round(coef, 6),
                "se": _round(std, 6),
            })

        if reference is not None:
            coefficients.insert(0, {
                "term": "phenotype::reference",
                "label": f"Reference phenotype: {reference}",
                "coef": 0.0,
                "se": 0.0,
            })

    return {
        "points": points,
        "coefficients": coefficients,
        "r_squared": _round(r_squared, 4),
        "n": len(points),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary_json(plot1_df: pd.DataFrame, scores_df: pd.DataFrame) -> Dict[str, Any]:
    def top_by(metric: str, cohort: str, split_layout: str = "year") -> List[Dict[str, Any]]:
        sub = plot1_df[
            (plot1_df["metric"] == metric)
            & (plot1_df["cohort"] == cohort)
            & (
                (plot1_df["split_layout"] == split_layout)
                | ((plot1_df["split_layout"] == "both") & (cohort == "novel"))
            )
        ].copy()
        sub = sub.dropna(subset=["value"])
        sub = sub.sort_values("value", ascending=False).head(5)
        return [
            {"model": r.model, "display": _resolve_display_name(r.model), "value": _round(r.value)}
            for r in sub.itertuples(index=False)
        ]

    summary = {
        "n_models": int(plot1_df["model"].nunique()),
        "n_screens": int(scores_df["dataset_name"].nunique()),
        "n_categories": int(plot1_df["category"].nunique()),
        "top_test_andcg100": top_by("adjusted_ndcg@100", "test"),
        "top_novel_andcg100": top_by("adjusted_ndcg@100", "novel"),
        "metrics": [{"key": k, "label": SITE_METRIC_LABELS.get(k, METRIC_LABELS[k])} for k in METRICS],
        "phenotypes": sorted(scores_df["biogrid_phenotype"].dropna().unique().tolist()),
    }
    return summary


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=None, separators=(",", ":"), ensure_ascii=False)
    path.write_text(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=REPO_DIR / "figures" / "journal_figures_cache" / "results_cache.pkl",
        help="Path to the results cache produced by generate_results_cache.py.",
    )
    parser.add_argument(
        "--bias-csv",
        type=Path,
        default=REPO_DIR / "figures" / "data" / "bias_matrix.csv",
    )
    parser.add_argument(
        "--duplicate-transfer-csv",
        type=Path,
        default=REPO_DIR / "figures" / "data" / "plot5_duplicate_transfer.csv",
    )
    parser.add_argument(
        "--citation-count-json",
        type=Path,
        default=REPO_DIR / "figures" / "data" / "citation_count.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_OUT_DIR,
    )
    args = parser.parse_args()

    cache_path = args.cache_path
    if not cache_path.exists():
        sys.stderr.write(
            f"Cache path not found: {cache_path}\n"
            "Run figures/generate_results_cache.py first or pass --cache-path.\n"
        )
        sys.exit(1)

    log(f"Loading cache from {cache_path}")
    with open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    plot1_df: pd.DataFrame = payload["plot1_df"]
    scores_df: pd.DataFrame = payload["plot5_model_scores_df"]
    log(f"Loaded plot1_df={plot1_df.shape}, plot5_model_scores_df={scores_df.shape}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = sorted(set(plot1_df["model"].unique()) | set(scores_df["model"].unique()))
    log(f"Writing models.json ({len(model_names)} models)")
    _save_json(build_models_json(model_names), out_dir / "models.json")
    _save_json(build_categories_json(), out_dir / "categories.json")

    log("Writing leaderboard.json")
    _save_json(build_leaderboard_json(plot1_df), out_dir / "leaderboard.json")

    log("Writing phenotype_means.json")
    _save_json(build_phenotype_means_json(scores_df), out_dir / "phenotype_means.json")

    log("Writing per_screen.json")
    per_screen = build_per_screen_json(scores_df)
    _save_json(per_screen, out_dir / "per_screen.json")

    log("Writing bias.json")
    _save_json(build_bias_json(args.bias_csv), out_dir / "bias.json")

    log("Writing duplicate_transfer.json")
    _save_json(build_duplicate_transfer_json(args.duplicate_transfer_csv), out_dir / "duplicate_transfer.json")

    log("Writing scaling.json")
    _save_json(build_scaling_json(scores_df), out_dir / "scaling.json")

    log("Writing memorization.json")
    _save_json(
        build_memorization_json(scores_df, args.citation_count_json),
        out_dir / "memorization.json",
    )

    log("Writing screens.json")
    _save_json(build_screens_json(scores_df, args.citation_count_json), out_dir / "screens.json")

    log("Writing summary.json")
    _save_json(build_summary_json(plot1_df, scores_df), out_dir / "summary.json")

    log("Done.")


if __name__ == "__main__":
    main()
