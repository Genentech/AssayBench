from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from journal_figures_common import (
    DATASET_PATH,
    DEFAULT_RESULTS_CACHE_PATH,
    OUTPUT_DIR,
    load_results_cache,
)


BIOGRID_INDEX_PATH = Path(
    "/cv/data/braid/wua33/gnesys/data/biogrid/"
    "BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt"
)
CITATION_CACHE_PATH = Path(
    "/cv/home/edwarc24/code/PromptOptBioGrid/promptoptbase/output/"
    "multi_model_ensemble/memorization_analysis/citation_cache.json"
)
PLOT6_BASENAME = "plot6_memorization_analysis"


def extract_screen_ids(dataset_name: str) -> List[int]:
    if dataset_name.startswith("U_TR_"):
        parts = dataset_name[5:].split("_")
        return [int(part) for part in parts[:-1] if part.isdigit()]
    if dataset_name.startswith("TR_"):
        parts = dataset_name[3:].split("_")
        return [int(part) for part in parts if part.isdigit()]
    if dataset_name.startswith("U_"):
        parts = dataset_name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return [int(parts[1])]
        return []
    try:
        return [int(dataset_name)]
    except ValueError:
        return []


def extract_publication_year(author_field: str) -> int | None:
    match = re.search(r"\((\d{4})\)", str(author_field))
    return int(match.group(1)) if match else None


def build_plot6_analysis_dataframe(cache_path: Path) -> pd.DataFrame:
    from datasets import load_from_disk
    from screensqa.utils.biogrid_maps import stratify_metrics_by_dataset_name

    payload = load_results_cache(cache_path)
    scores_df = payload.get("plot5_model_scores_df")
    if not isinstance(scores_df, pd.DataFrame) or scores_df.empty:
        raise ValueError(f"No plot5_model_scores_df table found in {cache_path}")

    scores_df = scores_df.copy()
    scores_df = scores_df.loc[(scores_df["model"]=="gemini-3-pro")
                              & (scores_df["metric"] == "adjusted_ndcg@100")
        & scores_df["example_key"].notna()][["example_key", "value"]].rename(columns={"value": "metric_value"})
    
    if scores_df.empty:
        raise ValueError("Cached results do not contain any val/test adjusted_ndcg@100 rows")

    dataset = load_from_disk(DATASET_PATH)
    rows: List[Dict[str, Any]] = []
    split_counters: Dict[str, int] = defaultdict(int)
    key_map = {"train": "train", "validation": "val", "test": "test"}
    for row in dataset:
        split = row["yearfold0"]
        index = split_counters[split]
        split_counters[split] += 1
        rows.append({
            "example_key": f"{key_map[split]}:{index}",
            "dataset_name": row["dataset_name"],
            "screen_ids": extract_screen_ids(row["dataset_name"]),
        })
    meta_df = pd.DataFrame(rows)

    phenotype_df = stratify_metrics_by_dataset_name({row["dataset_name"]: 0 for row in rows})
    phenotype_df = phenotype_df.rename(columns={"phenotype": "biogrid_phenotype"})
    phenotype_df.index.name = "dataset_name"
    phenotype_df = phenotype_df[["biogrid_phenotype"]].reset_index()
    meta_df = meta_df.merge(phenotype_df, on="dataset_name", how="left")


    biogrid_index = pd.read_csv(BIOGRID_INDEX_PATH, sep="\t")
    screen_id_to_year: Dict[int, int] = {}
    screen_id_to_pmid: Dict[int, str] = {}
    for _, row in biogrid_index.iterrows():
        screen_id = row["#SCREEN_ID"]
        year = extract_publication_year(row.get("AUTHOR", ""))
        if year is not None:
            screen_id_to_year[screen_id] = year
        pmid = str(row.get("SOURCE_ID", ""))
        if pmid and pmid != "nan":
            screen_id_to_pmid[screen_id] = pmid
    if 1686 in screen_id_to_year and screen_id_to_year[1686] == 1970:
        screen_id_to_year[1686] = 2021

    pmid_to_citations: Dict[str, Any] = {}
    if CITATION_CACHE_PATH.exists():
        import json

        with open(CITATION_CACHE_PATH) as handle:
            pmid_to_citations = json.load(handle)

    def get_year(screen_ids: List[int]) -> int | None:
        years = [screen_id_to_year[screen_id] for screen_id in screen_ids if screen_id in screen_id_to_year]
        return int(np.median(years)) if years else None

    def get_citations(screen_ids: List[int]) -> int | None:
        citations = [
            pmid_to_citations.get(screen_id_to_pmid.get(screen_id, ""))
            for screen_id in screen_ids
            if screen_id in screen_id_to_pmid
        ]
        citations = [citation for citation in citations if citation is not None]
        return int(np.median(citations)) if citations else None

    meta_df["publication_year"] = meta_df["screen_ids"].apply(get_year)
    meta_df = meta_df.dropna(subset=["publication_year"]).copy()
    meta_df["publication_year"] = meta_df["publication_year"].astype(int)
    meta_df["biogrid_phenotype"] = meta_df["biogrid_phenotype"].fillna("Not specified")
    meta_df["citation_count"] = meta_df["screen_ids"].apply(get_citations)
    meta_df["log_citations"] = np.log1p(meta_df["citation_count"].fillna(0).astype(float))

    analysis_df = scores_df.merge(meta_df, on="example_key", how="inner")
    if analysis_df.empty:
        raise ValueError("No overlap between cached model scores and metadata rows")

    screen_df = (
        analysis_df.groupby(
            ["example_key", "publication_year", "biogrid_phenotype", "citation_count", "log_citations"],
            as_index=False,
        )["metric_value"]
        .mean()
    )
    return screen_df


def _clean_term_label(term: str) -> str:
    if term == "year_c":
        return "Publication year"
    if term.startswith("C(biogrid_phenotype)[T.") and term.endswith("]"):
        return term[len("C(biogrid_phenotype)[T."):-1]
    return term


def _format_p_value(p_value: float) -> str:
    if p_value < 1e-300:
        return "<1e-300"
    return f"{p_value:.2e}"


def fit_simple_regression(screen_df: pd.DataFrame):
    import statsmodels.formula.api as smf

    df = screen_df.copy()
    df["year_c"] = df["publication_year"] - df["publication_year"].median()
    model = smf.ols("metric_value ~ year_c + C(biogrid_phenotype) + log_citations", data=df).fit()
    return model, df


def render_plot6_memorization_analysis(screen_df: pd.DataFrame, output_dir: Path, dpi: int) -> str:
    model, reg_df = fit_simple_regression(screen_df)

    rows = []
    for term, coef in model.params.items():
        if term == "Intercept":
            continue
        rows.append({
            "term": term,
            "label": _clean_term_label(term),
            "coef": float(coef),
            "std": float(model.bse[term]),
            "p_value": float(model.pvalues[term]),
        })
    coef_df = pd.DataFrame(rows)
    if coef_df.empty:
        raise ValueError("Regression produced no non-intercept coefficients")

    coef_df = pd.concat(
        [
            coef_df[coef_df["term"] == "year_c"],
            coef_df[coef_df["term"] != "year_c"].sort_values("coef"),
        ],
        ignore_index=True,
    )

    fig_height = max(4.5, 0.45 * len(coef_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    colors = ["#1f77b4" if term == "year_c" else "#4daf4a" for term in coef_df["term"]]
    y_pos = np.arange(len(coef_df))
    ax.barh(
        y_pos,
        coef_df["coef"],
        xerr=coef_df["std"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        capsize=4,
        alpha=0.9,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df["label"], fontsize=9)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_xlabel("Regression coefficient (+/- 1 SE)")
    ax.set_title("Performance ~ publication year + phenotype", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.margins(x=0.3)

    x_span = max((coef_df["coef"].abs() + coef_df["std"]).max(), 1e-6)
    for idx, row in coef_df.iterrows():
        x = row["coef"] + (0.03 * x_span if row["coef"] >= 0 else -0.03 * x_span)
        ha = "left" if row["coef"] >= 0 else "right"
        ax.text(
            x,
            idx,
            f"{row['coef']:+.4f} (p={_format_p_value(row['p_value'])})",
            va="center",
            ha=ha,
            fontsize=8,
            clip_on=False,
        )

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{PLOT6_BASENAME}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    present_phenotypes = {
        _clean_term_label(name)
        for name in model.params.index
        if name.startswith("C(biogrid_phenotype)[T.")
    }
    all_phenotypes = set(reg_df["biogrid_phenotype"].dropna().unique().tolist())
    missing_phenotypes = sorted(all_phenotypes - present_phenotypes)
    reference_phenotype = missing_phenotypes[0] if missing_phenotypes else "Not available"

    summary_lines = [
        "Plot 6: Simple memorization regression",
        "",
        "Model:",
        "  metric_value ~ publication_year + phenotype + log_citations",
        "",
        f"Screens used: {reg_df['example_key'].nunique()}",
        f"Publication year range: {reg_df['publication_year'].min()}-{reg_df['publication_year'].max()}",
        f"Phenotypes: {reg_df['biogrid_phenotype'].nunique()}",
        f"Screens with citation data: {int(reg_df['citation_count'].notna().sum())}",
        f"Reference phenotype: {reference_phenotype}",
        f"R^2: {model.rsquared:.4f}",
        f"Adjusted R^2: {model.rsquared_adj:.4f}",
        "",
        "Coefficients:",
    ]
    for _, row in coef_df.iterrows():
        summary_lines.append(
            f"  {row['label']}: coef={row['coef']:+.6f}, se={row['std']:.6f}, p={row['p_value']:.3e}"
        )
    summary = "\n".join(summary_lines)
    (output_dir / f"{PLOT6_BASENAME}_summary.txt").write_text(summary)
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a simple cache-backed regression of performance on year, phenotype, and log citations."
    )
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_RESULTS_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    screen_df = build_plot6_analysis_dataframe(Path(args.cache_path))
    render_plot6_memorization_analysis(screen_df, Path(args.output_dir), args.dpi)
    print(f"Saved Plot 6 to {Path(args.output_dir) / f'{PLOT6_BASENAME}.png'}")
    print(f"Saved summary to {Path(args.output_dir) / f'{PLOT6_BASENAME}_summary.txt'}")


if __name__ == "__main__":
    main()
