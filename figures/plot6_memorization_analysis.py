from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from assaybench import AssayBenchDataset

from journal_figures_common import (
    DEFAULT_RESULTS_CACHE_PATH,
    OUTPUT_DIR,
    load_results_cache,
)

PLOT6_BASENAME = "plot6_memorization_analysis"
DEFAULT_CITATION_CACHE_PATH = Path(__file__).resolve().parent / "data" / "citation_count.json"


def extract_publication_year(author_field: str) -> int | None:
    match = re.search(r"\((\d{4})\)", str(author_field))
    return int(match.group(1)) if match else None


def _load_dataset_metadata() -> Dict[str, Dict[str, Any]]:
    ds = AssayBenchDataset(dataset_name="biogrid", split_type="year", fold=0)
    examples = ds.get_list_examples()
    meta: Dict[str, Dict[str, Any]] = {}
    for ex in examples:
        name = ex["dataset_name"]
        if name in meta:
            continue
        year = extract_publication_year(ex.get("author", ""))
        source_id = str(ex.get("source_id", ""))
        if source_id in ("", "Not specified", "nan"):
            source_id = None
        meta[name] = {"publication_year": year, "source_id": source_id}
    return meta


def build_plot6_analysis_dataframe(
    cache_path: Path,
    citation_cache_path: Path | None = None,
) -> pd.DataFrame:
    payload = load_results_cache(cache_path)
    scores_df = payload.get("plot5_model_scores_df")
    if not isinstance(scores_df, pd.DataFrame) or scores_df.empty:
        raise ValueError(f"No plot5_model_scores_df table found in {cache_path}")

    scores_df = scores_df.copy()
    scores_df = scores_df.loc[
        (scores_df["model"] == "gemini-3-pro")
        & (scores_df["metric"] == "adjusted_ndcg@100")
        & scores_df["example_key"].notna()
    ][["example_key", "dataset_name", "biogrid_phenotype", "value"]].rename(
        columns={"value": "metric_value"}
    )

    if scores_df.empty:
        raise ValueError("Cached results do not contain any gemini-3-pro adjusted_ndcg@100 rows")

    dataset_meta = _load_dataset_metadata()

    scores_df["publication_year"] = scores_df["dataset_name"].map(
        lambda n: dataset_meta.get(n, {}).get("publication_year")
    )
    scores_df["source_id"] = scores_df["dataset_name"].map(
        lambda n: dataset_meta.get(n, {}).get("source_id")
    )

    pmid_to_citations: Dict[str, Any] = {}
    if citation_cache_path is not None and citation_cache_path.exists():
        import json

        with open(citation_cache_path) as handle:
            pmid_to_citations = json.load(handle)

    scores_df["citation_count"] = scores_df["source_id"].map(
        lambda sid: pmid_to_citations.get(sid) if sid else None
    )

    # Screen 1686 has an incorrect author year (1970) in the source data
    scores_df.loc[scores_df["dataset_name"] == "1686", "publication_year"] = 2021

    scores_df = scores_df.dropna(subset=["publication_year", "citation_count"]).copy()
    scores_df["publication_year"] = scores_df["publication_year"].astype(int)
    scores_df["biogrid_phenotype"] = scores_df["biogrid_phenotype"].fillna("Not specified")
    scores_df["log_citations"] = np.log1p(scores_df["citation_count"].astype(float))

    screen_df = (
        scores_df.groupby(
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

    n_phenotypes = df["biogrid_phenotype"].nunique()
    if n_phenotypes > 1:
        formula = "metric_value ~ year_c + C(biogrid_phenotype) + log_citations"
    else:
        print(f"Warning: only {n_phenotypes} phenotype level(s) — dropping phenotype from regression")
        formula = "metric_value ~ year_c + log_citations"

    model = smf.ols(formula, data=df).fit()
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
    parser.add_argument(
        "--citation-cache",
        type=str,
        default=str(DEFAULT_CITATION_CACHE_PATH),
        help="Path to a JSON mapping PMIDs to citation counts.",
    )
    args = parser.parse_args()

    citation_cache_path = Path(args.citation_cache)
    screen_df = build_plot6_analysis_dataframe(
        cache_path=Path(args.cache_path),
        citation_cache_path=citation_cache_path,
    )
    render_plot6_memorization_analysis(screen_df, Path(args.output_dir), args.dpi)
    print(f"Saved Plot 6 to {Path(args.output_dir) / f'{PLOT6_BASENAME}.png'}")
    print(f"Saved summary to {Path(args.output_dir) / f'{PLOT6_BASENAME}_summary.txt'}")


if __name__ == "__main__":
    main()
