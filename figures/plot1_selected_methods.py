from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from journal_figures_common import (
    DEFAULT_RESULTS_CACHE_PATH,
    METRICS,
    METRIC_LABELS,
    OUTPUT_DIR,
    MissingDataReport,
    display_model_name,
    load_results_cache,
    plot1_whole_dataset,
)

SELECTED_METHODS = [
    "gemini-3-pro",
    "gpt-5.4",
    "SFT + GRPO best (gpt-oss-120B)",
    "baseline/coarse-phenotype-hit-freq",
    "SFT (gpt-oss-120B)",
    "biomni-a1-claude-4",
    "Classifier",
    "gpt-oss-120b",
    "C2S (Gemma-2B LoRA)",
    "Oracle kNN",
    "Embedding kNN",
    "gepa/gemini-3-flash",
    "gemini-3-flash",
    "gemini-3-pro-fewshot-knn10",
    "LLM RRF Ensemble",
    "qwen3.5-2b"
]

COMPACT_TABLE_METRICS = (
    "adjusted_ndcg@100",
    "normalized_precision@100",
    "fdr@100",
)


def _read_methods_file(path: Path) -> list[str]:
    methods: list[str] = []
    for line in path.read_text().splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#"):
            methods.append(cleaned)
    return methods


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _format_metric_value(value: object) -> str:
    if value is None:
        return "--"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "--"
    if numeric != numeric:
        return "--"
    return f"{numeric:.4f}"


def _build_split_latex_table(
    filtered_df,
    split_layout: str,
    selected_methods: list[str],
    metrics: tuple[str, ...] = METRICS,
    *,
    caption_suffix: str = "across all metrics",
    label_suffix: str = "",
) -> str:
    import pandas as pd

    split_cohorts = {
        "year": ["val", "test", "novel"],
        "random": ["val", "test"],
    }
    cohort_order = split_cohorts[split_layout]

    split_df = filtered_df[filtered_df["split_layout"] == split_layout].copy()
    pivoted = split_df.pivot_table(
        index=["model", "cohort"],
        columns="metric",
        values="value",
        aggfunc="mean",
    )
    pivoted = pivoted.reindex(columns=list(metrics))

    index = pd.MultiIndex.from_product(
        [selected_methods, cohort_order],
        names=["model", "cohort"],
    )
    pivoted = pivoted.reindex(index)

    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{ll" + ("r" * len(metrics)) + "}",
        r"\toprule",
        "Method & Cohort & " + " & ".join(_latex_escape(METRIC_LABELS.get(metric, metric)) for metric in metrics) + r" \\",
        r"\midrule",
    ]

    previous_method = None
    for (method, cohort), row in pivoted.iterrows():
        if previous_method is not None and method != previous_method:
            lines.append(r"\midrule")
        values = [_format_metric_value(row.get(metric)) for metric in metrics]
        lines.append(
            _latex_escape(display_model_name(method))
            + " & "
            + _latex_escape(cohort)
            + " & "
            + " & ".join(values)
            + r" \\"
        )
        previous_method = method

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Selected Plot 1 methods on the {_latex_escape(split_layout)} split {_latex_escape(caption_suffix)}.}}",
        rf"\label{{tab:plot1-selected-methods-{split_layout}{_latex_escape(label_suffix)}}}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def _write_latex_tables(filtered_df, output_dir: Path, selected_methods: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_layout in ("year", "random"):
        table_specs = [
            {
                "filename": f"plot1_selected_methods_{split_layout}_table.tex",
                "metrics": METRICS,
                "caption_suffix": "across all metrics",
                "label_suffix": "",
            },
            {
                "filename": f"plot1_selected_methods_{split_layout}_table_compact_metrics.tex",
                "metrics": COMPACT_TABLE_METRICS,
                "caption_suffix": "for AnDCG@100, NPrecision@100, and FDR@100",
                "label_suffix": "-compact-metrics",
            },
        ]
        for spec in table_specs:
            table_text = _build_split_latex_table(
                filtered_df,
                split_layout,
                selected_methods,
                metrics=spec["metrics"],
                caption_suffix=spec["caption_suffix"],
                label_suffix=spec["label_suffix"],
            )
            table_path = output_dir / spec["filename"]
            table_path.write_text(table_text)
            print(f"Wrote LaTeX table: {table_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Plot 1 using only a selected list of methods from the cached results data.",
    )
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_RESULTS_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method name to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--methods-file",
        type=str,
        default=None,
        help="Optional text file with one method name per line.",
    )
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="Print all available Plot 1 methods from the cache and exit.",
    )
    args = parser.parse_args()

    payload = load_results_cache(Path(args.cache_path))
    plot_df = payload["plot1_df"]
    available_methods = sorted(plot_df["model"].dropna().unique().tolist())

    if args.list_available:
        for method in available_methods:
            print(method)
        return

    selected_methods = list(SELECTED_METHODS)
    if args.method:
        selected_methods = list(args.method)
    if args.methods_file:
        if args.method:
            selected_methods.extend(_read_methods_file(Path(args.methods_file)))
        else:
            selected_methods = _read_methods_file(Path(args.methods_file))
    selected_methods = _dedupe_keep_order(selected_methods)

    if not selected_methods:
        raise SystemExit(
            "No methods selected. Edit SELECTED_METHODS at the top of the script, "
            "pass --method, use --methods-file, or run with --list-available."
        )

    missing_methods = [method for method in selected_methods if method not in set(available_methods)]
    if missing_methods:
        missing_text = ", ".join(missing_methods)
        raise SystemExit(f"Selected methods not found in cache: {missing_text}")

    incomplete_methods = []
    for method in selected_methods:
        method_metrics = set(plot_df.loc[plot_df["model"] == method, "metric"].dropna().tolist())
        missing_metrics = [metric for metric in METRICS if metric not in method_metrics]
        if missing_metrics:
            incomplete_methods.append((method, missing_metrics))
    if incomplete_methods:
        details = "; ".join(
            f"{method} missing {', '.join(metrics)}"
            for method, metrics in incomplete_methods
        )
        print(
            "Warning: some selected methods do not yet have all Plot 1 metrics "
            f"in the current cache: {details}"
        )

    print("Selected methods:")
    for method in selected_methods:
        print(method)

    filtered_df = plot_df[plot_df["model"].isin(selected_methods)].copy()
    if filtered_df.empty:
        raise SystemExit("No rows left after filtering selected methods.")

    category_order = {method: idx for idx, method in enumerate(selected_methods)}
    filtered_df["_method_order"] = filtered_df["model"].map(category_order)
    filtered_df = (
        filtered_df.sort_values(["_method_order", "split_layout", "cohort", "metric"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )

    report = MissingDataReport.from_dict(payload.get("report"))
    plot1_whole_dataset(
        filtered_df,
        Path(args.output_dir),
        args.dpi,
        report,
        filename_stem="plot1_selected_methods",
        figure_title="Plot 1 — Selected methods",
        year_split_subtitle="year split",
        random_split_subtitle="random split",
    )
    _write_latex_tables(filtered_df, Path(args.output_dir), selected_methods)


if __name__ == "__main__":
    main()
