from __future__ import annotations

import argparse
from pathlib import Path

from journal_figures_common import (
    DEFAULT_RESULTS_CACHE_PATH,
    METRICS,
    OUTPUT_DIR,
    MissingDataReport,
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
    "baseline/random",
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


if __name__ == "__main__":
    main()
