from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from journal_figures_common import (
    DEFAULT_RESULTS_CACHE_PATH,
    OUTPUT_DIR,
    load_results_cache,
    plot2_coarse_phenotype_bar_year,
)

SELECTED_METHODS = [
    #"Oracle rerank (combined)",
    "Oracle kNN",
    "LLM RRF Ensemble",
    "gemini-3-pro",
    #"fewshot/gemini-3-pro-fewshot-knn10",
    "gpt-5.4",
    #"gepa/gemini-3-flash",
    #"SFT + GRPO best (gpt-oss-120B)",
    "baseline/coarse-phenotype-hit-freq",
    #"SFT (gpt-oss-120B)",
    #"biomni-a1-claude-4",
    "Classifier",
    "Embedding kNN",
    #"C2S (Gemma-2B LoRA)",
    #"baseline/random"
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
        description="Render a year-split test bar plot by coarse phenotype for selected methods from the results cache.",
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
        help="Print all available year-test methods from the cache and exit.",
    )
    args = parser.parse_args()

    payload = load_results_cache(Path(args.cache_path))
    per_ex_year = payload["plot5_model_scores_df"]
    meta_df = payload.get("meta_df")
    available_methods = sorted(per_ex_year["model"].dropna().unique().tolist())

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

    print("Selected methods:")
    for method in selected_methods:
        print(method)

    plot2_coarse_phenotype_bar_year(
        per_ex_year=per_ex_year,
        meta_df=meta_df,
        models=selected_methods,
        out_dir=Path(args.output_dir),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
