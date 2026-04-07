from __future__ import annotations

import argparse
from pathlib import Path

from journal_figures_common import (
    DEFAULT_RESULTS_CACHE_PATH,
    OUTPUT_DIR,
    load_results_cache,
    plot5_duplicate_transfer_vs_model,
)

COMPARISON_MODEL = "gemini-3-pro"
ORACLE_KNN_MODEL = "Oracle kNN"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render Plot 5: duplicate-screen transfer vs a selected model plus "
            "Oracle kNN from the cached results data."
        ),
    )
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_RESULTS_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--model",
        type=str,
        default=COMPARISON_MODEL,
        help="Model name to compare against duplicate transfer alongside Oracle kNN.",
    )
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="Print all available models with cached adjusted_ndcg@100 rows for Plot 5 and exit.",
    )
    args = parser.parse_args()

    payload = load_results_cache(Path(args.cache_path))
    duplicate_transfer_df = payload["plot5_duplicate_transfer_df"]
    model_scores_df = payload["plot5_model_scores_df"]

    available_models = sorted(
        model_scores_df.loc[
            model_scores_df["metric"] == "adjusted_ndcg@100",
            "model",
        ]
        .dropna()
        .unique()
        .tolist()
    )
    if args.list_available:
        for model in available_models:
            print(model)
        return

    if args.model == ORACLE_KNN_MODEL:
        raise SystemExit(
            f"--model {ORACLE_KNN_MODEL!r} is not allowed because Oracle kNN is "
            "already included automatically in Plot 5."
        )

    if args.model not in set(available_models):
        raise SystemExit(
            f"Selected model not found in cache: {args.model}. "
            "Run with --list-available to inspect valid choices."
        )

    plot5_duplicate_transfer_vs_model(
        duplicate_transfer_df=duplicate_transfer_df,
        model_scores_df=model_scores_df,
        model_name=args.model,
        out_dir=Path(args.output_dir),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
