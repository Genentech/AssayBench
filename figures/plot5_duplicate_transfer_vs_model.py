from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from journal_figures_common import (
    DEFAULT_RESULTS_CACHE_PATH,
    OUTPUT_DIR,
    load_results_cache,
    plot5_duplicate_transfer_vs_model,
)

COMPARISON_MODEL = "gemini-3-pro"
ORACLE_KNN_MODEL = "Oracle kNN"

DUPLICATE_TRANSFER_CSV = Path(__file__).resolve().parent / "data" / "plot5_duplicate_transfer.csv"


def _load_duplicate_transfer_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "duplicate_id" in df.columns:
        df["duplicate_id"] = df["duplicate_id"].apply(ast.literal_eval)
    return df


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
        "--duplicate-transfer-csv",
        type=str,
        default=str(DUPLICATE_TRANSFER_CSV),
        help="Path to CSV with pre-computed duplicate-transfer scores.",
    )
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
    model_scores_df = payload["plot5_model_scores_df"]

    duplicate_transfer_path = Path(args.duplicate_transfer_csv)
    if not duplicate_transfer_path.exists():
        raise SystemExit(
            f"Duplicate transfer CSV not found: {duplicate_transfer_path}\n"
            "This file contains pre-computed cross-transfer scores between "
            "duplicate BioGRID screens and must be provided."
        )
    duplicate_transfer_df = _load_duplicate_transfer_df(duplicate_transfer_path)

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
