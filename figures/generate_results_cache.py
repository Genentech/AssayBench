from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from journal_figures_common import DEFAULT_RESULTS_CACHE_PATH, OUTPUT_DIR, PREDICTIONS_DIR
from results_cache_data import generate_and_save_results_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and cache the direct-metrics inputs for the journal figures "
            "from harmonized prediction files."
        ),
    )
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_RESULTS_CACHE_PATH))
    parser.add_argument("--results-dir", type=str, default=str(PREDICTIONS_DIR))
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Only rescore and update the specified model(s) in the existing results cache. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use when scoring harmonized prediction files.",
    )
    parser.add_argument(
        "--missing-report-out",
        type=str,
        default=str(OUTPUT_DIR / "results_cache_missing_data_report.md"),
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_path)
    results_dir = Path(args.results_dir)
    missing_report_out = Path(args.missing_report_out)
    selected_models = set(args.model) if args.model else None

    generate_and_save_results_cache(
        cache_path=cache_path,
        results_dir=results_dir,
        missing_report_out=missing_report_out,
        workers=max(1, args.workers),
        selected_models=selected_models,
    )
    print(f"Saved results cache to {cache_path}")
    print(f"Updated missing-data report at {missing_report_out}")


if __name__ == "__main__":
    main()
