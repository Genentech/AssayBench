from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import argparse
from pathlib import Path

from journal_figures_common import DEFAULT_CACHE_PATH, OUTPUT_DIR
from journal_figures_data import generate_and_save_figure_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and cache all precomputed inputs needed by the journal figure scripts.",
    )
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--no-knn", action="store_true")
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--grpo-staging", type=str, default=None)
    parser.add_argument("--plot4-scaling-png", type=str, default=None)
    parser.add_argument("--captions-out", type=str, default=str(OUTPUT_DIR / "figure_captions.md"))
    parser.add_argument("--missing-report-out", type=str, default=str(OUTPUT_DIR / "missing_data_report.md"))
    args = parser.parse_args()

    cache_path = Path(args.cache_path)
    captions_out = Path(args.captions_out)
    missing_report_out = Path(args.missing_report_out)

    generate_and_save_figure_cache(
        cache_path=cache_path,
        no_knn=args.no_knn,
        no_oracle=args.no_oracle,
        grpo_staging=args.grpo_staging,
        plot4_scaling_png=args.plot4_scaling_png,
        captions_out=captions_out,
        missing_report_out=missing_report_out,
    )
    print(f"Saved figure cache to {cache_path}")
    print(f"Updated captions at {captions_out}")
    print(f"Updated missing-data report at {missing_report_out}")


if __name__ == "__main__":
    main()
