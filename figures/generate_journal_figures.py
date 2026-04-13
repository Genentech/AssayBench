"""
Generate all journal figures, using a cached precomputed payload whenever possible.

Typical usage:
  uv run python generate_figures_data.py
  uv run python plot1_whole_dataset.py
  uv run python generate_journal_figures.py --refresh-data
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import argparse
from pathlib import Path

from journal_figures_common import (
    DEFAULT_CACHE_PATH,
    OUTPUT_DIR,
    MissingDataReport,
    load_figure_cache,
    plot0_dataset_task,
    plot1_whole_dataset,
    plot2_phenotype,
    plot3_ensemble,
    plot4_memorization_composite,
    plot5_duplicate_transfer_vs_model,
    plot_grpo_staging_table,
    write_report_and_captions,
)
from journal_figures_data import generate_and_save_figure_cache


def _should_refresh_data(args: argparse.Namespace, cache_path: Path) -> bool:
    if args.refresh_data or not cache_path.exists():
        return True
    return any([
        args.no_knn,
        args.no_oracle,
        args.grpo_staging is not None,
        args.plot4_scaling_png is not None,
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all journal figures.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--no-knn", action="store_true")
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--grpo-staging", type=str, default=None)
    parser.add_argument("--plot4-scaling-png", type=str, default=None)
    parser.add_argument("--captions-out", type=str, default=None)
    parser.add_argument("--missing-report-out", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    cache_path = Path(args.cache_path)
    captions_out = Path(args.captions_out) if args.captions_out else out_dir / "figure_captions.md"
    missing_report_out = Path(args.missing_report_out) if args.missing_report_out else out_dir / "missing_data_report.md"

    if _should_refresh_data(args, cache_path):
        payload = generate_and_save_figure_cache(
            cache_path=cache_path,
            no_knn=args.no_knn,
            no_oracle=args.no_oracle,
            grpo_staging=args.grpo_staging,
            plot4_scaling_png=args.plot4_scaling_png,
            captions_out=captions_out,
            missing_report_out=missing_report_out,
        )
    else:
        payload = load_figure_cache(cache_path)

    report = MissingDataReport.from_dict(payload.get("report"))
    out_dir.mkdir(parents=True, exist_ok=True)

    plot0_dataset_task(payload["meta_df"], out_dir, args.dpi)
    plot1_whole_dataset(payload["plot1_df"], out_dir, args.dpi, report)
    plot2_phenotype(
        payload["plot2_year_df"],
        payload["meta_df"],
        payload["plot2_rand_df"],
        payload["plot2_models_year"],
        payload["plot2_models_rand"],
        payload["plot2_metric"],
        out_dir,
        args.dpi,
    )
    plot3_ensemble(payload["plot3_trials"], out_dir, args.dpi, report)
    plot4_memorization_composite(payload["plot4_panels"], out_dir, args.dpi, report)
    plot5_duplicate_transfer_vs_model(
        payload["plot5_duplicate_transfer_df"],
        payload["plot5_model_scores_df"],
        "gemini-3-pro",
        out_dir,
        args.dpi,
    )
    plot_grpo_staging_table(payload["grpo_rows"], out_dir, args.dpi)
    write_report_and_captions(report, captions_out, missing_report_out)

    print(f"Done. Outputs in {out_dir}")
    print(f"Cache in {cache_path}")


if __name__ == "__main__":
    main()
