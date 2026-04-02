from __future__ import annotations

import argparse
from pathlib import Path

from journal_figures_common import DEFAULT_CACHE_PATH, OUTPUT_DIR, load_figure_cache, plot2_phenotype


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Plot 2 from the cached figure data.")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    payload = load_figure_cache(Path(args.cache_path))
    plot2_phenotype(
        payload["plot2_year_df"],
        payload["meta_df"],
        payload["plot2_rand_df"],
        payload["plot2_models_year"],
        payload["plot2_models_rand"],
        payload["plot2_metric"],
        Path(args.output_dir),
        args.dpi,
    )


if __name__ == "__main__":
    main()
