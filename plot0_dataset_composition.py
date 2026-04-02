from __future__ import annotations

import argparse
from pathlib import Path

from journal_figures_common import (
    DEFAULT_CACHE_PATH,
    OUTPUT_DIR,
    load_figure_cache,
    plot_dataset_composition,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the dataset phenotype-composition pies and relevance-gene split bar plot from cached metadata.",
    )
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    payload = load_figure_cache(Path(args.cache_path))
    plot_dataset_composition(payload["meta_df"], Path(args.output_dir), args.dpi)


if __name__ == "__main__":
    main()
