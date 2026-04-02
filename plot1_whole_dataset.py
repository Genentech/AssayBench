from __future__ import annotations

import argparse
from pathlib import Path

from journal_figures_common import (
    DEFAULT_CACHE_PATH,
    OUTPUT_DIR,
    MissingDataReport,
    load_figure_cache,
    plot1_whole_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Plot 1 from the cached figure data.")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    payload = load_figure_cache(Path(args.cache_path))
    report = MissingDataReport.from_dict(payload.get("report"))
    plot1_whole_dataset(payload["plot1_df"], Path(args.output_dir), args.dpi, report)


if __name__ == "__main__":
    main()
