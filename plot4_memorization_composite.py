from __future__ import annotations

import argparse
from pathlib import Path

from journal_figures_common import (
    DEFAULT_RESULTS_CACHE_PATH,
    OUTPUT_DIR,
    MissingDataReport,
    load_results_cache,
    plot4_memorization_composite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Plot 4 from the cached results data.")
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_RESULTS_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    payload = load_results_cache(Path(args.cache_path))
    report = MissingDataReport.from_dict(payload.get("report"))
    plot4_memorization_composite(payload["plot4_panels"], Path(args.output_dir), args.dpi, report)


if __name__ == "__main__":
    main()
