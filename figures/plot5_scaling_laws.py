from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from journal_figures_common import (
    DEFAULT_RESULTS_CACHE_PATH,
    OUTPUT_DIR,
    load_results_cache,
    plot7_scaling_laws,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render Plot 7: Qwen3.5 scaling laws using mean AnDCG@100 across "
            "train, val, and test screens from the cached results data."
        ),
    )
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_RESULTS_CACHE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    payload = load_results_cache(Path(args.cache_path))
    plot7_scaling_laws(
        model_scores_df=payload["plot5_model_scores_df"],
        out_dir=Path(args.output_dir),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
