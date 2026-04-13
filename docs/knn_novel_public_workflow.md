# Novel Public kNN Workflow

This note documents the helper script used to add the `novel_public_2026_dataset`
records to the harmonized kNN baseline files:

- `Oracle__kNN.json`
- `Embedding__kNN.json`

Both outputs live in:

```text
/cv/data/braid/gnesys/datasets/screensQA/results
```

## Script

The script is:

[`scripts/update_oracle_knn_novel_public.py`](/cv/home/debroue1/from_prescient/projects/screensqa_paper/scripts/update_oracle_knn_novel_public.py)

Despite the filename, it now supports both kNN baselines through:

```bash
--method oracle
--method embedding
```

The script writes harmonized prediction records with:

```json
{
  "split": "novel_public_dataset",
  "split_layout": "novel"
}
```

## What Each Mode Does

### `--method oracle`

This mode recomputes the oracle match for each screen in:

```text
/cv/data/braid/gnesys/datasets/screensQA/novel_public_2026_dataset/
```

using the `year`, fold-0 training pool from:

```text
/cv/data/braid/gnesys/datasets/screensQA/biogrid_v0.4_combined
```

For each novel public screen, it:

1. ranks each training screen’s genes by that training screen’s own relevance scores
2. evaluates which training screen transfers best to the target screen
3. stores the winning training screen’s top-100 genes as the prediction

The resulting records are written into `Oracle__kNN.json`.

### `--method embedding`

This mode does not recompute embedding matches from scratch.

Instead, it reuses the precomputed embedding kNN novel matches from:

```text
/cv/home/edwarc24/code/PromptOptBioGrid/promptoptbase/output_latent_biology/knn_test/year_fold0/knn_results.json
```

It then reconstructs the harmonized prediction records by:

1. reading the `embedding.novel.matched_train_indices`
2. mapping those indices back to the `year`, fold-0 training screens
3. using the matched training screen’s top-100 genes as the prediction

The resulting records are written into `Embedding__kNN.json`.

## Typical Commands

From the repo root:

```bash
cd /cv/home/debroue1/from_prescient/projects/screensqa_paper
```

Dry run:

```bash
uv run python scripts/update_oracle_knn_novel_public.py --method oracle --dry-run
uv run python scripts/update_oracle_knn_novel_public.py --method embedding --dry-run
```

Write Oracle:

```bash
uv run python scripts/update_oracle_knn_novel_public.py --method oracle
```

Write Embedding:

```bash
uv run python scripts/update_oracle_knn_novel_public.py --method embedding
```

Optional backup:

```bash
uv run python scripts/update_oracle_knn_novel_public.py --method oracle --backup
uv run python scripts/update_oracle_knn_novel_public.py --method embedding --backup
```

## Refreshing the Results Cache

After either harmonized JSON changes, rescore those methods into the paper cache:

```bash
uv run python generate_results_cache.py --model "Oracle kNN" --model "Embedding kNN"
```

You can optionally add workers:

```bash
uv run python generate_results_cache.py --model "Oracle kNN" --model "Embedding kNN" --workers 4
```

This updates the cached metrics used by the figure scripts without rebuilding the
entire cache for every model.

## Notes

- The harmonized kNN JSON files store predictions and match metadata, not just
  scalar metrics.
- `generate_results_cache.py` is the downstream scoring step. It should be run
  after changing the harmonized JSON files.
- The embedding updater depends on the legacy `year_fold0/knn_results.json`
  remaining available at the path above.
