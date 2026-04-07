# Journal Figure Workflow

Build the cached figure inputs once:

```bash
uv run python generate_figures_data.py
```

Build the direct-metrics cache for Plot 1, Plot 4, and Plot 5 from the
harmonized prediction files in `/cv/data/braid/gnesys/datasets/screensQA/results`:

```bash
uv run python generate_results_cache.py
```

Then render any figure very quickly from that cache:

```bash
uv run python plot0_dataset_composition.py
uv run python plot0_dataset_task.py
uv run python plot1_whole_dataset.py
uv run python plot1_selected_methods.py --method "Classifier" --method "gemini-3-pro"
uv run python plot2_phenotype.py
uv run python plot2_phenotype_bar_plot_year.py
uv run python plot3_ensemble_val_vs_test.py
uv run python plot4_memorization_composite.py
uv run python plot5_duplicate_transfer_vs_model.py
uv run python plot_grpo_staging_table.py
```

`plot1_whole_dataset.py`, `plot1_selected_methods.py`,
`plot4_memorization_composite.py`, and `plot5_duplicate_transfer_vs_model.py`
now read from `generate_results_cache.py` by default. The other figure scripts
still read from `generate_figures_data.py`.

To refresh the cache and regenerate everything in one command:

```bash
uv run python generate_journal_figures.py --refresh-data
```

# Harmonized Prediction Export

To export a harmonized `dataset_name -> ranked genes` JSON file for each model,
run:

```bash
uv run python export_harmonized_model_predictions.py --overwrite
```

By default this writes one JSON file per model to:

```text
/cv/data/braid/gnesys/datasets/screensQA/results
```

It also writes a `harmonized_predictions_manifest.json` file in that directory.

Each model file stores records grouped by `dataset_name`, with metadata such as:
- `split`
- `split_layout`
- `example_key` when available
- `predicted_genes`
- `prediction_runs`
- `n_runs`

Useful variants:

```bash
uv run python export_harmonized_model_predictions.py --dry-run
uv run python export_harmonized_model_predictions.py --model gemini-3-pro --model Classifier
uv run python export_harmonized_model_predictions.py --model "Oracle kNN" --model "Embedding kNN"
uv run python export_harmonized_model_predictions.py --model "gemini-3-pro-fewshot-knn10"
uv run python export_harmonized_model_predictions.py --skip-baselines --skip-classifier
```

The exporter currently covers:
- canonical LLM prediction directories
- few-shot LLM prediction directories under `fewshot_predictions` (exported under their plain model names, for example `gemini-3-pro-fewshot-knn10`)
- baseline prediction directories
- reconstructed learned ensemble: `LLM RRF Ensemble`
- reconstructed kNN transfer baselines: `Oracle kNN` and `Embedding kNN`
- trained and staged models such as GRPO, C2S, SFT, and SFT + GRPO
- the classifier predictions from the result CSVs

There is also a standalone kNN-only exporter:

```bash
uv run python export_harmonized_knn_predictions.py --overwrite
```
