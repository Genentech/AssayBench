# ScreensQA Paper — Figure Generation

## Repository Structure

```
screensqa_paper/
  journal_figures_common.py      # shared constants, paths, plotting helpers
  journal_figures_data.py        # builds figure_data_cache.pkl
  results_cache_data.py          # builds results_cache.pkl from harmonized predictions

  figures/                       # all figure rendering scripts
    generate_figures_data.py     # CLI: build figure data cache
    generate_results_cache.py    # CLI: build results cache
    generate_journal_figures.py  # CLI: refresh cache + render all plots
    plot0_dataset_composition.py
    plot0_dataset_task.py
    plot1_whole_dataset.py
    plot1_selected_methods.py
    plot2_phenotype.py
    plot2_phenotype_bar_plot_year.py
    plot3_ensemble_val_vs_test.py
    plot4_memorization_composite.py
    plot5_duplicate_transfer_vs_model.py
    plot6_memorization_analysis.py
    plot7_scaling_laws.py
    plot_grpo_staging_table.py

  exporters/                     # harmonized prediction export pipelines
    export_harmonized_model_predictions.py
    export_harmonized_knn_predictions.py
    export_harmonized_ensemble_predictions.py

  predictions/                   # harmonized prediction JSONs (by model category)
    harmonized_predictions_manifest.json
    baselines/
    llm/
    trained/
    fewshot/
    knn/
    ensemble/
    gepa/
    classifier/

  scripts/                       # data update utilities
    shared_utils.py
    run_ensemble_baseline.py
    update_oracle_knn_novel_public.py
    update_gepa_novel_public.py
    update_fewshot_novel_public.py
    update_gpt_oss_120b_novel_public.py
    update_baseline_coarse_phenotype_novel_public.py

  docs/
    knn_novel_public_workflow.md
```

## Journal Figure Workflow

### Step 1: Build the figure data cache

```bash
uv run python figures/generate_figures_data.py
```

### Step 2: Build the results cache

Scores models from the harmonized prediction files in `predictions/`:

```bash
uv run python figures/generate_results_cache.py
```

### Step 3: Render individual figures

```bash
uv run python figures/plot0_dataset_composition.py
uv run python figures/plot0_dataset_task.py
uv run python figures/plot1_whole_dataset.py
uv run python figures/plot1_selected_methods.py --method "Classifier" --method "gemini-3-pro"
uv run python figures/plot2_phenotype.py
uv run python figures/plot2_phenotype_bar_plot_year.py
uv run python figures/plot3_ensemble_val_vs_test.py
uv run python figures/plot4_memorization_composite.py
uv run python figures/plot5_duplicate_transfer_vs_model.py
uv run python figures/plot6_memorization_analysis.py
uv run python figures/plot7_scaling_laws.py
uv run python figures/plot_grpo_staging_table.py
```

`plot1_whole_dataset.py`, `plot1_selected_methods.py`,
`plot4_memorization_composite.py`, and `plot5_duplicate_transfer_vs_model.py`
read from the **results cache**. The other figure scripts
read from the **figure data cache**.

### One-shot: refresh cache and regenerate everything

```bash
uv run python figures/generate_journal_figures.py --refresh-data
```

## Harmonized Predictions

The `predictions/` directory contains one JSON file per model, organized by
category (baselines, llm, trained, fewshot, knn, ensemble, gepa, classifier).
Gene lists are trimmed to the top 200 per screen.

Each model file stores records grouped by `dataset_name`, with fields:
- `split` — train, val, test, or novel_public_dataset
- `split_layout` — year or random
- `predicted_genes` — ranked gene list (up to 200)
- `prediction_runs` — per-run gene lists
- `n_runs`
- `example_key`

### Exporting new harmonized predictions

```bash
uv run python exporters/export_harmonized_model_predictions.py --overwrite
uv run python exporters/export_harmonized_knn_predictions.py --overwrite
uv run python exporters/export_harmonized_ensemble_predictions.py --overwrite
```
