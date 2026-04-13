# ScreensQA Paper — Figure Generation and Analysis

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

  scripts/                       # prediction collection, evaluation, and training
    collect_llm_predictions.py           # collect LLM predictions (Hydra)
    collect_fewshot_predictions.py       # collect few-shot kNN-augmented predictions
    generate_baseline_predictions.py     # generate non-LLM baselines (Hydra)
    evaluate_model_splits.py             # evaluate models across splits (Hydra)
    evaluate_best_ensemble.py            # evaluate best RRF ensemble trial
    run_ensemble_baseline.py             # single-model multi-run ensemble
    run_gepa_collect_predictions.py      # GEPA optimization + prediction collection
    run_gepa_optimization.py             # standalone GEPA optimization
    knn_test.py                          # kNN ranking (oracle / embedding)
    train_relevance_predictor_classifier.py  # gene relevance classifier (NOTE: placeholder, needs update)
    bollm_gene_embeddings.py             # BOLLM gene embedding loader
    download_gene_summaries.py           # download NCBI gene summaries for BM25
    shared_utils.py                      # shared loaders and utilities
    update_*_novel_public.py             # update specific model tracks with novel_public data

  configs/                       # Hydra YAML configs for runner scripts
    collect-predictions.yaml             # base template for LLM collection
    collect-<model>.yaml                 # per-model overrides (30 models)
    collect-fewshot-predictions.yaml     # few-shot collection config
    generate-baselines.yaml              # baseline generation config
    evaluate-model-splits.yaml           # evaluation config
    ensemble-baseline.yaml               # ensemble baseline config
    gepa/gemini-3-flash.yaml             # GEPA config for Gemini Flash

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

  docs/
    knn_novel_public_workflow.md
```

## Quick Start: Generating Journal Figures

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

## Collecting Predictions

### LLM predictions

Each model has a Hydra config in `configs/`. To collect predictions for a model:

```bash
uv run python scripts/collect_llm_predictions.py --config-name collect-gpt-5-mini
```

For local models served via vLLM, start the server first, then run:

```bash
uv run python scripts/collect_llm_predictions.py --config-name collect-deepseek-v3.2
```

### Few-shot predictions

```bash
uv run python scripts/collect_fewshot_predictions.py
```

### Non-LLM baselines

```bash
uv run python scripts/download_gene_summaries.py   # one-time: download NCBI gene summaries
uv run python scripts/generate_baseline_predictions.py
```

### GEPA predictions

```bash
uv run python scripts/run_gepa_collect_predictions.py \
    --config-path ../configs/gepa --config-name gemini-3-flash
```

### kNN predictions

```bash
uv run python scripts/knn_test.py --mode oracle
uv run python scripts/knn_test.py --mode embedding
```

### Evaluation

```bash
uv run python scripts/evaluate_model_splits.py
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

## Environment Setup

API keys should be provided via environment variables or a `.env` file:
- `AZURE_API_KEY` / `AZURE_API_BASE` — for Azure OpenAI models
- `GEMINI_API_KEY` — for Gemini models
- `ANTHROPIC_API_KEY` — for Claude models
- `PORTKEY_API_KEY` / `PORTKEY_API_BASE` — for Portkey-proxied models
- `AGENECY_API_KEY` / `BIOMNI_API_BASE` — for Biomni agent models

For local models, set `api_key: "token-abc123"` in the config (default placeholder for vLLM servers).
