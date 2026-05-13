# AssayBench Pages site

This folder hosts the GitHub Pages site that lives at
`https://genentech.github.io/AssayBench/` (or wherever the fork's Pages is
configured). It is a pure static site — vanilla HTML + JavaScript + JSON,
with Plotly served from a CDN — so it deploys directly from this directory
on every push to `main`.

## Layout

```
docs/
├── .nojekyll                # disables Jekyll processing on GH Pages
├── _config.yml              # title/description fallback
├── index.html               # paper landing page
├── leaderboard.html         # Figure 2 (interactive)
├── phenotypes.html          # Figure 3 (interactive)
├── scaling.html             # Figure 4 left + optimization deltas
├── memorization.html        # Figure 4 right (citations vs perf)
├── bias.html                # Figure 5 (interactive)
├── screens.html             # per-screen / per-model drill-down
├── umap.html                # iframe wrapper around the public-only UMAP
├── build_data.py            # exports JSON for the pages from the results cache
├── build_public_umap.py     # public-only UMAP rebuild
├── build_figures.py         # PDF → PNG helper
└── assets/
    ├── css/site.css
    ├── js/site.js + per-page JS modules
    ├── data/                # generated JSON (see below)
    ├── figures/             # static PNG fallbacks for the landing page
    └── umap/                # generated public-only UMAP explorer + CSV
```

## Regenerate the data

The pages are powered by a small set of JSON files in `assets/data/`. All of
them are derived from the results cache produced by
`figures/generate_results_cache.py` plus the CSV/JSON files in
`figures/data/`.

```bash
# from the repo root, with the assaybench package installed:
uv sync           # or: pip install -e .

# optional: rebuild the results cache from raw predictions (slow)
python figures/generate_results_cache.py

# rebuild the JSON files consumed by the website
python docs/build_data.py \
    --cache-path figures/journal_figures_cache/results_cache.pkl
```

`build_data.py` writes ten JSON files into `assets/data/`:

| File | Purpose |
| --- | --- |
| `models.json` | display name, category, color, parameter hints for every method |
| `categories.json` | category list with palette colors |
| `leaderboard.json` | `(model, split_layout, cohort, metric, value)` long table |
| `phenotype_means.json` | per (phenotype, model, metric) means + screen counts |
| `per_screen.json` | per (model, screen, metric) score on the year split (compact) |
| `screens.json` | per-screen metadata (phenotype, split, genes, year, citations) |
| `bias.json` | parsed `bias_matrix.csv` (model × gene set) |
| `duplicate_transfer.json` | parsed `plot5_duplicate_transfer.csv` |
| `scaling.json` | Qwen3.5 scaling rows + optimization deltas |
| `memorization.json` | Gemini 3 Pro per-screen rows + regression coefficients |
| `summary.json` | small bundle (counts, top models, metric labels) for the landing page |

Total size is around 3 MB and the script runs in under two minutes (mostly
HuggingFace dataset metadata fetching).

## Refresh the public UMAP explorer

```bash
uv pip install umap-learn scikit-learn
python docs/build_public_umap.py \
    --transfer-matrix-dir /path/to/transfer_matrix
```

The script:

1. Loads `transfer_matrix.npy` + `screen_metadata.json` from the requested
   transfer-matrix directory.
2. Loads the public `AssayBenchDataset` (`biogrid` + `LaTest`).
3. Filters the matrix down to screens whose `dataset_name` is in the public
   set and asserts zero leakage.
4. Replaces every textual metadata field with the value from the public
   HuggingFace dataset.
5. Fits UMAP on the symmetric transfer-distance matrix and writes a
   self-contained Plotly HTML to `assets/umap/screen_umap_explorer.html`.

Pass `--transfer-matrix-dir` to point at any local transfer-matrix copy.

## Refresh the figure PNGs

The landing page falls back to static figure thumbnails if interactive views
fail to load. To rebuild them from the arXiv source:

```bash
uv pip install pymupdf
python docs/build_figures.py --source-dir /path/to/arxiv/figures
```

## Deploy on GitHub Pages

1. Push the `docs/` folder to `main`.
2. In the repo settings: **Pages → Build and deployment → Source: Deploy from
   a branch**, then **Branch: `main`**, **Folder: `/docs`**.
3. GitHub Pages will publish the site at
   `https://<org>.github.io/<repo>/`.

`.nojekyll` is already present so GH Pages serves the files as-is — no
Jekyll processing, no template engine.

## Local preview

```bash
cd docs
python -m http.server 8000
# then open http://localhost:8000
```

(The pages use `fetch()` for the JSON files, which requires an HTTP server;
opening `index.html` directly via `file://` will not work.)

## Conventions

- **Voice.** The landing page uses an accessible "press-release" tone for
  the headline and TL;DR, and switches to a paper-faithful voice for the
  abstract and figure captions. Per-page captions are direct quotes / light
  edits from the paper.
- **Colors.** Model categories use the same palette as
  `figures/journal_figures_common.py` (`CATEGORY_COLORS`). Phenotype
  palettes are defined inline in the per-page JS for the few pages that
  group by phenotype class.
- **Data freshness.** Re-run `build_data.py` whenever
  `figures/journal_figures_cache/results_cache.pkl` is refreshed.
