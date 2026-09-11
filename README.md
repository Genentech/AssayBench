# AssayBench

A benchmark for evaluating machine learning models on phenotypic screen prediction.

[:globe_with_meridians: Website](https://genentech.github.io/AssayBench/) | [:octocat: Code](https://github.com/Genentech/AssayBench) | [:hugs: Dataset](https://huggingface.co/datasets/Genentech/assaybench) | [:page_with_curl: Paper](https://arxiv.org/abs/2605.10876) | [![PyPI](https://img.shields.io/pypi/v/assaybench)](https://pypi.org/project/assaybench/)

## 0. News

**September 2026 — [AssayLoop is now on arXiv](https://arxiv.org/abs/2609.11877).**
This follow-on work extends AssayBench to a lab-in-the-loop active learning framework with adaptive sequential hit discovery. 

[![Try it yourself](https://img.shields.io/badge/Try_it_yourself!-Run_AssayFormer_in_your_browser-176b87?style=for-the-badge)](https://genentech.github.io/AssayLoop/try.html)

**May 2026 —**
We released a [website](https://genentech.github.io/AssayBench/) with interactive data visualization!

[<img width="1352" height="675" alt="image" src="https://github.com/user-attachments/assets/74201853-1505-429f-af1d-0c3ad64065c7" />](https://genentech.github.io/AssayBench/)


## 1. Installation

```bash
pip install assaybench
```

Or install directly from the repository:

```bash
pip install git+ssh://git@github.com/Genentech/AssayBench.git
```

Or clone and install in editable mode:

```bash
git clone git@github.com:Genentech/AssayBench.git && cd AssayBench
pip install -e .
```

This installs the `assaybench` package, which provides:
- `AssayBenchDataset` — loads screens and splits from HuggingFace (`Genentech/assaybench`)
- `RankingMetrics` — computes ranking metrics (adjusted nDCG, precision, FDR, etc.)
- `assaybench.benchmark.sequential` — [sequential-acquisition metrics](#sequential-screens-assaybench-loop) (EF, adjusted nAUC, shortfall, %essential) for adaptive screens
- `assaybench.benchmark.effective_pathways` — Effective Pathways (EP-B, EP-S, EP-D) for biological diversity across batches, screens, and datasets
- `assaybench` (CLI) — [manages external data assets](#external-data-assets) that cannot be redistributed


### With `uv` 

You can add it to your project with
```
dependencies = [
    "assaybench @ git+ssh://git@github.com/Genentech/AssayBench.git",
]
```



## 2. Usage

### Loading data and scoring a model

Each example in the dataset contains a `question` prompt describing a CRISPR screen, along with ground-truth `relevance_genes` and `relevance_scores`. To evaluate a model, pass its predicted gene ranking (a plain `list[str]`) together with the ground-truth genes and scores to `RankingMetrics.evaluate()`:

```python
from assaybench import AssayBenchDataset
from assaybench.benchmark.metrics import RankingMetrics

# Load the dataset with year-based splits
ds = AssayBenchDataset(
    dataset_name="biogrid",
    split_type="year",
    fold=0,
    novel_dataset_name="LaTest",
)
train, val, test, latest = ds.get_train_test_split()

# Define your model — any function that returns a ranked list of gene names
def my_model(prompt: str) -> list[str]:
    return ["BRCA1", "TP53", "MYC", ...]  # top predicted genes

# Score predictions
metrics = RankingMetrics(k_values=[10, 100])

for example in val:
    predicted_genes = my_model(example["question"])
    scores = metrics.evaluate(
        predicted_genes=predicted_genes,
        ground_truth_genes=example["relevance_genes"],
        relevance_scores=example["relevance_scores"],
    )
    print(f"Screen {example['dataset_name']}: AnDCG@100 = {scores['adjusted_ndcg@100']:.4f}")
```

See [`examples/load_data.ipynb`](examples/load_data.ipynb) for a complete walkthrough.


### Dataset fields

Each screen returned by `get_train_test_split()` is a dictionary with the following fields:

| Field | Type | Description |
|---|---|---|
| `question` | str | The prompt describing the screen and ranking task |
| `relevance_genes` | list[str] | All genes in the screen library |
| `relevance_scores` | list[float] | Thresholded percentile scores for each gene (higher = more relevant) |
| `hit` | list[bool] | Whether each gene is a hit in the screen |
| `dataset_name` | str | Screen identifier |
| `screen_ids` | list[int] | BioGRID screen ID(s) (>1 for merged duplicate screens) |
| `phenotype` | str | Full phenotype description |
| `cleaned_phenotype` | str | Coarse phenotype category (e.g. "Fitness / Proliferation / Viability") |
| `condition_clause` | str | Experimental condition (e.g. drug treatment, dose) |
| `cell_type` | str | Cell type used in the screen |
| `cell_line` | str | Cell line name |
| `screen_type` | str | Selection type (e.g. "Positive Selection", "Negative Selection") |
| `library_methodology` | str | Screen methodology (e.g. "Knockout", "Activation") |
| `screen_rationale` | str | Scientific rationale for the screen |
| `screen_category` | str | Screen directionality (e.g. "unidirectional", "bidirectional") |
| `num_genes` | int | Number of genes in the screen library |
| `author` | str | Publication author and year (e.g. "Wang T (2014)") |
| `source_id` | str | PubMed ID of the source publication |
| `split` | str | Data split assignment: `train`, `validation`, `test`, or `novel_dataset` |
| `answer` | str | Top 10 genes by relevance score (comma-separated, for reference) |

### Metrics

`RankingMetrics.evaluate()` returns a dictionary of scores. The primary metrics (computed at each `k` in `k_values`) are:

| Metric | Description |
|---|---|
| `ndcg@k` | Normalized Discounted Cumulative Gain — measures ranking quality using graded relevance scores |
| `adjusted_ndcg@k` | nDCG adjusted for chance performance — the main benchmark metric (AnDCG) |
| `precision@k` | Fraction of top-k predictions that are hits |
| `normalized_precision@k` | Precision normalized by the number of true positives (NPrecision) |
| `fdr@k` | Fraction of top-k predictions that are non-hits (False Discovery Rate) |
| `normalized_fdr@k` | FDR normalized by the number of true negatives |
| `recall@k` | Fraction of true hits recovered in the top-k predictions |
| `auroc` | Area Under the ROC Curve over the full ranked list |
| `mrr` | Mean Reciprocal Rank — reciprocal of the rank of the first hit |
| `hallucination_rate` | Fraction of predicted genes not found in the screen library |
| `hit_scaled_ndcg@k` | nDCG computed using binary hit labels instead of graded relevance |
| `hit_scaled_adjusted_ndcg@k` | Adjusted nDCG using binary hit labels |

By default all metric groups are computed. Pass `metric_groups={"adjusted_ndcg", "precision"}` to restrict to a subset.

### Sequential screens (ASSAYBENCH-LOOP)

`RankingMetrics` scores a single ranked gene list. `assaybench.benchmark.sequential`
scores a *trajectory*: the genes a policy acquires over N rounds of an adaptive
screen, where each round's labels are revealed before the next round is
proposed. These are the metrics defined in
[AssayLoop](https://arxiv.org/abs/2609.11877) §3.3.

```python
from assaybench import adjusted_nauc, enrichment_factor, shortfall

rounds = [["TP53", "MYC", "KRAS"], ["BRCA1", "ATM"]]      # what the policy asked for
acquired = [gene for batch in rounds for gene in batch]

ef = enrichment_factor(acquired, library, hits, universe=hgnc_symbols, budget=100)
nauc = adjusted_nauc(rounds, library, hits, universe=hgnc_symbols)
sf = shortfall(acquired, library)
```

| Function | Metric | What it answers |
|---|---|---|
| `enrichment_factor` | EF | How many more hits than random, over the same effective budget. The headline number, and the only one comparable across screens with different hit rates. |
| `adjusted_nauc` | adjusted nAUC | Did it reach the hits *early*, or only by the last round? |
| `fraction_of_hits` | FH | Of everything there was to find, how much was found. |
| `shortfall` | SF | What fraction of picks produced no label at all. |
| `percent_essential` | %essential | How many of the hits found are common-essential genes — always hits, in any screen. |
| `batch_diversity` | Vendi, pathway overlap | Did the batch explore, or propose twelve subunits of one complex? |
| `effective_pathways` | EP-B, EP-S, EP-D | How many biological programs did the picks cover within batches, screens, and the pooled dataset? |

Effective Pathways is data-independent: pass the pathway vocabulary you intend
to use as a gene-to-pathways mapping. AssayBench does not bundle or silently
select a Reactome release.

```python
from assaybench import effective_pathways

membership = {"TP53": ("Cell Cycle",), "ATM": ("DNA Repair",)}
screen_batches = [[["TP53", "ATM"]]]  # screen -> batch -> genes
ep = effective_pathways(screen_batches, membership=membership)
```

Two conventions make EF comparable across screens. **Real genes outside the
screen library are forgiven** — no label exists for them, so scoring them
either way would measure library composition rather than the policy.
**Hallucinated symbols and unspent budget slots are charged** — otherwise a
policy games EF by naming three genes it is sure of and leaving 97 slots empty.
`classify_acquisitions` returns the full breakdown if you want to report the
terms separately.

`percent_essential` uses DepMap Public 26Q1 common-essential genes. The pinned
CSV is not included in the package and is never downloaded or hosted by
AssayBench. Users obtain it from DepMap; AssayBench locates it locally and
verifies its checksum before use. DepMap data is not covered by AssayBench's
MIT license and remains subject to the
[current DepMap Terms and Conditions](https://depmap.org/portal/terms/). See
`src/assaybench/data/depmap/PROVENANCE.md` and `THIRD_PARTY_NOTICES.md`.

### Running a policy: `assaybench.core`

The metrics above score a trajectory you already have. To *produce* one, use
the loop. `assaybench.core` is four abstract base classes and the driver that
turns them into a run:

| ABC | Implement it to supply | Key methods |
|---|---|---|
| `Task` | the screen: what can be picked, what happens when you pick it | `candidates()`, `reveal(batch)`, `ground_truth()` |
| `Model` | a belief over the unacquired candidates | `predict(observations, candidates)` |
| `AcquisitionFunction` | the choice of the next batch | `suggest(history, candidates, batch_size, ...)` |
| `Metric` | a per-step score | `score(observations, prediction, ground_truth, ...)` |

Model and acquisition are separate on purpose: an acquisition may consume the
model's belief (greedy, UCB) or ignore it entirely (random, or an LLM that
picks genes by name). Pair any acquisition with a do-nothing model to ablate.

```python
from assaybench import SequentialLoop

loop = SequentialLoop(task, model, acquisition, metrics, batch_size=100)
result = loop.run(n_steps=10)          # -> RunResult(history=[StepRecord], final_metrics={...})
```

Two behaviours are worth knowing before you trust a number out of it:

- **An under-supplied batch is never padded.** If an acquisition returns 40
  genes when asked for 100, the loop records 40 and charges the other 60 to
  `shortfall_frac`. Filling them with random picks would credit the policy
  with acquisitions it did not make. A run that under-supplies more than
  `max_shortfall_frac` of its slots is aborted and flagged in
  `final_metrics["error"]`, so an aggregate can drop it rather than average in
  a flat curve as though it were a genuine result.
- **Warm-start observations count.** A `Task` that pre-reveals data must
  return it from `initial_observations()`; the loop emits those as step 0, so
  the model, the acquisition and the metrics all see them from the start.

The loop itself does no I/O and never touches the network. If you need to
correlate whatever your model does internally — LLM calls, say — with the run,
pass a `trace_scope` context manager; it is called as
`trace_scope(run_id, sweep_id=..., task_id=...)` around the whole run.

[AssayLoop](https://arxiv.org/abs/2609.11877) is the reference implementation: a model zoo, LLM
acquisition policies and the paper's experiments, all built on these ABCs.

### External data assets

Pathway-overlap diversity scores against MSigDB, which is **not** bundled: its
C2 collection mixes CC BY 4.0 sets with KEGG sets under Kanehisa Laboratories'
terms and BioCarta sets held by qualified permission, none of which this
package is entitled to redistribute. Fetch it yourself, on the Broad's terms:

```bash
assaybench assets                    # what is registered, and under what licence
assaybench download msigdb-go-bp     # ~5 MB, into ~/.cache/assaybench
```

The DepMap Public 26Q1 common-essential list must be downloaded manually from
the [DepMap downloads page](https://depmap.org/portal/download/). Keep its
original filename, then point AssayBench at it:

```console
export ASSAYBENCH_DEPMAP_PATH=/absolute/path/CRISPRInferredCommonEssentials.csv
```

Alternatively, place it in the cache directory printed by `assaybench assets`.
When `percent_essential` first needs the file, a missing-file error repeats
these instructions. AssayBench never contacts DepMap for this asset.

All assets are checksum-pinned to the exact revisions behind the published
results, so a different release or corrupted file is an error rather than a
silent change in your numbers. Set `$ASSAYBENCH_CACHE` to relocate the cache.

Nothing here falls back to a different release. Downloads must match the
recorded checksum—a plausible number computed against substituted data is
worse than no number.

The embedding views (Vendi, cosine spread) take whatever gene-embedding space
you hand them, so no large download is implied:

```python
from assaybench import GeneEmbeddings, batch_diversity, load_gene_sets

embeddings = GeneEmbeddings.from_frame(frame, name="presage:GenePT_ada")
batch_diversity(batch, embeddings, load_gene_sets())
```

The published numbers use GenePT ada-002 vectors from the
[PRESAGE](https://github.com/Genentech/PRESAGE) cache, which is 3.4 GB and
therefore neither a dependency nor a download this package performs.

### Custom prompts

By default, `AssayBenchDataset` formats each screen's `question` field using a built-in prompt template (see `src/assaybench/data/prompts/objective_prompts.yaml`). You can override it by passing a `prompt_template` string to the constructor:

```python
my_template = """
You are a genetics expert. Given the following CRISPR screen:
- Cell line: {cell_line} ({cell_type})
- Library: {library_type} ({library_methodology})
- Phenotype: {phenotype}

Rank the top 100 genes most likely to be hits.
Format: GENE1, GENE2, ..., GENE100
"""

ds = AssayBenchDataset(
    dataset_name="biogrid",
    split_type="year",
    fold=0,
    prompt_template=my_template,
)
```

The template is formatted with Python's `str.format()` using each screen's metadata fields. Available placeholders:

| Placeholder | Description |
|---|---|
| `{cell_line}` | Cell line name |
| `{cell_type}` | Cell type description |
| `{library_type}` | Library type (e.g. "CRISPRn") |
| `{library_methodology}` | Methodology (e.g. "Knockout", "Activation") |
| `{experimental_setup}` | Experimental design (e.g. "Drug Exposure") |
| `{duration}` | Screen duration (e.g. "12 Days") |
| `{condition_clause}` | Condition details (e.g. " under Etoposide treatment (130.0 nM)") |
| `{phenotype}` | Phenotype description |
| `{significance_criteria}` | Statistical threshold for hit calling |
| `{ranking_rationale}` | What makes a gene rank highly |
| `{notes}` | Additional screen notes |

### Collecting LLM Results

Results from LLMs can be collected using [this script](benchmarking/predictions_generation/collect_llm_predictions.py); it uses DSPy and a couple additional instructions:
> Your goal is to provide a list of genes that meet the screen criteria, even if you do not have access to the actual experimental data. The genes must use HGNC symbols. Use your knowledge of biology, gene function, and relevant pathways to predict which genes are most likely to be hits. Do not refuse to answer or say you need more data—make your best predictions based on your understanding of the biological context.

Example Command: 
```python
uv run python benchmarking/predictions_generation/collect_llm_predictions.py --config-name=collect-GLM-5
```


## 3. Paper reproduction

All figure scripts live in `figures/` and read from a results cache built from the prediction files in `benchmarking/predictions/`.

### Step 1: Build the results cache

```bash
cd figures
python generate_results_cache.py
```

This scores all prediction files against the ground truth and saves the results to `figures/journal_figures_cache/results_cache.pkl`.

To rescore only specific models (faster):

```bash
python generate_results_cache.py --model "gemini-3-pro" --model "gpt-5.4"
```

### Step 2: Generate figures and tables

```bash
python plot0_proportions.py
python plot1_selected_methods.py
python plot2_phenotype_bar_plot_year.py
python plot3_duplicate_transfer_vs_model.py
python plot4_memorization_analysis.py
python plot5_scaling_laws.py
python plot6_bias.py
```

Outputs (PNG, PDF, LaTeX tables) are saved to `figures/journal_figures/`.

| Script | Description |
|---|---|
| `plot0_proportions.py` | Dataset statistics table and phenotype composition pie charts |
| `plot1_selected_methods.py` | Main benchmark bar plot + LaTeX tables for selected methods |
| `plot2_phenotype_bar_plot_year.py` | Per-phenotype performance bar plot (year split) |
| `plot3_duplicate_transfer_vs_model.py` | Duplicate-screen cross-transfer vs model performance |
| `plot4_memorization_analysis.py` | Regression of performance on publication year, phenotype, and citations |
| `plot5_scaling_laws.py` | Qwen3.5 scaling laws (AnDCG@100 vs model size) |
| `plot6_bias.py` | Gene-level prediction bias analysis across models |



## Citation
If you found our work useful, please cite:
```bibtex
@article{debrouwer2026assaybench,
  title={AssayBench: An Assay-Level Virtual Cell Benchmark for LLMs and Agents},
  author={De Brouwer, Edward and Edwards, Carl and Wu, Alexander and Collier, Jenna and Heimberg, Graham and Li, Xiner and Subramaniam, Meena and Hajiramezanali, Ehsan and Richmond, David and H{\"u}tter, Jan-Christian and others},
  journal={arXiv preprint arXiv:2605.10876},
  year={2026}
}
```
