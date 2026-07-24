# AssayBench screen similarity analysis

Quantifies **biological similarity between the CRISPR screens** in the main AssayBench dataset
(`Genentech/assaybench`, config `biogrid`, 1,901 screens), groups them into clusters of
biologically similar assays, and names each cluster.

## Contents

- `screen_similarity_jaccard.ipynb` — the analysis (executed, with outputs).
- `results/`
  - `screen_cluster_assignments.csv` — one row per screen with its cluster + auto-name.
  - `cluster_summary.csv` — per-cluster size, dominant phenotype/tissue, characteristic genes, name.
  - `similarity_matrices.npz` — raw & IDF-weighted Jaccard matrices + labels *(git-ignored; regenerate by running the notebook)*.
- `figures/` — hit-set sizes, Jaccard distribution, cluster heatmap, phenotype composition, UMAP.

## Method

1. **Hit-gene sets** — genes with a non-zero relevance score per screen (sign = direction).
2. **Jaccard** of hit-gene sets via a sparse gene×screen incidence matrix (the starting point).
3. **IDF-weighted Jaccard** — down-weights ubiquitous core-essential genes that otherwise make
   all fitness screens look alike.
4. **Clustering** — Louvain community detection on a k-NN similarity graph (similarity is sparse,
   so hierarchical clustering chains into one blob; graph communities are the right tool).
5. **Naming** — dominant coarse phenotype + tissue + characteristic (within-cluster-enriched) genes.

Yields 36 coherent clusters, e.g. SARS-CoV-2 entry (ACE2/TMPRSS2), interferon/antigen
presentation (JAK2/STAT1/B2M), autophagy (ATG14/WIPI2/WDR45), CRL2 degron (FEM1C/CUL2),
glycolysis (PFKP/GOT1/G6PD), mitochondrial/OXPHOS fitness (NDUFA/MRPS).

## Run

```bash
cd analysis/assaybench_model_analysis
HF_HUB_OFFLINE=1 jupyter nbconvert --to notebook --execute --inplace screen_similarity_jaccard.ipynb
```

Seed fixed to 42. Tunables: `K_NN`, `RESOLUTION` (cluster granularity) in the clustering cell.
See the notebook's final section for caveats (hit-overlap only; library-composition confound)
and next steps (directional/score-aware similarity, pathway enrichment, LLM naming).
