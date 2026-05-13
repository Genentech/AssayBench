// AssayBench Pages — phenotype heatmap

(async function () {
  const [models, phenoData] = await Promise.all([
    AssayBench.fetchJSON("assets/data/models.json"),
    AssayBench.fetchJSON("assets/data/phenotype_means.json"),
  ]);
  const modelByKey = AssayBench.indexBy(models, "key");

  const PRESETS = {
    paper: [
      "gemini-3-pro", "gpt-5.4", "gemini-3-pro-fewshot-knn10", "gemini-3-flash",
      "biomni-a1-claude-4", "C2S (Gemma-2B LoRA)", "gpt-oss-120b",
      "SFT + GRPO best (gpt-oss-120B)", "qwen3.5-2b", "Embedding kNN",
      "Oracle kNN", "LLM RRF Ensemble", "baseline/coarse-phenotype-hit-freq", "baseline/random",
    ],
    frontier: [
      "gemini-3-pro", "gemini-3.1-pro", "gemini-3-flash",
      "gpt-5.4", "gpt-5.2", "gpt-5-mini",
      "claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5",
      "deepseek-v3.2", "GLM-5", "Kimi-K2.5", "MiniMax-M2.5",
    ],
    baselines: [
      "baseline/random", "baseline/global-hit-freq", "baseline/screen-type-hit-freq",
      "baseline/phenotype-knn-hit-freq", "baseline/library-size-prior",
      "baseline/phenotype-hit-freq", "baseline/coarse-phenotype-hit-freq",
      "baseline/bm25", "baseline/gene-name-overlap", "baseline/pagerank", "baseline/degree",
    ],
    trained: [
      "Classifier", "SFT (gpt-oss-120B)", "SFT + GRPO best (gpt-oss-120B)",
      "GRPO (qwen3-30b-instruct-2507)", "C2S (Gemma-2B LoRA)",
      "gepa/gemini-3-flash", "fewshot/gemini-3-flash-fewshot-knn10",
      "gemini-3-pro-fewshot-knn10", "LLM RRF Ensemble", "Oracle kNN", "Embedding kNN",
    ],
  };

  const $ = (sel) => document.querySelector(sel);
  const splitSelect = $("#ctrl-split");
  const presetSelect = $("#ctrl-preset");
  [splitSelect, presetSelect].forEach((el) => el.addEventListener("change", render));
  // Only AnDCG@100 has per-screen scores in the results cache.
  const METRIC = "adjusted_ndcg@100";

  const PHENOTYPE_ORDER = [
    "Fitness / Proliferation / Viability",
    "Drug / Chemical / Environmental Response",
    "Host-Pathogen / Infection Response",
    "Molecular Output / Reporter / Pathway Activity",
    "Trafficking / Localization / Structural Phenotypes",
    "Not specified",
  ];

  function activeModelKeys() {
    const preset = presetSelect.value;
    if (preset === "all") {
      // Use models that appear in phenoData rows.
      const present = new Set(phenoData.rows.map((r) => r.model));
      return models.map((m) => m.key).filter((k) => present.has(k));
    }
    return PRESETS[preset];
  }

  function render() {
    const metric = METRIC;
    const split = splitSelect.value;
    const modelKeys = activeModelKeys();
    const present = new Set();
    const lookup = {};
    for (const row of phenoData.rows) {
      if (row.split !== split || row.metric !== metric) continue;
      if (!modelKeys.includes(row.model)) continue;
      const key = `${row.model}|${row.phenotype}`;
      lookup[key] = row;
      present.add(row.model);
    }
    const orderedModels = modelKeys.filter((k) => present.has(k));

    const phenotypeCounts = (phenoData.phenotype_counts && phenoData.phenotype_counts[split]) || {};
    const yLabels = PHENOTYPE_ORDER
      .filter((p) => Object.keys(phenotypeCounts).includes(p) || orderedModels.some((m) => lookup[`${m}|${p}`]));

    if (yLabels.length === 0 || orderedModels.length === 0) {
      document.getElementById("phenotype-plot").innerHTML = "<div class='error-state'>No data for the selected combination.</div>";
      return;
    }

    const z = yLabels.map((phenotype) =>
      orderedModels.map((model) => {
        const row = lookup[`${model}|${phenotype}`];
        return row ? row.value : null;
      })
    );
    const hover = yLabels.map((phenotype) =>
      orderedModels.map((model) => {
        const row = lookup[`${model}|${phenotype}`];
        const display = modelByKey[model]?.display_name || model;
        const value = row ? row.value.toFixed(4) : "n/a";
        const n = row ? row.n : 0;
        return `<b>${display}</b><br>${phenotype}<br>${metric}: ${value}<br>n examples = ${n}`;
      })
    );

    const yTickText = yLabels.map((p) => {
      const n = phenotypeCounts[p];
      return n !== undefined ? `${p} (n=${n})` : p;
    });
    const xTickText = orderedModels.map((m) => modelByKey[m]?.display_name || m);
    const trace = {
      type: "heatmap",
      z,
      x: xTickText,
      y: yTickText,
      colorscale: "YlOrRd",
      hovertemplate: "%{customdata}<extra></extra>",
      customdata: hover,
      colorbar: { title: "AnDCG@100", thickness: 14 },
    };

    const layout = AssayBench.mergeLayout({
      margin: { l: 220, r: 32, t: 16, b: 160 },
      xaxis: {
        tickangle: -42,
        automargin: true,
        title: "",
      },
      yaxis: {
        automargin: true,
        title: "",
        autorange: "reversed",
      },
      height: Math.max(420, 60 * yLabels.length + 200),
    });
    Plotly.react("phenotype-plot", [trace], layout, AssayBench.PLOTLY_CONFIG);
  }

  render();
})();
