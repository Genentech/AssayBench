// AssayBench Pages — bias heatmap

(async function () {
  const [bias, models] = await Promise.all([
    AssayBench.fetchJSON("assets/data/bias.json"),
    AssayBench.fetchJSON("assets/data/models.json"),
  ]);
  const modelByKey = AssayBench.indexBy(models, "key");

  // value[gene_set][model_key] = bias
  const valueMap = {};
  for (const row of bias.rows) {
    (valueMap[row.gene_set] = valueMap[row.gene_set] || {})[row.model] = row.value;
  }

  const SIZE_ORDER = ["Small", "Medium", "Large", "Very Large", "Frontier"];
  function sizeBucket(meta) {
    if (!meta || meta.total_params_b === undefined || meta.total_params_b === null) return "Frontier";
    if (meta.total_params_b < 10) return "Small";
    if (meta.total_params_b < 50) return "Medium";
    if (meta.total_params_b < 250) return "Large";
    return "Very Large";
  }

  function sortBySize() {
    const items = bias.models.map((name) => {
      const meta = modelByKey[name];
      return {
        key: name,
        display_name: meta?.display_name || name,
        category: meta?.category_display || "—",
        size: sizeBucket(meta),
        total_params_b: meta?.total_params_b,
      };
    });
    items.sort((a, b) => {
      const sa = SIZE_ORDER.indexOf(a.size);
      const sb = SIZE_ORDER.indexOf(b.size);
      if (sa !== sb) return sa - sb;
      const pa = a.total_params_b || Number.POSITIVE_INFINITY;
      const pb = b.total_params_b || Number.POSITIVE_INFINITY;
      if (pa !== pb) return pa - pb;
      return a.key.localeCompare(b.key);
    });
    return items;
  }

  function render() {
    const ordered = sortBySize();
    const yLabels = ordered.map((m) => m.display_name);
    const xLabels = bias.gene_sets;
    const z = ordered.map((m) => xLabels.map((gs) => (valueMap[gs] || {})[m.key] ?? null));
    const flat = z.flat().filter((v) => v !== null && v !== undefined).map(Math.abs);
    const max = flat.length ? Math.max(...flat) : 1;

    const trace = {
      type: "heatmap",
      z,
      x: xLabels,
      y: yLabels,
      colorscale: "RdBu",
      reversescale: true,
      zmin: -max,
      zmax: max,
      hovertemplate: "<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>",
      // Push the colorbar to the far right so the size-bucket brackets and
      // labels can live cleanly between the heatmap and the colorbar.
      colorbar: {
        title: { text: "bias", side: "right" },
        x: 1.17,
        xanchor: "left",
        thickness: 10,
        len: 0.85,
      },
      xgap: 1,
      ygap: 1,
    };

    // Build a right-margin bracket per size bucket. Annotations sit between
    // the heatmap (x=1.0) and the colorbar (x~=1.17), with no overlap.
    const annotations = [];
    const shapes = [];
    const buckets = [];
    let start = 0;
    for (let i = 1; i <= ordered.length; i++) {
      if (i === ordered.length || ordered[i].size !== ordered[start].size) {
        buckets.push({ size: ordered[start].size, start, end: i - 1 });
        start = i;
      }
    }
    buckets.forEach((b) => {
      annotations.push({
        x: 1.045,
        xref: "paper",
        y: (b.start + b.end) / 2,
        yref: "y",
        text: `<b>${b.size}</b>`,
        xanchor: "left",
        yanchor: "middle",
        showarrow: false,
        font: { family: "Inter, sans-serif", size: 11, color: "#1b1f24" },
      });
      shapes.push({
        type: "line",
        xref: "paper",
        x0: 1.02,
        x1: 1.02,
        yref: "y",
        y0: b.start - 0.4,
        y1: b.end + 0.4,
        line: { color: "#374151", width: 1.4 },
      });
    });

    const layout = AssayBench.mergeLayout({
      height: Math.max(560, 22 * yLabels.length + 90),
      margin: { l: 200, r: 220, t: 56, b: 40 },
      xaxis: { side: "top", title: "", tickfont: { size: 12 } },
      yaxis: { automargin: true, autorange: "reversed", tickfont: { size: 11 } },
      annotations,
      shapes,
    });
    Plotly.react("bias-plot", [trace], layout, AssayBench.PLOTLY_CONFIG);
  }

  render();
})();
