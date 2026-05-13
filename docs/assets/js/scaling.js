// AssayBench Pages — scaling laws + optimization deltas

(async function () {
  const scaling = await AssayBench.fetchJSON("assets/data/scaling.json");

  function renderScaling() {
    const rows = (scaling.qwen35 || []).slice().sort((a, b) => a.param_billions - b.param_billions);
    if (!rows.length) {
      document.getElementById("scaling-plot").innerHTML = "<div class='error-state'>No Qwen3.5 rows in the cache.</div>";
      return;
    }
    const dense = rows.filter((r) => !r.is_moe);
    const moe = rows.filter((r) => r.is_moe);

    // Plot every model against its *total* parameter count. We split dense and
    // MoE into two traces only for styling: dense models share a solid line,
    // MoE models a dotted line, but both axes use total params so the scale is
    // directly comparable.
    const traceDense = {
      type: "scatter",
      mode: "lines+markers",
      name: "Dense",
      x: dense.map((r) => r.param_billions),
      y: dense.map((r) => r.mean_andcg100),
      marker: { size: 11, color: "#F58518", line: { color: "#fff", width: 1 } },
      line: { color: "#4C78A8", width: 2.6 },
      text: dense.map((r) => r.display_name),
      hovertemplate: "<b>%{text}</b><br>Total params: %{x}B<br>AnDCG@100: %{y:.4f}<extra></extra>",
    };
    const traceMoE = {
      type: "scatter",
      mode: "lines+markers",
      name: "Mixture-of-experts",
      x: moe.map((r) => r.param_billions),
      y: moe.map((r) => r.mean_andcg100),
      marker: { size: 11, color: "#1FA187", symbol: "diamond", line: { color: "#fff", width: 1 } },
      line: { color: "#1FA187", width: 2.6, dash: "dot" },
      text: moe.map((r) => `${r.display_name} (total ${r.param_billions}B, active ${r.active_params_b}B)`),
      hovertemplate: "<b>%{text}</b><br>Total params: %{x}B<br>AnDCG@100: %{y:.4f}<extra></extra>",
    };

    const annotations = rows.map((r) => ({
      x: r.param_billions,
      y: r.mean_andcg100,
      text: r.display_name.replace(/^Qwen3\.5-/, ""),
      showarrow: false,
      xanchor: "left",
      yanchor: "bottom",
      yshift: 10,
      xshift: 8,
      font: { family: "Inter, sans-serif", size: 10.5, color: "#374151" },
    }));

    const xValues = rows.map((r) => r.param_billions);
    const xMin = Math.max(0.3, Math.min(...xValues) * 0.6);
    const xMax = Math.max(...xValues) * 2.2;

    const layout = AssayBench.mergeLayout({
      height: 480,
      xaxis: {
        type: "log",
        title: "Parameter count (billions, log scale)",
        tickmode: "array",
        tickvals: [0.3, 1, 3, 10, 30, 100, 300, 1000],
        ticktext: ["0.3B", "1B", "3B", "10B", "30B", "100B", "300B", "1T"],
        tickangle: 0,
        ticks: "outside",
        showgrid: true,
        range: [Math.log10(xMin), Math.log10(xMax)],
      },
      yaxis: {
        title: "Mean AnDCG@100 (train + val + test)",
        rangemode: "tozero",
      },
      annotations,
      legend: { orientation: "h", y: -0.18 },
      margin: { l: 70, r: 40, t: 20, b: 80 },
    });

    Plotly.react("scaling-plot", [traceDense, traceMoE], layout, AssayBench.PLOTLY_CONFIG);
  }

  function renderOptimization() {
    // Cohort is fixed to "test" — the cache has no LaTest scores for any
    // optimized variants right now, so we don't expose the selector at all.
    const cohort = "test";
    const rows = scaling.optimization || [];
    if (!rows.length) {
      document.getElementById("optimization-plot").innerHTML = "<div class='error-state'>No optimization comparisons in the cache.</div>";
      return;
    }
    const valueFor = (r) => cohort === "test" ? r.test_andcg100 : r.novel_andcg100;

    // Group variants by base model and only keep groups where the *base*
    // model itself has a number for the chosen cohort. (LaTest is missing for
    // some models, in which case we drop the whole group rather than render
    // empty bars.)
    const groups = {};
    rows.forEach((r) => { (groups[r.base_model] = groups[r.base_model] || []).push(r); });
    const orderedGroups = Object.values(groups)
      .map((g) => {
        const base = g.find((x) => x.kind === "base");
        return { base, variants: g.filter((x) => x.kind !== "base") };
      })
      .filter((g) => g.base && valueFor(g.base) !== null && valueFor(g.base) !== undefined)
      .filter((g) => g.variants.some((v) => valueFor(v) !== null && valueFor(v) !== undefined));

    if (!orderedGroups.length) {
      document.getElementById("optimization-plot").innerHTML =
        `<div class='error-state'>No models in the cache have ${cohort === "test" ? "test" : "LaTest"} scores for both the base and an optimized variant.</div>`;
      return;
    }

    const COLORS = {
      base: "#9CA3AF",
      SFT: "#4C90D9",
      "SFT+GRPO": "#3B5DC9",
      GRPO: "#1FA187",
      GEPA: "#9C6ADE",
      "Few-shot": "#F28E2B",
      Variant: "#7E7E7E",
    };

    const yLabels = [];
    const traces = [];
    const seenKinds = new Set();

    const pushBar = (label, value, kind, hoverLabel, deltaText) => {
      yLabels.push(label);
      traces.push({
        type: "bar",
        orientation: "h",
        y: [label],
        x: [value],
        marker: { color: COLORS[kind] || "#7E7E7E" },
        name: kind === "base" ? "Base" : kind,
        showlegend: !seenKinds.has(kind),
        legendgroup: kind,
        text: deltaText ? [deltaText] : undefined,
        textposition: deltaText ? "outside" : undefined,
        cliponaxis: false,
        hovertemplate: `<b>${hoverLabel}</b><br>${cohort === "test" ? "Test" : "LaTest"} AnDCG@100: %{x:.4f}<extra></extra>`,
      });
      seenKinds.add(kind);
    };

    orderedGroups.forEach((group) => {
      const baseValue = valueFor(group.base);
      pushBar(`${group.base.label} (base)`, baseValue, "base", group.base.label, null);
      group.variants
        .filter((v) => valueFor(v) !== null && valueFor(v) !== undefined)
        .forEach((variant) => {
          const value = valueFor(variant);
          const delta = value - baseValue;
          const deltaText = (delta >= 0 ? "+" : "") + delta.toFixed(3);
          pushBar(variant.label, value, variant.kind, variant.label, deltaText);
        });
    });

    const layout = AssayBench.mergeLayout({
      height: Math.max(380, 36 * yLabels.length + 90),
      barmode: "group",
      margin: { l: 280, r: 80, t: 16, b: 50 },
      xaxis: { title: cohort === "test" ? "Test AnDCG@100" : "LaTest AnDCG@100", automargin: true },
      yaxis: { automargin: true, autorange: "reversed", tickfont: { size: 11 } },
      legend: { orientation: "h", y: -0.18, title: { text: "" } },
    });
    Plotly.react("optimization-plot", traces, layout, AssayBench.PLOTLY_CONFIG);
  }

  renderScaling();
  renderOptimization();
})();
