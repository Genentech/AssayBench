// AssayBench Pages — memorization analysis

(async function () {
  const data = await AssayBench.fetchJSON("assets/data/memorization.json");
  const points = data.points || [];
  const coeffs = data.coefficients || [];

  if (points.length === 0) {
    document.getElementById("scatter-plot").innerHTML = "<div class='error-state'>No memorization rows.</div>";
    return;
  }

  const PHENOTYPE_PALETTE = {
    "Fitness / Proliferation / Viability": "#E07A5F",
    "Drug / Chemical / Environmental Response": "#4C90D9",
    "Host-Pathogen / Infection Response": "#1FA187",
    "Molecular Output / Reporter / Pathway Activity": "#9C6ADE",
    "Trafficking / Localization / Structural Phenotypes": "#F28E2B",
    "Not specified": "#7E7E7E",
  };

  const byPhenotype = {};
  for (const p of points) {
    (byPhenotype[p.phenotype || "Not specified"] = byPhenotype[p.phenotype || "Not specified"] || []).push(p);
  }

  const traces = Object.keys(byPhenotype).map((phenotype) => {
    const arr = byPhenotype[phenotype];
    return {
      type: "scatter",
      mode: "markers",
      name: phenotype,
      x: arr.map((p) => p.log_citations),
      y: arr.map((p) => p.andcg100),
      marker: {
        color: PHENOTYPE_PALETTE[phenotype] || "#7E7E7E",
        size: 8,
        opacity: 0.78,
        line: { width: 0.5, color: "#fff" },
      },
      text: arr.map((p) => `${p.example_key} · year ${p.publication_year} · ${p.citation_count} citations`),
      hovertemplate: "<b>%{text}</b><br>AnDCG@100: %{y:.4f}<br>log(1+cites): %{x:.2f}<extra></extra>",
    };
  });

  // Simple OLS fit y = a + b * log_citations for the trend line.
  const xs = points.map((p) => p.log_citations);
  const ys = points.map((p) => p.andcg100);
  const xBar = xs.reduce((s, v) => s + v, 0) / xs.length;
  const yBar = ys.reduce((s, v) => s + v, 0) / ys.length;
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) {
    num += (xs[i] - xBar) * (ys[i] - yBar);
    den += (xs[i] - xBar) * (xs[i] - xBar);
  }
  const b = den > 0 ? num / den : 0;
  const a = yBar - b * xBar;
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const trendTrace = {
    type: "scatter",
    mode: "lines",
    name: `Trend: y = ${a.toFixed(3)} + ${b.toFixed(3)} · log(1+cites)`,
    x: [minX, maxX],
    y: [a + b * minX, a + b * maxX],
    line: { color: "#111827", width: 2, dash: "dot" },
    hoverinfo: "skip",
  };

  const scatterLayout = AssayBench.mergeLayout({
    height: 460,
    xaxis: { title: "log(1 + citation count)" },
    yaxis: { title: "Gemini 3 Pro AnDCG@100", rangemode: "tozero" },
    legend: { orientation: "h", y: -0.18 },
  });
  Plotly.react("scatter-plot", [...traces, trendTrace], scatterLayout, AssayBench.PLOTLY_CONFIG);

  // Coefficient bar chart
  if (coeffs.length === 0) {
    document.getElementById("coefficients-plot").innerHTML = "<div class='error-state'>No regression coefficients available.</div>";
  } else {
    const isYear = (term) => term === "year_c";
    const isCitations = (term) => term === "log_citations";
    const colorFor = (term) => isYear(term) ? "#4C90D9" : isCitations(term) ? "#C75146" : "#4daf4a";

    const sorted = [...coeffs].sort((a, b) => {
      if (a.term === "year_c") return -1;
      if (b.term === "year_c") return 1;
      if (a.term === "log_citations") return -1;
      if (b.term === "log_citations") return 1;
      return a.coef - b.coef;
    });

    const trace = {
      type: "bar",
      orientation: "h",
      y: sorted.map((c) => c.label),
      x: sorted.map((c) => c.coef),
      error_x: {
        type: "data",
        array: sorted.map((c) => c.se),
        thickness: 1.4,
        width: 5,
        color: "#374151",
      },
      marker: { color: sorted.map((c) => colorFor(c.term)) },
      hovertemplate: "<b>%{y}</b><br>coef: %{x:.4f}<br>se: %{error_x.array:.4f}<extra></extra>",
    };
    const layout = AssayBench.mergeLayout({
      height: Math.max(320, 30 * sorted.length + 80),
      margin: { l: 220, r: 32, t: 14, b: 50 },
      xaxis: { title: "Regression coefficient (±1 SE)", zeroline: true, zerolinewidth: 1.4, zerolinecolor: "#111" },
      yaxis: { automargin: true, autorange: "reversed", tickfont: { size: 11 } },
    });
    Plotly.react("coefficients-plot", [trace], layout, AssayBench.PLOTLY_CONFIG);
  }

  const r2El = document.getElementById("r2-line");
  if (r2El) {
    const parts = [];
    if (data.r_squared !== null && data.r_squared !== undefined) parts.push(`R² = ${data.r_squared.toFixed(3)}`);
    if (data.n !== undefined) parts.push(`n = ${data.n.toLocaleString()} screens`);
    r2El.textContent = parts.join(" · ");
  }
})();
