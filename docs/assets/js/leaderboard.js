// AssayBench Pages — leaderboard view

(async function () {
  const $ = (sel) => document.querySelector(sel);
  const LOWER_IS_BETTER = new Set(["fdr@100", "normalized_fdr@100"]);

  const [summary, models, categories, leaderboard] = await Promise.all([
    AssayBench.fetchJSON("assets/data/summary.json"),
    AssayBench.fetchJSON("assets/data/models.json"),
    AssayBench.fetchJSON("assets/data/categories.json"),
    AssayBench.fetchJSON("assets/data/leaderboard.json"),
  ]);

  const modelByKey = AssayBench.indexBy(models, "key");
  const categoryByKey = AssayBench.indexBy(categories, "key");

  // Populate metric select using order from summary.json
  const metricSelect = $("#ctrl-metric");
  summary.metrics.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.key;
    opt.textContent = m.label;
    metricSelect.appendChild(opt);
  });
  metricSelect.value = "adjusted_ndcg@100";

  const splitSelect = $("#ctrl-split");
  const cohortSelect = $("#ctrl-cohort");

  // Category filter chips
  const categoryEl = $("#category-filter");
  const activeCategories = new Set(categories.map((c) => c.key));
  categoryEl.innerHTML = categories.map((c) => `
    <span class="tag active" data-key="${c.key}">
      <span class="swatch" style="background:${c.color}"></span>${c.display}
    </span>
  `).join("");
  Array.from(categoryEl.querySelectorAll(".tag")).forEach((tag) => {
    tag.addEventListener("click", () => {
      const key = tag.dataset.key;
      if (activeCategories.has(key)) {
        activeCategories.delete(key);
        tag.classList.remove("active");
      } else {
        activeCategories.add(key);
        tag.classList.add("active");
      }
      render();
    });
  });

  $("#ctrl-reset").addEventListener("click", () => {
    activeCategories.clear();
    categories.forEach((c) => activeCategories.add(c.key));
    Array.from(categoryEl.querySelectorAll(".tag")).forEach((t) => t.classList.add("active"));
    metricSelect.value = "adjusted_ndcg@100";
    splitSelect.value = "year";
    cohortSelect.value = "test";
    sortKey = "test";
    sortDir = "desc";
    render();
  });

  [metricSelect, splitSelect, cohortSelect].forEach((el) => {
    el.addEventListener("change", () => {
      if (el === cohortSelect) {
        sortKey = cohortSelect.value;
      }
      render();
    });
  });

  let sortKey = "test";
  let sortDir = "desc";

  Array.from(document.querySelectorAll("#leaderboard-table thead th")).forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (sortKey === key) {
        sortDir = sortDir === "asc" ? "desc" : "asc";
      } else {
        sortKey = key;
        sortDir = LOWER_IS_BETTER.has(metricSelect.value) ? "asc" : "desc";
      }
      render();
    });
  });

  function buildRows() {
    const metric = metricSelect.value;
    const splitLayout = splitSelect.value;
    const ascending = LOWER_IS_BETTER.has(metric);

    const pivot = {};
    for (const row of leaderboard) {
      if (row.metric !== metric) continue;
      // LaTest scores are only defined under the year split (LaTest is a
      // temporal holdout), but the cohort is independent of the val/test split
      // choice, so surface novel rows under both split layouts.
      const allowed = row.split_layout === splitLayout || row.cohort === "novel";
      if (!allowed) continue;
      if (row.value === null || row.value === undefined) continue;
      if (!activeCategories.has(row.category)) continue;
      const entry = pivot[row.model] || (pivot[row.model] = { model: row.model, category: row.category });
      entry[row.cohort] = row.value;
    }

    const rows = Object.values(pivot).map((r) => {
      const model = modelByKey[r.model] || { display_name: r.model, color: "#7E7E7E", category_display: r.category };
      return Object.assign(r, {
        display_name: model.display_name,
        color: model.color,
        category_display: model.category_display,
      });
    });

    const cmp = (a, b) => {
      const va = a[sortKey];
      const vb = b[sortKey];
      if (typeof va === "string" || typeof vb === "string") {
        return (sortDir === "asc" ? 1 : -1) * String(va || "").localeCompare(String(vb || ""));
      }
      if (va === undefined || va === null) return 1;
      if (vb === undefined || vb === null) return -1;
      return (sortDir === "asc" ? 1 : -1) * (va - vb);
    };
    rows.sort(cmp);
    return rows;
  }

  function fmtSign(metric, value) {
    if (value === null || value === undefined) return "\u2014";
    const digits = metric.startsWith("fdr") || metric.startsWith("normalized_fdr") ? 3 : 3;
    return Number(value).toFixed(digits);
  }

  function renderTable(rows) {
    const tbody = document.querySelector("#leaderboard-table tbody");
    const metric = metricSelect.value;
    const indicators = {};
    Array.from(document.querySelectorAll("#leaderboard-table thead th")).forEach((th) => {
      indicators[th.dataset.key] = th.querySelector(".sort-indicator");
      th.querySelector(".sort-indicator").textContent = "";
    });
    if (indicators[sortKey]) {
      indicators[sortKey].textContent = sortDir === "asc" ? "▲" : "▼";
    }

    tbody.innerHTML = rows.map((r) => `
      <tr>
        <td class="left">
          <span class="model-pill"><span class="swatch" style="background:${r.color}"></span>${r.display_name}</span>
        </td>
        <td class="left"><span class="category-tag">${r.category_display}</span></td>
        <td>${fmtSign(metric, r.val)}</td>
        <td>${fmtSign(metric, r.test)}</td>
        <td>${fmtSign(metric, r.novel)}</td>
      </tr>
    `).join("");
  }

  function renderChart(rows) {
    const metric = metricSelect.value;
    const ascending = LOWER_IS_BETTER.has(metric);
    const sorted = rows.slice().sort((a, b) => {
      const va = a.test, vb = b.test;
      if (va === undefined || va === null) return 1;
      if (vb === undefined || vb === null) return -1;
      return ascending ? va - vb : vb - va;
    });

    const yLabels = sorted.map((r) => r.display_name);
    const colors = sorted.map((r) => r.color);

    const traceFor = (cohort, symbol, name) => ({
      type: "scatter",
      mode: "markers",
      name,
      y: yLabels,
      x: sorted.map((r) => r[cohort] !== undefined ? r[cohort] : null),
      marker: {
        symbol,
        size: 11,
        color: colors,
        line: { color: "#ffffff", width: 1 },
      },
      hovertemplate: `<b>%{y}</b><br>${name}: %{x:.4f}<extra></extra>`,
    });

    // Connecting lines between min and max cohort values per row
    const lines = {
      type: "scatter",
      mode: "lines",
      showlegend: false,
      hoverinfo: "skip",
      line: { color: "rgba(180,180,180,0.55)", width: 1 },
      x: [],
      y: [],
    };
    sorted.forEach((r) => {
      const vals = ["val", "test", "novel"].map((c) => r[c]).filter((v) => v !== undefined && v !== null);
      if (vals.length >= 2) {
        lines.x.push(Math.min(...vals), Math.max(...vals), null);
        lines.y.push(r.display_name, r.display_name, null);
      }
    });

    const labelText = summary.metrics.find((m) => m.key === metric)?.label || metric;
    const layout = AssayBench.mergeLayout({
      height: Math.max(420, 22 * sorted.length + 80),
      margin: { l: 220, r: 24, t: 14, b: 50 },
      xaxis: { title: labelText, automargin: true },
      yaxis: { automargin: true, ticks: "outside" },
      legend: { orientation: "h", x: 0, y: -0.08 },
    });

    Plotly.react(
      "leaderboard-plot",
      [
        lines,
        traceFor("val", "square", "Val"),
        traceFor("test", "circle", "Test"),
        traceFor("novel", "diamond-tall", "LaTest"),
      ],
      layout,
      AssayBench.PLOTLY_CONFIG,
    );
  }

  function render() {
    const rows = buildRows();
    renderTable(rows);
    renderChart(rows);
  }

  render();
})();
