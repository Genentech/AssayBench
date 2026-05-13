// AssayBench Pages — per-screen drill-down

(async function () {
  const [models, screens, perScreen] = await Promise.all([
    AssayBench.fetchJSON("assets/data/models.json"),
    AssayBench.fetchJSON("assets/data/screens.json"),
    AssayBench.fetchJSON("assets/data/per_screen.json"),
  ]);

  const modelByKey = AssayBench.indexBy(models, "key");
  const screenByName = AssayBench.indexBy(screens, "dataset_name");

  const $ = (sel) => document.querySelector(sel);
  function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#39;",
    })[ch]);
  }

  const modeSelect = $("#ctrl-mode");
  const splitSelect = $("#ctrl-split");
  const screenSelect = $("#ctrl-screen");
  const modelSelect = $("#ctrl-model");
  const modelWrap = $("#ctrl-model-wrap");
  const searchInput = $("#ctrl-search");
  const metaPane = $("#screens-meta");

  const METRIC = "adjusted_ndcg@100";
  const SPLIT_LABEL = {
    train: "Train",
    val: "Validation",
    test: "Test",
    novel_public_dataset: "LaTest",
  };

  // ----------------------------------------------------------------- options
  function activeScreens() {
    const split = splitSelect.value;
    if (split === "all") return screens;
    return screens.filter((s) => s.split_year === split);
  }

  function buildScreenOptions(filterText) {
    const t = (filterText || "").toLowerCase();
    const filtered = activeScreens().filter((s) => {
      if (!t) return true;
      return (
        s.dataset_name.toLowerCase().includes(t) ||
        (s.biogrid_phenotype || "").toLowerCase().includes(t) ||
        (s.phenotype || "").toLowerCase().includes(t) ||
        (s.cell_line || "").toLowerCase().includes(t)
      );
    });
    screenSelect.innerHTML = filtered.slice(0, 1500).map((s) => {
      const phen = s.biogrid_phenotype ? ` · ${s.biogrid_phenotype}` : "";
      return `<option value="${escapeHTML(s.dataset_name)}">${escapeHTML(s.dataset_name)}${escapeHTML(phen)}</option>`;
    }).join("");
    if (filtered.length > 1500) {
      const o = document.createElement("option");
      o.disabled = true;
      o.textContent = `… ${filtered.length - 1500} more (refine search)`;
      screenSelect.appendChild(o);
    }
    // Update count badge next to the metadata pane via a tiny side-effect
    const countEl = document.getElementById("screens-count");
    if (countEl) countEl.textContent = `${filtered.length.toLocaleString()} screen${filtered.length === 1 ? "" : "s"} match the current filter`;
  }

  function buildModelOptions() {
    const present = new Set(perScreen.models);
    const ordered = models.filter((m) => present.has(m.key));
    modelSelect.innerHTML = ordered.map((m) => `<option value="${escapeHTML(m.key)}">${escapeHTML(m.display_name)}</option>`).join("");
  }

  buildScreenOptions("");
  buildModelOptions();
  if (screenByName["TR_71_72"]) screenSelect.value = "TR_71_72";
  if (perScreen.models.length) modelSelect.value = "gemini-3-pro";

  // ----------------------------------------------------------------- index
  const datasetIndex = {};
  perScreen.datasets.forEach((d, i) => { datasetIndex[d] = i; });
  const modelIndex = {};
  perScreen.models.forEach((m, i) => { modelIndex[m] = i; });
  const metricIndex = {};
  perScreen.metrics.forEach((m, i) => { metricIndex[m] = i; });

  function rowsForScreen(datasetName, metric) {
    const di = datasetIndex[datasetName];
    const mi = metricIndex[metric];
    if (di === undefined || mi === undefined) return [];
    return perScreen.rows
      .filter((r) => r[1] === di && r[2] === mi)
      .map((r) => ({ model: perScreen.models[r[0]], value: r[3] }));
  }

  function rowsForModel(modelKey, metric) {
    const mIdx = modelIndex[modelKey];
    const mi = metricIndex[metric];
    if (mIdx === undefined || mi === undefined) return [];
    return perScreen.rows
      .filter((r) => r[0] === mIdx && r[2] === mi)
      .map((r) => ({ dataset_name: perScreen.datasets[r[1]], value: r[3] }));
  }

  // ----------------------------------------------------------------- render
  function renderMeta(meta) {
    if (!meta) {
      metaPane.innerHTML = "<div class='error-state'>No metadata for that screen.</div>";
      return;
    }
    const rows = [
      ["Dataset name", meta.dataset_name],
      ["Year-split assignment", meta.split_year],
      ["Random-split assignment", meta.split_random || "—"],
      ["Phenotype class", meta.biogrid_phenotype],
      ["Phenotype (raw)", meta.phenotype],
      ["Screen type", meta.screen_type],
      ["Library methodology", meta.library_methodology],
      ["Cell type", meta.cell_type],
      ["Cell line", meta.cell_line],
      ["Library size", (meta.num_genes || 0).toLocaleString()],
      ["Author", meta.author],
      ["PMID / DOI", meta.source_id || "—"],
      ["Publication year", meta.publication_year || "—"],
      ["Citation count", meta.citation_count === null || meta.citation_count === undefined ? "—" : meta.citation_count],
    ];
    metaPane.innerHTML = `
      <h3>${escapeHTML(meta.dataset_name)}</h3>
      <p style="color: var(--ink-2); font-size: 13px; margin: 0 0 12px;">
        ${escapeHTML(meta.screen_rationale || "Public BioGRID ORCS screen.")}
      </p>
      <dl>
        ${rows.map(([k, v]) => `<dt>${escapeHTML(k)}</dt><dd>${escapeHTML(v)}</dd>`).join("")}
      </dl>
    `;
  }

  function renderByScreen() {
    const screenName = screenSelect.value;
    const meta = screenByName[screenName];
    renderMeta(meta);

    const rows = rowsForScreen(screenName, METRIC).sort((a, b) => b.value - a.value);
    const top = rows.slice(0, 40);
    const trace = {
      type: "bar",
      orientation: "h",
      y: top.map((r) => modelByKey[r.model]?.display_name || r.model),
      x: top.map((r) => r.value),
      marker: { color: top.map((r) => modelByKey[r.model]?.color || "#7E7E7E") },
      hovertemplate: "<b>%{y}</b><br>AnDCG@100: %{x:.4f}<extra></extra>",
    };
    const layout = AssayBench.mergeLayout({
      height: Math.max(420, 22 * top.length + 80),
      margin: { l: 220, r: 32, t: 14, b: 50 },
      xaxis: { title: "AnDCG@100", automargin: true, range: [0, Math.max(0.1, Math.max(...top.map((r) => r.value || 0)) * 1.1)] },
      yaxis: { automargin: true, autorange: "reversed" },
    });
    Plotly.react("screens-plot", [trace], layout, AssayBench.PLOTLY_CONFIG);
  }

  const PHENOTYPE_PALETTE = {
    "Fitness / Proliferation / Viability": "#E07A5F",
    "Drug / Chemical / Environmental Response": "#4C90D9",
    "Host-Pathogen / Infection Response": "#1FA187",
    "Molecular Output / Reporter / Pathway Activity": "#9C6ADE",
    "Trafficking / Localization / Structural Phenotypes": "#F28E2B",
    "Not specified": "#7E7E7E",
  };

  const PHENOTYPE_ORDER = [
    "Fitness / Proliferation / Viability",
    "Drug / Chemical / Environmental Response",
    "Host-Pathogen / Infection Response",
    "Molecular Output / Reporter / Pathway Activity",
    "Trafficking / Localization / Structural Phenotypes",
    "Not specified",
  ];

  function renderByModel() {
    const modelKey = modelSelect.value;
    const model = modelByKey[modelKey];
    const splitFilter = splitSelect.value;
    const splitTag = splitFilter === "all" ? "all year-split cohorts" : SPLIT_LABEL[splitFilter];

    const allowed = new Set(activeScreens().map((s) => s.dataset_name));
    const rows = rowsForModel(modelKey, METRIC)
      .filter((r) => allowed.has(r.dataset_name))
      .map((r) => {
        const meta = screenByName[r.dataset_name];
        return Object.assign(r, {
          phenotype: meta?.biogrid_phenotype || "Not specified",
        });
      });

    // Group by coarse phenotype and order rows by group median (best on top).
    const byPhenotype = {};
    rows.forEach((r) => {
      (byPhenotype[r.phenotype] = byPhenotype[r.phenotype] || []).push(r);
    });
    const medians = Object.fromEntries(
      Object.entries(byPhenotype).map(([k, v]) => {
        const sorted = v.map((x) => x.value).filter((x) => x !== null && x !== undefined).sort((a, b) => a - b);
        const m = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
        return [k, m];
      }),
    );
    const phenotypes = Object.keys(byPhenotype)
      .sort((a, b) => {
        const oa = PHENOTYPE_ORDER.indexOf(a);
        const ob = PHENOTYPE_ORDER.indexOf(b);
        if (oa !== -1 && ob !== -1) return oa - ob;
        return medians[b] - medians[a];
      });

    const traces = phenotypes.map((phenotype) => {
      const screens = byPhenotype[phenotype];
      const color = PHENOTYPE_PALETTE[phenotype] || "#7E7E7E";
      return {
        type: "box",
        name: phenotype,
        y: screens.map(() => phenotype),
        x: screens.map((r) => r.value),
        text: screens.map((r) => r.dataset_name),
        customdata: screens.map((r) => r.phenotype),
        boxpoints: "all",
        jitter: 0.55,
        pointpos: 0,
        fillcolor: color + "55",
        line: { color, width: 1.4 },
        marker: { color, size: 5, opacity: 0.75, line: { color: "#fff", width: 0.5 } },
        orientation: "h",
        showlegend: false,
        hovertemplate: "<b>%{text}</b><br>%{customdata}<br>AnDCG@100: %{x:.4f}<extra></extra>",
      };
    });

    metaPane.innerHTML = `
      <h3>${escapeHTML(model?.display_name || modelKey)}</h3>
      <p style="color: var(--ink-2); font-size: 13px; margin: 0 0 10px;">
        Each dot is one screen in <em>${escapeHTML(splitTag)}</em>. Rows are coarse phenotype classes; boxes summarize the
        distribution of AnDCG@100 within each class.
      </p>
      <dl>
        <dt>Category</dt><dd>${escapeHTML(model?.category_display || "")}</dd>
        ${model?.total_params_b ? `<dt>Total parameters</dt><dd>${model.total_params_b.toLocaleString()}B${model.is_moe ? " (MoE)" : ""}</dd>` : ""}
        ${model?.active_params_b ? `<dt>Active parameters</dt><dd>${model.active_params_b.toLocaleString()}B</dd>` : ""}
        <dt>Screens shown</dt><dd>${rows.length.toLocaleString()}</dd>
      </dl>
    `;

    const layout = AssayBench.mergeLayout({
      height: Math.max(480, 90 * phenotypes.length + 80),
      margin: { l: 240, r: 32, t: 18, b: 50 },
      xaxis: { title: "AnDCG@100", automargin: true, range: [0, Math.max(0.1, Math.max(...rows.map((r) => r.value || 0)) * 1.1)] },
      yaxis: { automargin: true, autorange: "reversed", type: "category", tickfont: { size: 12 } },
      boxgap: 0.35,
    });
    Plotly.react("screens-plot", traces, layout, AssayBench.PLOTLY_CONFIG);
  }

  // Some splits have sparse per-screen coverage in the cache (e.g. LaTest
  // currently only has the LLM ensemble). Surface that to avoid the page
  // looking broken.
  function coverageNote() {
    const split = splitSelect.value;
    if (split !== "novel_public_dataset") return "";
    const allowed = new Set(activeScreens().map((s) => s.dataset_name));
    const modelsForSplit = new Set();
    perScreen.rows.forEach((r) => {
      const ds = perScreen.datasets[r[1]];
      if (allowed.has(ds)) modelsForSplit.add(perScreen.models[r[0]]);
    });
    if (modelsForSplit.size <= 1) {
      return `<div class="callout" style="margin: 0 0 16px;">
        <strong>Note.</strong> Only ${modelsForSplit.size} model has per-screen predictions on LaTest in the
        public cache. See the leaderboard for the full LaTest comparison.
      </div>`;
    }
    return "";
  }

  function render() {
    const mode = modeSelect.value;
    modelWrap.style.display = mode === "by_model" ? "" : "none";
    document.querySelector('label[for="ctrl-screen"]').parentElement.style.display = mode === "by_screen" ? "" : "none";
    // Keep the screen dropdown in sync with the current year-split filter,
    // since the active list depends on it in both modes.
    if (mode === "by_screen") renderByScreen();
    else renderByModel();
    const noteHost = document.getElementById("screens-note");
    if (noteHost) noteHost.innerHTML = coverageNote();
  }

  modeSelect.addEventListener("change", render);
  screenSelect.addEventListener("change", render);
  modelSelect.addEventListener("change", render);
  splitSelect.addEventListener("change", () => {
    buildScreenOptions(searchInput.value);
    // If the previously selected screen falls outside the new split, pick a fallback.
    const allowed = new Set(activeScreens().map((s) => s.dataset_name));
    if (!allowed.has(screenSelect.value)) {
      const first = screenSelect.options[0];
      if (first && !first.disabled) screenSelect.value = first.value;
    }
    render();
  });
  searchInput.addEventListener("input", AssayBench.debounce((evt) => {
    buildScreenOptions(evt.target.value);
    render();
  }, 200));

  render();
})();
