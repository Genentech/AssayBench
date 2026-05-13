// Shared chrome + utilities for the AssayBench Pages site.

(function () {
  const NAV_LINKS = [
    { href: "index.html", label: "Overview", match: ["", "index.html"] },
    { href: "leaderboard.html", label: "Leaderboard", match: ["leaderboard.html"] },
    { href: "phenotypes.html", label: "Phenotypes", match: ["phenotypes.html"] },
    { href: "scaling.html", label: "Scaling", match: ["scaling.html"] },
    { href: "memorization.html", label: "Memorization", match: ["memorization.html"] },
    { href: "bias.html", label: "Bias", match: ["bias.html"] },
    { href: "screens.html", label: "Screens", match: ["screens.html"] },
    { href: "umap.html", label: "UMAP", match: ["umap.html"] },
    { href: "metric.html", label: "Metric", match: ["metric.html"] },
    { href: "cite.html", label: "Citation", match: ["cite.html"] },
  ];

  function currentSegment() {
    const path = window.location.pathname.split("/").pop() || "";
    return path;
  }

  function injectHeader() {
    const placeholder = document.getElementById("site-header");
    if (!placeholder) return;
    const current = currentSegment();
    placeholder.innerHTML = `
      <header class="site-header">
        <div class="site-header-inner">
          <a href="index.html" class="site-brand">
            AssayBench
            <span class="site-brand-sub">An Assay-Level Virtual Cell Benchmark</span>
          </a>
          <nav class="site-nav">
            ${NAV_LINKS.map((link) => {
              const active = link.match.includes(current) ? " class=\"active\"" : "";
              return `<a href=\"${link.href}\"${active}>${link.label}</a>`;
            }).join("")}
          </nav>
        </div>
      </header>
    `;
  }

  function injectFooter() {
    const placeholder = document.getElementById("site-footer");
    if (!placeholder) return;
    placeholder.innerHTML = `
      <footer class="site-footer">
        <div class="site-footer-inner">
          <div>AssayBench &nbsp;&middot;&nbsp; Genentech, 2026 &nbsp;&middot;&nbsp; <a href="https://arxiv.org/abs/2605.10876">arXiv</a>
            &nbsp;&middot;&nbsp; <a href="https://github.com/Genentech/AssayBench">GitHub</a>
            &nbsp;&middot;&nbsp; <a href="https://huggingface.co/datasets/Genentech/assaybench">HuggingFace dataset</a>
          </div>
          <div>Static site rendered from the public AssayBench results cache.</div>
        </div>
      </footer>
    `;
  }

  // ---------------------------------------------------------------- helpers
  async function fetchJSON(path) {
    const resp = await fetch(path);
    if (!resp.ok) throw new Error(`Failed to fetch ${path}: ${resp.status}`);
    return resp.json();
  }

  const PLOTLY_THEME = {
    font: { family: "Inter, Segoe UI, sans-serif", color: "#1b1f24", size: 12 },
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    margin: { l: 56, r: 24, t: 36, b: 64 },
    hoverlabel: { bgcolor: "#ffffff", bordercolor: "#d1d5db", font: { family: "Inter, sans-serif", size: 12, color: "#1b1f24" } },
    xaxis: { gridcolor: "rgba(128,128,128,0.18)", linecolor: "#d1d5db", zerolinecolor: "rgba(128,128,128,0.35)" },
    yaxis: { gridcolor: "rgba(128,128,128,0.18)", linecolor: "#d1d5db", zerolinecolor: "rgba(128,128,128,0.35)" },
    legend: { bgcolor: "rgba(255,255,255,0.7)", bordercolor: "#e5e7eb", borderwidth: 1 },
  };

  const PLOTLY_CONFIG = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
  };

  function mergeLayout(layout) {
    layout = layout || {};
    return Object.assign({}, PLOTLY_THEME, layout, {
      xaxis: Object.assign({}, PLOTLY_THEME.xaxis, layout.xaxis || {}),
      yaxis: Object.assign({}, PLOTLY_THEME.yaxis, layout.yaxis || {}),
      margin: Object.assign({}, PLOTLY_THEME.margin, layout.margin || {}),
      hoverlabel: Object.assign({}, PLOTLY_THEME.hoverlabel, layout.hoverlabel || {}),
      legend: Object.assign({}, PLOTLY_THEME.legend, layout.legend || {}),
    });
  }

  function indexBy(records, key) {
    const out = {};
    for (const record of records) out[record[key]] = record;
    return out;
  }

  function fmt(value, digits) {
    if (value === null || value === undefined || Number.isNaN(value)) return "\u2014";
    return Number(value).toFixed(digits === undefined ? 3 : digits);
  }

  function setActiveTag(container, value) {
    Array.from(container.querySelectorAll(".tag")).forEach((node) => {
      node.classList.toggle("active", node.dataset.value === value);
    });
  }

  function debounce(fn, ms) {
    let handle = null;
    return function (...args) {
      clearTimeout(handle);
      handle = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  document.addEventListener("DOMContentLoaded", () => {
    injectHeader();
    injectFooter();
  });

  window.AssayBench = {
    fetchJSON,
    mergeLayout,
    PLOTLY_CONFIG,
    indexBy,
    fmt,
    setActiveTag,
    debounce,
  };
})();
