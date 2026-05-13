"""Rebuild the screen UMAP explorer on the *public* AssayBench dataset.

This script starts from a local transfer matrix and rebuilds the public
metadata used by the static explorer:

1. Loads ``transfer_matrix.npy`` + ``screen_metadata.json`` from the requested
   transfer-matrix directory.
2. Loads the *public* dataset via ``AssayBenchDataset(dataset_name="biogrid")``
   and the ``LaTest`` novel split.
3. Keeps only screens whose ``dataset_name`` is in the public set (this should
   already be the whole transfer matrix, since the public dataset is the
   superset, but the assertion guarantees zero leakage).
4. Replaces every textual metadata field on each kept screen with the value
   from the public HuggingFace dataset.
5. Re-fits UMAP on the filtered submatrix and writes a self-contained
   Plotly HTML to ``docs/assets/umap/screen_umap_explorer.html`` plus a
   companion ``umap_coordinates.csv``.

Requires ``umap-learn`` and ``scikit-learn``. They are not in the AssayBench
runtime dependencies; install separately:

    uv pip install umap-learn scikit-learn

Usage::

    python docs/build_public_umap.py \\
        --transfer-matrix-dir /path/to/transfer_matrix \\
        --out-dir docs/assets/umap
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_DIR / "docs"
DEFAULT_OUT_DIR = DOCS_DIR / "assets" / "umap"
DEFAULT_TRANSFER_DIR = Path("output") / "transfer_matrix"


def log(msg: str) -> None:
    print(f"[build_public_umap {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Public dataset metadata
# ---------------------------------------------------------------------------

def _load_public_metadata() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Return ``(public_meta_by_name, split_by_name)`` for every screen in
    the public AssayBench dataset (train / val / test / LaTest novel)."""

    from assaybench import AssayBenchDataset  # local import for --help speed

    ds = AssayBenchDataset(
        dataset_name="biogrid",
        split_type="year",
        fold=0,
        novel_dataset_name="LaTest",
    )
    train, val, test, novel = ds.get_train_test_split()

    public: Dict[str, Dict[str, Any]] = {}
    split_by_name: Dict[str, str] = {}
    for split, examples in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
        ("LaTest", novel),
    ]:
        for ex in examples:
            name = str(ex["dataset_name"])
            split_by_name.setdefault(name, split)
            public.setdefault(
                name,
                {
                    "dataset_name": name,
                    "phenotype": (ex.get("phenotype") or "Not specified"),
                    "cleaned_phenotype": (ex.get("cleaned_phenotype") or "Not specified"),
                    "screen_rationale": ex.get("screen_rationale") or "",
                    "screen_type": ex.get("screen_type") or "",
                    "screen_category": ex.get("screen_category") or "",
                    "library_methodology": ex.get("library_methodology") or "",
                    "cell_type": ex.get("cell_type") or "",
                    "cell_line": ex.get("cell_line") or "",
                    "num_genes": int(ex.get("num_genes") or 0),
                    "author": ex.get("author") or "",
                    "source_id": str(ex.get("source_id") or ""),
                },
            )
    return public, split_by_name


# ---------------------------------------------------------------------------
# Filtering + UMAP
# ---------------------------------------------------------------------------

def _filter_transfer_matrix(
    transfer_matrix: np.ndarray,
    metadata_records: List[Dict[str, Any]],
    public_meta: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    keep_idx: List[int] = []
    dropped: List[str] = []
    for i, meta in enumerate(metadata_records):
        name = str(meta.get("dataset_name", ""))
        if name in public_meta:
            keep_idx.append(i)
        else:
            dropped.append(name)
    log(f"Kept {len(keep_idx)} / {len(metadata_records)} screens after public-only filtering")
    if dropped:
        log(f"  Dropped {len(dropped)} non-public screens (first 5): {dropped[:5]}")
    keep_idx_array = np.array(keep_idx, dtype=int)
    sub_matrix = transfer_matrix[np.ix_(keep_idx_array, keep_idx_array)].copy()
    sub_meta = [metadata_records[i] for i in keep_idx]
    return sub_matrix, sub_meta


def _ground_truth_distance(transfer_matrix: np.ndarray) -> np.ndarray:
    symmetric = (transfer_matrix + transfer_matrix.T) / 2.0
    distance = 1.0 - symmetric
    np.fill_diagonal(distance, 0.0)
    return np.clip(distance, 0.0, 1.0)


def _fit_all_umap(
    distance: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> np.ndarray:
    from umap import UMAP  # type: ignore  # noqa: PLC0415

    model = UMAP(
        n_neighbors=min(n_neighbors, max(2, distance.shape[0] - 1)),
        min_dist=min_dist,
        metric="precomputed",
        random_state=random_state,
        n_components=2,
    )
    return model.fit_transform(distance)


def _fit_train_project(
    distance: np.ndarray,
    train_mask: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    projection_k: int = 10,
) -> np.ndarray:
    from umap import UMAP  # type: ignore  # noqa: PLC0415

    train_indices = np.where(train_mask)[0]
    if len(train_indices) < 4:
        return _fit_all_umap(distance, n_neighbors, min_dist, random_state)

    train_dist = distance[np.ix_(train_indices, train_indices)]
    model = UMAP(
        n_neighbors=min(n_neighbors, max(2, len(train_indices) - 1)),
        min_dist=min_dist,
        metric="precomputed",
        random_state=random_state,
        n_components=2,
    )
    train_coords = model.fit_transform(train_dist)
    n_total = distance.shape[0]
    coords = np.zeros((n_total, 2), dtype=float)
    coords[train_indices] = train_coords

    other_indices = np.where(~train_mask)[0]
    if len(other_indices) == 0:
        return coords
    other_to_train = distance[np.ix_(other_indices, train_indices)]
    k = min(projection_k, len(train_indices))
    for row, ot in zip(other_indices, other_to_train):
        nn = np.argpartition(ot, k - 1)[:k]
        nn_dist = ot[nn]
        weights = 1.0 / (nn_dist + 1e-10)
        weights /= weights.sum()
        coords[row] = (weights[:, None] * train_coords[nn]).sum(axis=0)
    return coords


# ---------------------------------------------------------------------------
# DataFrame + HTML
# ---------------------------------------------------------------------------

def _extract_screen_ids(dataset_name: str) -> List[int]:
    if dataset_name.startswith("U_TR_"):
        parts = dataset_name[5:].split("_")
        return [int(p) for p in parts[:-1] if p.isdigit()]
    if dataset_name.startswith("TR_"):
        parts = dataset_name[3:].split("_")
        return [int(p) for p in parts if p.isdigit()]
    if dataset_name.startswith("U_"):
        parts = dataset_name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return [int(parts[1])]
        return []
    try:
        return [int(dataset_name)]
    except ValueError:
        return []


_SPLIT_SHAPE_MAP = {
    "Train": "circle",
    "Validation": "square",
    "Test": "diamond",
    "LaTest": "star",
    "Unknown": "x",
}


_PUB_YEAR_RE = re.compile(r"\((\d{4})\)")


def _extract_publication_year(author: Any) -> Optional[int]:
    if not author:
        return None
    match = _PUB_YEAR_RE.search(str(author))
    return int(match.group(1)) if match else None


def _build_dataframe(
    sub_meta: List[Dict[str, Any]],
    public_meta: Dict[str, Dict[str, Any]],
    split_by_name: Dict[str, str],
    umap_coords: Dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, meta in enumerate(sub_meta):
        name = str(meta["dataset_name"])
        public = public_meta[name]
        reverse = bool(meta.get("reverse", False))
        display_name = f"{name} ({'reverse' if reverse else 'forward'})"
        split = split_by_name.get(name, "Unknown")

        year = _extract_publication_year(public.get("author"))
        if name == "1686":  # known correction (mirrors plot4)
            year = 2021

        record = {
            "index": i,
            "dataset_name": name,
            "display_name": display_name,
            "reverse": "Reverse" if reverse else "Forward",
            "reverse_bool": reverse,
            "split": split,
            "shape": _SPLIT_SHAPE_MAP.get(split, "circle"),
            "phenotype": public["phenotype"],
            "cleaned_phenotype": public["cleaned_phenotype"],
            "screen_rationale": public["screen_rationale"],
            "screen_type": public["screen_type"],
            "screen_category": public["screen_category"],
            "library_methodology": public["library_methodology"],
            "cell_type": public["cell_type"],
            "cell_line": public["cell_line"],
            "num_genes": int(public.get("num_genes") or 0),
            "publication_year": year,
            "author": public["author"],
            "source_id": public["source_id"],
        }
        for layout, coords in umap_coords.items():
            record[f"umap_x_{layout}"] = float(coords[i, 0])
            record[f"umap_y_{layout}"] = float(coords[i, 1])
        rows.append(record)
    return pd.DataFrame(rows)


def _generate_html(
    df: pd.DataFrame,
    layout_names: List[str],
    layout_labels: List[str],
    umap_params: Dict[str, Any],
) -> str:
    categorical_cols = [
        "split",
        "reverse",
        "cleaned_phenotype",
        "screen_type",
        "screen_category",
        "library_methodology",
        "cell_type",
    ]
    continuous_cols = [c for c in ("num_genes", "publication_year") if c in df.columns]

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown").astype(str).replace({"": "Unknown"})

    data_records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = {
            "idx": int(row["index"]),
            "name": row["display_name"],
            "dataset_name": row["dataset_name"],
            "phenotype": row["phenotype"],
            "screen_rationale": row.get("screen_rationale", ""),
            "split": row["split"],
            "shape": row["shape"],
            "num_genes": int(row["num_genes"]),
            "cell_type": row.get("cell_type", ""),
            "cell_line": row.get("cell_line", ""),
            "author": row.get("author", ""),
        }
        for ln in layout_names:
            rec[f"x_{ln}"] = float(row[f"umap_x_{ln}"])
            rec[f"y_{ln}"] = float(row[f"umap_y_{ln}"])
        for col in categorical_cols:
            rec[col] = str(row[col])
        for col in continuous_cols:
            rec[col] = float(row[col]) if pd.notna(row[col]) else None
        rec["reverse"] = row["reverse"]
        data_records.append(rec)

    pretty_overrides = {
        "split": "Year split",
        "reverse": "Forward / Reverse",
        "cleaned_phenotype": "Phenotype class",
        "screen_type": "Screen type",
        "screen_category": "Screen category",
        "library_methodology": "Library methodology",
        "cell_type": "Cell type",
        "num_genes": "Library size",
        "publication_year": "Publication year",
    }

    def _pretty(name: str) -> str:
        return pretty_overrides.get(name, name.replace("_", " ").title())

    color_options = [{"value": c, "label": _pretty(c), "type": "categorical"} for c in categorical_cols]
    color_options.extend({"value": c, "label": _pretty(c), "type": "continuous"} for c in continuous_cols)
    size_options = [{"value": "uniform", "label": "Uniform"}]
    size_options.extend({"value": c, "label": _pretty(c)} for c in continuous_cols)

    split_colors = {
        "Train": "#3498db",
        "Validation": "#f39c12",
        "Test": "#e74c3c",
        "LaTest": "#9b59b6",
        "Unknown": "#95a5a6",
    }
    fixed_palettes = {"split": split_colors, "reverse": {"Forward": "#2ecc71", "Reverse": "#9b59b6"}}

    n_screens = len(df)
    n_forward = int((df["reverse"] == "Forward").sum())
    n_reverse = int((df["reverse"] == "Reverse").sum())
    info_text = (
        f"Public AssayBench screens: {n_screens} ({n_forward} forward, {n_reverse} reverse) "
        f"| UMAP n_neighbors={umap_params['n_neighbors']}, min_dist={umap_params['min_dist']}"
    )

    data_json = json.dumps(data_records)
    layout_json = json.dumps(layout_names)
    layout_labels_json = json.dumps(layout_labels)
    color_options_json = json.dumps(color_options)
    size_options_json = json.dumps(size_options)
    fixed_palettes_json = json.dumps(fixed_palettes)
    shape_map_json = json.dumps(_SPLIT_SHAPE_MAP)

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\"/>
<title>AssayBench — Screen UMAP Explorer (public)</title>
<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#f4f5f7; color:#222; }}
#topbar {{ display:flex; align-items:center; gap:18px; padding:10px 20px; background:#fff;
           border-bottom:1px solid #ddd; flex-wrap:wrap; }}
#topbar label {{ font-size:12px; color:#555; font-weight:600; }}
#topbar select, #topbar input {{ font-size:13px; padding:4px 8px; border:1px solid #ccc;
                                   border-radius:4px; background:#fff; }}
#topbar select {{ max-width:260px; }}
#search {{ width:220px; }}
#main {{ display:flex; height:calc(100vh - 60px); }}
#plot-container {{ flex:1; min-width:0; }}
#detail-panel {{ width:340px; background:#fff; border-left:1px solid #ddd; overflow-y:auto;
                  padding:16px; display:none; font-size:13px; }}
#detail-panel h3 {{ margin-bottom:10px; font-size:15px; }}
#detail-panel .field {{ margin-bottom:6px; }}
#detail-panel .field-label {{ font-weight:600; color:#555; }}
#detail-panel .close-btn {{ float:right; cursor:pointer; font-size:18px; color:#999; }}
#detail-panel .close-btn:hover {{ color:#333; }}
#info {{ font-size:11px; color:#888; padding:4px 20px; background:#fff; border-top:1px solid #eee; }}
</style>
</head>
<body>
<div id=\"topbar\">
  <div><label>Layout</label><br/><select id=\"sel-layout\"></select></div>
  <div><label>Color by</label><br/><select id=\"sel-color\"></select></div>
  <div><label>Size by</label><br/><select id=\"sel-size\"></select></div>
  <div><label>Search</label><br/><input id=\"search\" type=\"text\" placeholder=\"Screen / phenotype / cell type...\"/></div>
</div>
<div id=\"main\">
  <div id=\"plot-container\"><div id=\"plot\" style=\"width:100%;height:100%;\"></div></div>
  <div id=\"detail-panel\">
    <span class=\"close-btn\" id=\"close-detail\">&times;</span>
    <h3 id=\"detail-title\"></h3>
    <div id=\"detail-body\"></div>
  </div>
</div>
<div id=\"info\">{info_text}</div>
<script>
(function() {{
  const DATA = {data_json};
  const LAYOUTS = {layout_json};
  const LAYOUT_LABELS = {layout_labels_json};
  const COLOR_OPTS = {color_options_json};
  const SIZE_OPTS = {size_options_json};
  const FIXED_PALETTES = {fixed_palettes_json};
  const SHAPE_MAP = {shape_map_json};

  const QUAL_PALETTE = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
    '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
    '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5',
    '#c49c94','#f7b6d2','#c7c7c7','#dbdb8d','#9edae5'
  ];

  const selLayout = document.getElementById('sel-layout');
  const selColor = document.getElementById('sel-color');
  const selSize = document.getElementById('sel-size');

  LAYOUTS.forEach((l, i) => {{
    const o = document.createElement('option'); o.value = l; o.text = LAYOUT_LABELS[i];
    selLayout.appendChild(o);
  }});
  COLOR_OPTS.forEach(c => {{
    const o = document.createElement('option'); o.value = c.value;
    o.text = c.label + (c.type === 'continuous' ? ' (cont.)' : '');
    o.dataset.ctype = c.type;
    selColor.appendChild(o);
  }});
  SIZE_OPTS.forEach(s => {{
    const o = document.createElement('option'); o.value = s.value; o.text = s.label;
    selSize.appendChild(o);
  }});

  let currentLayout = LAYOUTS[0];
  let currentColor = COLOR_OPTS[0].value;
  let currentColorType = COLOR_OPTS[0].type;
  let currentSize = 'uniform';
  let searchTerm = '';

  function getSizes() {{
    if (currentSize === 'uniform') return DATA.map(() => 9);
    const vals = DATA.map(d => d[currentSize]);
    const valid = vals.filter(v => v !== null && v !== undefined);
    if (valid.length === 0) return DATA.map(() => 9);
    const mn = Math.min(...valid), mx = Math.max(...valid);
    const range = mx - mn || 1;
    return vals.map(v => (v !== null && v !== undefined) ? 6 + 18 * (v - mn) / range : 5);
  }}
  function getMask() {{
    if (!searchTerm) return DATA.map(() => true);
    const t = searchTerm.toLowerCase();
    return DATA.map(d =>
      (d.name || '').toLowerCase().includes(t) ||
      (d.phenotype || '').toLowerCase().includes(t) ||
      (d.cell_type || '').toLowerCase().includes(t) ||
      (d.screen_rationale || '').toLowerCase().includes(t)
    );
  }}
  function buildHover(d) {{
    let s = '<b>' + d.name + '</b><br>';
    s += '<b>Split:</b> ' + d.split + '<br>';
    s += '<b>Phenotype:</b> ' + d.phenotype + '<br>';
    if (d.cleaned_phenotype) s += '<b>Class:</b> ' + d.cleaned_phenotype + '<br>';
    if (d.cell_line) s += '<b>Cell line:</b> ' + d.cell_line + '<br>';
    s += '<b>Genes in library:</b> ' + d.num_genes.toLocaleString();
    return s;
  }}
  function buildTraces() {{
    const coords = {{ x: DATA.map(d => d['x_' + currentLayout]), y: DATA.map(d => d['y_' + currentLayout]) }};
    const sizes = getSizes();
    const mask = getMask();
    if (currentColorType === 'continuous') {{
      const vals = DATA.map(d => d[currentColor]);
      return [{{
        x: coords.x, y: coords.y, mode: 'markers',
        marker: {{
          size: sizes,
          color: vals.map(v => (v !== null && v !== undefined) ? v : NaN),
          colorscale: 'Viridis',
          colorbar: {{ title: selColor.options[selColor.selectedIndex].text, thickness: 15 }},
          symbol: DATA.map(d => d.shape),
          opacity: mask.map(m => m ? 0.85 : 0.08),
          line: {{ width: 0.5, color: 'white' }},
        }},
        text: DATA.map(buildHover),
        hovertemplate: '%{{text}}<extra></extra>',
        type: 'scatter',
      }}];
    }}
    const catVals = DATA.map(d => d[currentColor] || 'Unknown');
    const uniq = [...new Set(catVals)].sort();
    const palette = FIXED_PALETTES[currentColor] || {{}};
    const traces = [];
    uniq.forEach((cat, ci) => {{
      const idx = [];
      DATA.forEach((_, i) => {{ if (catVals[i] === cat) idx.push(i); }});
      traces.push({{
        x: idx.map(i => coords.x[i]),
        y: idx.map(i => coords.y[i]),
        mode: 'markers',
        name: cat,
        marker: {{
          size: idx.map(i => sizes[i]),
          color: palette[cat] || QUAL_PALETTE[ci % QUAL_PALETTE.length],
          symbol: idx.map(i => DATA[i].shape),
          opacity: idx.map(i => mask[i] ? 0.85 : 0.08),
          line: {{ width: 0.5, color: 'white' }},
        }},
        text: idx.map(i => buildHover(DATA[i])),
        hovertemplate: '%{{text}}<extra></extra>',
        customdata: idx,
        type: 'scatter',
      }});
    }});
    return traces;
  }}
  function render() {{
    const layout = {{
      xaxis: {{ title: 'UMAP 1', showgrid: true, gridcolor: 'rgba(128,128,128,0.15)', zeroline: false }},
      yaxis: {{ title: 'UMAP 2', showgrid: true, gridcolor: 'rgba(128,128,128,0.15)', zeroline: false }},
      legend: {{ title: {{ text: selColor.options[selColor.selectedIndex].text }}, font: {{ size: 11 }} }},
      plot_bgcolor: '#fafafa', paper_bgcolor: '#fff',
      margin: {{ l: 50, r: 20, t: 20, b: 50 }},
      hoverlabel: {{ bgcolor: 'white', font: {{ size: 12 }}, align: 'left' }},
    }};
    Plotly.react('plot', buildTraces(), layout, {{
      displayModeBar: true, displaylogo: false, responsive: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    }});
  }}

  selLayout.addEventListener('change', function() {{ currentLayout = this.value; render(); }});
  selColor.addEventListener('change', function() {{
    currentColor = this.value;
    currentColorType = this.options[this.selectedIndex].dataset.ctype;
    render();
  }});
  selSize.addEventListener('change', function() {{ currentSize = this.value; render(); }});

  let searchTimeout;
  document.getElementById('search').addEventListener('input', function() {{
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {{ searchTerm = this.value; render(); }}, 250);
  }});

  const detailPanel = document.getElementById('detail-panel');
  const detailTitle = document.getElementById('detail-title');
  const detailBody = document.getElementById('detail-body');

  // Render once so Plotly attaches itself to the plot div, then wire up the
  // click handler (Plotly's `.on()` is only defined after the first draw).
  render();

  const plotDiv = document.getElementById('plot');
  plotDiv.on('plotly_click', function(evtData) {{
    if (!evtData || !evtData.points || !evtData.points.length) return;
    const pt = evtData.points[0];
    const idx = pt.customdata !== undefined ? pt.customdata : pt.pointIndex;
    if (idx === undefined) return;
    const d = DATA[idx];
    if (!d) return;
    detailTitle.textContent = d.name;
    const fields = [
      ['Split', d.split], ['Reverse', d.reverse],
      ['Phenotype (raw)', d.phenotype],
      ['Cell line', d.cell_line], ['Cell type', d.cell_type],
      ['Library size', d.num_genes ? d.num_genes.toLocaleString() : '-'],
      ['Author', d.author],
    ];
    let html = '';
    fields.forEach(([lbl, val]) => {{
      if (!val) return;
      html += '<div class=\"field\"><span class=\"field-label\">' + lbl
              + ':</span> <span class=\"field-value\">' + val + '</span></div>';
    }});
    if (d.screen_rationale) {{
      html += '<hr style=\"margin:10px 0\"/>' +
              '<div class=\"field\"><span class=\"field-label\">Screen rationale:</span><br/>'
              + '<span class=\"field-value\" style=\"font-size:12px;line-height:1.5\">' + d.screen_rationale + '</span></div>';
    }}
    detailBody.innerHTML = html;
    detailPanel.style.display = 'block';
  }});
  document.getElementById('close-detail').addEventListener('click', function() {{
    detailPanel.style.display = 'none';
  }});
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transfer-matrix-dir",
        type=Path,
        default=DEFAULT_TRANSFER_DIR,
        help="Directory containing transfer_matrix.npy + screen_metadata.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where to write screen_umap_explorer.html and umap_coordinates.csv.",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    matrix_path = args.transfer_matrix_dir / "transfer_matrix.npy"
    metadata_path = args.transfer_matrix_dir / "screen_metadata.json"
    if not matrix_path.exists() or not metadata_path.exists():
        sys.stderr.write(
            f"Could not find {matrix_path.name} / {metadata_path.name} in {args.transfer_matrix_dir}\n"
        )
        sys.exit(1)

    log(f"Loading transfer matrix from {args.transfer_matrix_dir}")
    transfer_matrix = np.load(matrix_path)
    with open(metadata_path) as handle:
        meta = json.load(handle)
    metadata_records = meta["screen_metadata"]
    log(f"  Transfer matrix shape: {transfer_matrix.shape}")
    log(f"  Metadata rows: {len(metadata_records)}")

    log("Loading public AssayBench metadata ...")
    public_meta, split_by_name = _load_public_metadata()
    log(f"  Public dataset_names: {len(public_meta)}")

    sub_matrix, sub_meta = _filter_transfer_matrix(transfer_matrix, metadata_records, public_meta)

    # Defensive: every kept screen must be in the public set.
    for record in sub_meta:
        assert str(record["dataset_name"]) in public_meta, (
            f"Leakage check failed for {record['dataset_name']}"
        )

    log("Computing ground-truth distance matrix ...")
    distance = _ground_truth_distance(sub_matrix)

    log("Fitting UMAP on all public screens ...")
    coords_all = _fit_all_umap(
        distance,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )

    log("Fitting UMAP on train, projecting val/test/LaTest via 10-NN ...")
    train_mask = np.array(
        [split_by_name.get(str(r["dataset_name"])) == "Train" for r in sub_meta]
    )
    coords_train = _fit_train_project(
        distance,
        train_mask=train_mask,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )

    umap_coords = {
        "ground_truth": coords_all,
        "ground_truth_train_fitted": coords_train,
    }
    df = _build_dataframe(sub_meta, public_meta, split_by_name, umap_coords)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    coords_path = out_dir / "umap_coordinates.csv"
    df.to_csv(coords_path, index=False)
    log(f"Saved coordinates CSV to {coords_path}")

    html_path = out_dir / "screen_umap_explorer.html"
    html = _generate_html(
        df=df,
        layout_names=list(umap_coords.keys()),
        layout_labels=[
            "Ground Truth (All Public Screens)",
            "Ground Truth (Train-Fitted, Project Val/Test/LaTest)",
        ],
        umap_params={"n_neighbors": args.n_neighbors, "min_dist": args.min_dist},
    )
    html_path.write_text(html)
    log(f"Saved interactive UMAP HTML to {html_path} ({html_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
