from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import blended_transform_factory

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "journal_figures"

# ── Model metadata ──────────────────────────────────────────────────────────
# total_params_b=None marks frontier (undisclosed size).
MODEL_META = {
    "qwen3.5-0.8b":          {"total_params_b": 0.8,  "is_moe": False},
    "qwen3.5-2b":            {"total_params_b": 2,    "is_moe": False},
    "qwen3-4b-2507":         {"total_params_b": 4,    "is_moe": False},
    "qwen3.5-4b":            {"total_params_b": 4,    "is_moe": False},
    "qwen3.5-9b":            {"total_params_b": 9,    "is_moe": False},
    "gpt-oss-20b":           {"total_params_b": 20,   "is_moe": False},
    "qwen3.5-27b":           {"total_params_b": 27,   "is_moe": False},
    "qwen3-30b-a3b-2507":    {"total_params_b": 30,   "is_moe": True},
    "olmo-3.1-32b-think":    {"total_params_b": 32,   "is_moe": False},
    "qwen3.5-35b-a3b":       {"total_params_b": 35,   "is_moe": True},
    "gpt-oss-120b":          {"total_params_b": 120,  "is_moe": False},
    "qwen3-coder-next":      {"total_params_b": 80,   "is_moe": True},
    "qwen3.5-122b-a10b":     {"total_params_b": 122,  "is_moe": True},
    "MiniMax-M2.5":          {"total_params_b": 229,  "is_moe": True},
    "qwen3-235b-a22b-2507":  {"total_params_b": 235,  "is_moe": True},
    "qwen3.5-397b-a17b":     {"total_params_b": 397,  "is_moe": True},
    "deepseek-v3.2":         {"total_params_b": 685,  "is_moe": True},
    "deepseek-v3.2-nothink": {"total_params_b": 685,  "is_moe": True},
    "GLM-5":                 {"total_params_b": 744,  "is_moe": True},
    "Kimi-K2.5":             {"total_params_b": 1000, "is_moe": True},
    "claude-haiku-4.5":      {"total_params_b": None, "is_moe": None},
    "claude-sonnet-4.5":     {"total_params_b": None, "is_moe": None},
    "claude-opus-4.5":       {"total_params_b": None, "is_moe": None},
    "gpt-5-mini":            {"total_params_b": None, "is_moe": None},
    "gpt-5.2":               {"total_params_b": None, "is_moe": None},
    "gpt-5.4":               {"total_params_b": None, "is_moe": None},
    "gemini-3-flash":        {"total_params_b": None, "is_moe": None},
    "gemini-3-pro":          {"total_params_b": None, "is_moe": None},
    "gemini-3.1-pro":        {"total_params_b": None, "is_moe": None},
    "biomni-a1-claude-4":    {"total_params_b": None, "is_moe": None},
}

SIZE_ORDER = [
    "Small (<10B)",
    "Medium (10-50B)",
    "Large (50-250B)",
    "Very Large (>250B)",
    "Frontier",
]


def _size_category(model_name: str) -> str:
    meta = MODEL_META.get(model_name)
    if meta is None or meta["total_params_b"] is None:
        return "Frontier"
    p = meta["total_params_b"]
    if p < 10:
        return "Small (<10B)"
    if p < 50:
        return "Medium (10-50B)"
    if p < 250:
        return "Large (50-250B)"
    return "Very Large (>250B)"


def _sort_key(model_name: str):
    cat = _size_category(model_name)
    return (SIZE_ORDER.index(cat) if cat in SIZE_ORDER else 99, model_name)


# ── Edit this list to select which models appear in the plot ────────────────
MODELS_TO_PLOT: list[str] | None = [
    "qwen3.5-0.8b",
    "qwen3.5-2b",
    #"qwen3-4b-2507",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "gpt-oss-20b",
    "qwen3.5-27b",
    #"qwen3-30b-a3b-2507",
    #"olmo-3.1-32b-think",
    "qwen3.5-35b-a3b",
    "gpt-oss-120b",
    #"qwen3-coder-next",
    "qwen3.5-122b-a10b",
    "MiniMax-M2.5",
    #"qwen3-235b-a22b-2507",
    "qwen3.5-397b-a17b",
    "deepseek-v3.2",
    #"deepseek-v3.2-nothink",
    "GLM-5",
    "Kimi-K2.5",
    #"claude-haiku-4.5",
    #"claude-sonnet-4.5",
    #"claude-opus-4.5",
    #"gpt-5-mini",
    "gpt-5.2",
    "gpt-5.4",
    "gemini-3-flash",
    "gemini-3-pro",
    "gemini-3.1-pro",
    "biomni-a1-claude-4",
]


# ── Edit this dict to override how model names appear on the plot ─────────
# Models not listed here keep their original name from the CSV.
DISPLAY_NAMES: dict[str, str] = {
    "qwen3.5-0.8b":       "Qwen3.5-0.8B",
    "qwen3.5-2b":         "Qwen3.5-2B",
    "qwen3.5-4b":         "Qwen3.5-4B",
    "qwen3.5-9b":         "Qwen3.5-9B",
    "qwen3.5-27b":        "Qwen3.5-27B",
    "qwen3.5-35b-a3b":    "Qwen3.5-35B",
    "qwen3.5-122b-a10b":  "Qwen3.5-122B",
    "qwen3.5-397b-a17b":  "Qwen3.5-397B",
    "biomni-a1-claude-4": "Biomni-A1 (Claude 4)",
    "gpt-oss-20b":        "GPT-OSS-20B",
    "gpt-oss-120b":       "GPT-OSS-120B",
    "deepseek-v3.2":      "DeepSeek v3.2",
    "gemini-3-flash":       "Gemini 3 Flash",
    "gemini-3-pro":         "Gemini 3 Pro",
    "gemini-3.1-pro":       "Gemini 3.1 Pro",
    "gpt-5.2":              "GPT-5.2",
    "gpt-5.4":              "GPT-5.4",
}


def main() -> None:
    # ── LaTeX-style fonts (works without a TeX installation) ───────────────
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    # ── Load data ───────────────────────────────────────────────────────────
    bias = pd.read_csv(DATA_DIR / "bias_matrix.csv", index_col="model_name")

    if MODELS_TO_PLOT is not None:
        available = [m for m in MODELS_TO_PLOT if m in bias.index]
        missing = set(MODELS_TO_PLOT) - set(available)
        if missing:
            print(f"Warning: skipping {len(missing)} model(s) not in data: {sorted(missing)}")
        bias = bias.loc[available]

    bias = bias.loc[sorted(bias.index, key=_sort_key)]

    # ── Size-category spans and boundaries (computed on original names) ─────
    row_cats = [_size_category(m) for m in bias.index]

    # ── Rename models for display ───────────────────────────────────────────
    bias.index = [DISPLAY_NAMES.get(m, m) for m in bias.index]
    boundaries: list[int] = []
    cat_spans: list[tuple[str, int, int]] = []
    prev, start = row_cats[0], 0
    for i, cat in enumerate(row_cats):
        if cat != prev:
            boundaries.append(i)
            cat_spans.append((prev, start, i))
            start, prev = i, cat
    cat_spans.append((prev, start, len(row_cats)))

    # ── Clustermap ──────────────────────────────────────────────────────────
    vmax = np.abs(bias.values).max()

    g = sns.clustermap(
        bias,
        col_cluster=False,
        row_cluster=False,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.3,
        figsize=(bias.shape[1] * 0.35, bias.shape[0] * 0.25),
        xticklabels=True,
        yticklabels=True,
    )
    g.ax_col_dendrogram.set_visible(False)
    g.ax_row_dendrogram.set_visible(False)

    # Size-category separators
    ytick_trans = blended_transform_factory(
        g.ax_heatmap.transAxes, g.ax_heatmap.transData
    )
    for b in boundaries:
        g.ax_heatmap.plot(
            [-0.8, 0], [b, b],
            transform=ytick_trans, color="black", lw=0.5, clip_on=False,
        )
    for cat, ys, ye in cat_spans:
        g.ax_heatmap.text(
            -0.55, (ys + ye) / 2,
            cat.replace("(", "\n("),
            ha="center", va="center", fontsize=7, fontweight="bold",
            transform=g.ax_heatmap.get_yaxis_transform(),
        )

    # Horizontal colorbar on top
    g.cax.set_visible(False)
    hm_pos = g.ax_heatmap.get_position()
    cbar_ax = g.fig.add_axes([
        hm_pos.x0 + hm_pos.width * 0.1,
        hm_pos.y1 + 0.03,
        hm_pos.width * 0.8,
        0.01,
    ])
    g.fig.colorbar(g.ax_heatmap.collections[0], cax=cbar_ax, orientation="horizontal")
    cbar_ax.tick_params(labelsize=6)
    cbar_ax.set_title(r"Bias (pred $-$ GT fraction)", fontsize=8, pad=4)

    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=6, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6, rotation=0, fontweight="bold")

    # ── Save ────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "plot6_bias.pdf"
    g.fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
    plt.close(g.fig)


if __name__ == "__main__":
    main()
