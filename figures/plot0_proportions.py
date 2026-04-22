from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from assaybench import AssayBenchDataset

from journal_figures_common import OUTPUT_DIR

PHENOTYPE_ORDER = [
    "Fitness / Proliferation / Viability",
    "Drug / Chemical / Environmental Response",
    "Host-Pathogen / Infection Response",
    "Molecular Output / Reporter / Pathway Activity",
    "Trafficking / Localization / Structural Phenotypes",
]

PHENOTYPE_COLORS = {
    "Fitness / Proliferation / Viability": "#1f77b4",
    "Drug / Chemical / Environmental Response": "#aec7e8",
    "Host-Pathogen / Infection Response": "#ff7f0e",
    "Molecular Output / Reporter / Pathway Activity": "#ffbb78",
    "Trafficking / Localization / Structural Phenotypes": "#2ca02c",
}

EXTRA_PALETTE = ["#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]

SPLIT_ORDER = ["train", "val", "test"]


def load_dataset_dataframe() -> pd.DataFrame:
    ds = AssayBenchDataset(
        dataset_name="biogrid",
        split_type="year",
        fold=0,
        novel_dataset_name="2026Q1",
    )
    train, val, test, novel = ds.get_train_test_split()

    rows: List[Dict] = []
    split_map = {"train": train, "val": val, "test": test, "internal": novel}
    for split_label, examples in split_map.items():
        for ex in examples:
            rows.append({
                "dataset_name": ex["dataset_name"],
                "split": split_label,
                "phenotype": ex.get("cleaned_phenotype", "Not specified"),
                "num_genes": len(ex["relevance_genes"]),
                "num_screens": len(ex["screen_ids"]),
                "screen_ids": ex["screen_ids"],
            })

    return pd.DataFrame(rows)


def _summarize_row(label: str, values_by_split: pd.Series, total_value=None) -> pd.Series:
    values_by_split = values_by_split.reindex(SPLIT_ORDER).fillna(0)
    if total_value is None:
        total_value = values_by_split.sum()
    return pd.Series({
        "Metric": label,
        "Total": total_value,
        "Train": values_by_split["train"],
        "Val": values_by_split["val"],
        "Test": values_by_split["test"],
    })


def _format_count_and_pct(count, denominator) -> str:
    pct = 0.0 if denominator == 0 else 100 * count / denominator
    return f"{int(count):,} ({pct:.1f}%)"


def build_latex_table(df_biogrid: pd.DataFrame) -> str:
    benchmark_entries_by_split = df_biogrid.groupby("split").size().reindex(SPLIT_ORDER).fillna(0)
    total_screens_by_split = df_biogrid.groupby("split")["num_screens"].sum().reindex(SPLIT_ORDER).fillna(0)
    unique_screen_ids_df = df_biogrid.explode("screen_ids")
    unique_screen_ids_by_split = (
        unique_screen_ids_df.groupby("split")["screen_ids"].nunique().reindex(SPLIT_ORDER).fillna(0)
    )

    rows = [
        _summarize_row("Benchmark entries", benchmark_entries_by_split, total_value=len(df_biogrid)),
        _summarize_row(
            "Unique screen IDs",
            unique_screen_ids_by_split,
            total_value=unique_screen_ids_df["screen_ids"].nunique(),
        ),
    ]

    phenotype_counts = df_biogrid.groupby(["split", "phenotype"])["num_screens"].sum()
    for phenotype in PHENOTYPE_ORDER:
        if phenotype in phenotype_counts.index.get_level_values("phenotype"):
            rows.append(
                _summarize_row(
                    f"Phenotype: {phenotype}",
                    phenotype_counts.xs(phenotype, level="phenotype"),
                    total_value=df_biogrid.loc[df_biogrid["phenotype"] == phenotype, "num_screens"].sum(),
                )
            )

    rows.append(
        _summarize_row(
            "Avg. relevance genes / entry",
            df_biogrid.groupby("split")["num_genes"].mean(),
            total_value=df_biogrid["num_genes"].mean(),
        )
    )
    rows.append(
        _summarize_row(
            "Merged replicate entries",
            df_biogrid.assign(is_merged=df_biogrid["num_screens"] > 1).groupby("split")["is_merged"].sum(),
            total_value=(df_biogrid["num_screens"] > 1).sum(),
        )
    )

    latex_table_df = pd.DataFrame(rows).set_index("Metric")

    entry_denominators = pd.Series({
        "Total": len(df_biogrid),
        "Train": benchmark_entries_by_split["train"],
        "Val": benchmark_entries_by_split["val"],
        "Test": benchmark_entries_by_split["test"],
    })
    screen_denominators = pd.Series({
        "Total": total_screens_by_split.sum(),
        "Train": total_screens_by_split["train"],
        "Val": total_screens_by_split["val"],
        "Test": total_screens_by_split["test"],
    })

    formatted_table = latex_table_df.copy().astype(object)
    formatted_table.loc["Avg. relevance genes / entry"] = formatted_table.loc[
        "Avg. relevance genes / entry"
    ].map(lambda v: f"{v:,.1f}")
    formatted_table.loc["Unique screen IDs"] = formatted_table.loc["Unique screen IDs"].map(
        lambda v: f"{int(v):,}"
    )

    for row_name in formatted_table.index:
        if row_name in {"Avg. relevance genes / entry", "Unique screen IDs"}:
            continue
        denominators = screen_denominators if row_name.startswith("Phenotype:") else entry_denominators
        formatted_table.loc[row_name] = [
            _format_count_and_pct(latex_table_df.loc[row_name, col], denominators[col])
            for col in latex_table_df.columns
        ]

    latex_ready = formatted_table.apply(
        lambda col: col.map(lambda v: v.replace("%", r"\%") if isinstance(v, str) else v)
    )
    latex_str = latex_ready.to_latex(
        index_names=False,
        escape=False,
        caption=r"AssayBench dataset statistics by split.",
        label="tab:biogrid_dataset_stats",
        column_format="lrrrr",
    )
    return latex_str


def render_pie_charts(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    df_biogrid = df.loc[df["split"] != "internal"]

    phenotype_colors = dict(PHENOTYPE_COLORS)
    extra_phenotypes = [
        p for p in sorted(df.loc[df["split"] == "internal", "phenotype"].dropna().unique())
        if p not in phenotype_colors
    ]
    for color, phenotype in zip(EXTRA_PALETTE, extra_phenotypes):
        phenotype_colors[phenotype] = color
    full_phenotype_order = list(PHENOTYPE_ORDER) + extra_phenotypes

    plot_frames = {
        "Total": df_biogrid,
        "Train": df.loc[df["split"] == "train"],
        "Val": df.loc[df["split"] == "val"],
        "Test": df.loc[df["split"] == "test"],
        "Novel dataset": df.loc[df["split"] == "internal"],
    }

    def phenotype_entry_counts(dataframe):
        return dataframe.groupby("phenotype").size().reindex(full_phenotype_order).fillna(0)

    def autopct_if_large(pct):
        return f"{pct:.1f}%" if pct >= 2 else ""

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, (title, subset) in zip(axes, plot_frames.items()):
        counts = phenotype_entry_counts(subset)
        nonzero_counts = counts[counts > 0]
        ax.pie(
            nonzero_counts,
            labels=None,
            colors=[phenotype_colors[p] for p in nonzero_counts.index],
            autopct=autopct_if_large,
            startangle=90,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"fontsize": 10},
        )
        ax.set_title(title, fontsize=13, fontweight="semibold")

    legend_handles = [
        Patch(facecolor=phenotype_colors[p], label=p)
        for p in full_phenotype_order
        if p in phenotype_colors
    ]
    axes[-1].axis("off")
    axes[-1].legend(
        handles=legend_handles,
        loc="center",
        frameon=False,
        title="Phenotypes",
        title_fontsize=14,
        fontsize=11,
        handlelength=1.2,
        handleheight=1.4,
    )

    fig.suptitle("Phenotype composition by split", fontsize=16, fontweight="bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"plot0_phenotype_composition_by_split.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dataset statistics LaTeX table and phenotype composition pie charts.",
    )
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from AssayBench...")
    df = load_dataset_dataframe()
    df_biogrid = df.loc[df["split"] != "internal"]

    print(f"Loaded {len(df)} entries ({len(df_biogrid)} biogrid + {(df['split'] == 'internal').sum()} novel)")

    latex_str = build_latex_table(df_biogrid)
    latex_path = output_dir / "plot0_dataset_stats.tex"
    latex_path.write_text(latex_str)
    print(f"\nLaTeX table saved to {latex_path}")
    print(latex_str)

    render_pie_charts(df, output_dir, args.dpi)
    print(f"Pie charts saved to {output_dir / 'plot0_phenotype_composition_by_split.png'}")


if __name__ == "__main__":
    main()
