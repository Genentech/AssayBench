from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from journal_figures_common import BASE_DIR, OUTPUT_DIR, apply_style, plt, short_label
from scripts.shared_utils import load_additional_ground_truth, load_all_model_predictions

ONGOING_SCREEN_DATASET_PATHS = [
    "/cv/data/braid/wua33/screens-workspace/datasets/treg_suppression_v1",
    "/cv/data/braid/wua33/screens-workspace/datasets/zhu2025_perturbseq_v1",
    "/cv/data/braid/gnesys/datasets/screensQA/NGS7126/NGS7126_v1",
]

DEFAULT_MODEL = "gemini-3-pro"
DEFAULT_PREDICTIONS_BASE_DIR = BASE_DIR / "output" / "multi_model_ensemble"
DEFAULT_CT_PREDICTIONS_JSON = (
    Path(__file__).resolve().parent.parent / "celltypeQAnew" / "outputs" / "jake_tregs.json"
)
DEFAULT_CT_PREDICTIONS_JSONS = [
    DEFAULT_CT_PREDICTIONS_JSON,
    Path(__file__).resolve().parent.parent / "celltypeQAnew" / "outputs" / "mike.json",
]
FILENAME_STEM = "plot_internal_perf"
_TREG_LABELS = {
    "jake_treg_suppresion": "Treg full",
    "jake_treg_suppresion_simplified": "Treg simplified",
}
_NGS_PATTERN = re.compile(
    r"^NGS7126_Day(?P<day>\d+)_(?P<condition>.+)_(?P<direction>increase|decrease)$"
)
_GROUP_COLORS = {
    "Treg": "#4C78A8",
    "NGS7126": "#F58518",
}
_GROUP_ORDER = {
    "Treg": 0,
    "NGS7126": 1,
}
_METHOD_COLORS = {
    "gemini-3-pro": "#4C78A8",
    "CT": "#54A24B",
}


def _classify_internal_group(dataset_name: str) -> Optional[str]:
    if dataset_name in _TREG_LABELS:
        return "Treg"
    if dataset_name.startswith("NGS7126_"):
        return "NGS7126"
    return None


def _format_dataset_label(dataset_name: str) -> str:
    if dataset_name in _TREG_LABELS:
        return _TREG_LABELS[dataset_name]

    match = _NGS_PATTERN.match(dataset_name)
    if match:
        condition = match.group("condition").replace("_vs_DMSO", "")
        direction = "up" if match.group("direction") == "increase" else "down"
        return f"Day {match.group('day')} {condition} {direction}"

    return dataset_name


def _load_internal_ground_truth() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    gt_map = load_additional_ground_truth(
        split_name="ongoing_screens",
        dataset_paths=ONGOING_SCREEN_DATASET_PATHS,
        use_existing_prompt=True,
        display_library_genes=False,
    )
    screen_keys = sorted(gt_map.keys(), key=lambda key: int(key.split(":")[1]))
    for screen_order, key in enumerate(screen_keys):
        ground_truth = gt_map[key]
        dataset_name = str(ground_truth.get("dataset_name", "unknown"))
        group = _classify_internal_group(dataset_name)
        if group is None:
            continue
        rows.append(
            {
                "screen_order": screen_order,
                "screen_key": key,
                "dataset_name": dataset_name,
                "dataset_label": _format_dataset_label(dataset_name),
                "group": group,
                "ground_truth": ground_truth,
                "num_hits": sum(
                    1
                    for score in ground_truth["relevance_scores"]
                    if isinstance(score, (int, float)) and score > 0
                ),
                "num_genes": len(ground_truth["genes"]),
            }
        )
    return rows


def _evaluate_runs_against_ground_truth(
    *,
    method_name: str,
    prediction_runs_by_dataset: Dict[str, List[List[str]]],
    internal_ground_truth_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    from screensqa.benchmark.ranking_metrics import RankingMetrics

    evaluator = RankingMetrics(k_values=[5, 10, 20, 50, 100], use_thresholded_scoring=True)

    rows: List[Dict[str, Any]] = []
    for entry in internal_ground_truth_rows:
        dataset_name = entry["dataset_name"]
        ground_truth = entry["ground_truth"]
        prediction_runs = prediction_runs_by_dataset.get(dataset_name)
        if prediction_runs is None:
            rows.append(
                {
                    "screen_order": entry["screen_order"],
                    "dataset_name": dataset_name,
                    "dataset_label": entry["dataset_label"],
                    "group": entry["group"],
                    "method": method_name,
                    "adjusted_ndcg@100": None,
                    "n_runs": 0,
                    "num_hits": entry["num_hits"],
                    "num_genes": entry["num_genes"],
                    "status": "missing_predictions",
                }
            )
            continue

        scores: List[float] = []
        for run in prediction_runs:
            result = evaluator.evaluate(
                predicted_genes=list(run),
                ground_truth_genes=ground_truth["genes"],
                relevance_scores=ground_truth["relevance_scores"],
            )
            value = result.get("adjusted_ndcg@100")
            if value is not None:
                scores.append(float(value))

        rows.append(
            {
                "screen_order": entry["screen_order"],
                "dataset_name": dataset_name,
                "dataset_label": entry["dataset_label"],
                "group": entry["group"],
                "method": method_name,
                "adjusted_ndcg@100": sum(scores) / len(scores) if scores else None,
                "n_runs": len(scores),
                "num_hits": entry["num_hits"],
                "num_genes": entry["num_genes"],
                "status": "ok" if scores else "missing_scores",
            }
        )

    return pd.DataFrame(rows)


def _load_gemini_prediction_runs(
    *,
    model_name: str,
    predictions_base_dir: Path,
    internal_ground_truth_rows: List[Dict[str, Any]],
) -> Dict[str, List[List[str]]]:
    pred_map = load_all_model_predictions(
        predictions_base_dir,
        model_name,
        additional_splits=["ongoing_screens"],
    )
    dataset_by_key = {
        entry["screen_key"]: entry["dataset_name"] for entry in internal_ground_truth_rows
    }
    prediction_runs_by_dataset: Dict[str, List[List[str]]] = {}
    for example_key, prediction_record in pred_map.items():
        dataset_name = dataset_by_key.get(example_key)
        if dataset_name is None:
            continue
        prediction_runs_by_dataset[dataset_name] = prediction_record["predictions"]
    return prediction_runs_by_dataset


def _parse_ct_prediction_text(raw_value: Any) -> List[str]:
    if isinstance(raw_value, list):
        return [str(gene).strip() for gene in raw_value if str(gene).strip()]

    text = str(raw_value).strip()
    if not text:
        return []

    genes: List[str] = []
    for part in text.split(","):
        cleaned = part.strip()
        cleaned = re.sub(r"^\d+\s*[\.\)\-:]\s*", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            genes.append(cleaned)
    return genes


def _load_ct_prediction_runs(ct_predictions_jsons: Sequence[Path]) -> Dict[str, List[List[str]]]:
    prediction_runs_by_dataset: Dict[str, List[List[str]]] = {}
    for ct_predictions_json in ct_predictions_jsons:
        with open(ct_predictions_json) as handle:
            payload = json.load(handle)

        for dataset_name, prediction_value in payload.items():
            genes = _parse_ct_prediction_text(prediction_value)
            if not genes:
                continue
            prediction_runs_by_dataset[str(dataset_name)] = [genes]
    return prediction_runs_by_dataset


def _evaluate_adjusted_ndcg_per_dataset(
    *,
    model_name: str,
    predictions_base_dir: Path,
    ct_predictions_jsons: Sequence[Path],
) -> pd.DataFrame:
    internal_ground_truth_rows = _load_internal_ground_truth()
    frames = [
        _evaluate_runs_against_ground_truth(
            method_name=model_name,
            prediction_runs_by_dataset=_load_gemini_prediction_runs(
                model_name=model_name,
                predictions_base_dir=predictions_base_dir,
                internal_ground_truth_rows=internal_ground_truth_rows,
            ),
            internal_ground_truth_rows=internal_ground_truth_rows,
        )
    ]

    existing_ct_jsons = [path for path in ct_predictions_jsons if path.exists()]
    if existing_ct_jsons:
        frames.append(
            _evaluate_runs_against_ground_truth(
                method_name="CT",
                prediction_runs_by_dataset=_load_ct_prediction_runs(existing_ct_jsons),
                internal_ground_truth_rows=internal_ground_truth_rows,
            )
        )

    return pd.concat(frames, ignore_index=True)


def _build_plot(
    results_df: pd.DataFrame,
    *,
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    plot_df = results_df.dropna(subset=["adjusted_ndcg@100"]).copy()
    if plot_df.empty:
        raise ValueError(f"No internal adjusted_ndcg@100 rows found for model '{model_name}'.")

    plot_df["_group_order"] = plot_df["group"].map(_GROUP_ORDER).fillna(99)
    plot_df = (
        plot_df.sort_values(["_group_order", "screen_order"], ascending=[True, True])
        .drop(columns="_group_order")
        .reset_index(drop=True)
    )
    method_order = [model_name]
    if "CT" in set(plot_df["method"]):
        method_order.append("CT")

    dataset_rows = (
        plot_df[
            ["dataset_name", "dataset_label", "group", "screen_order", "num_hits", "num_genes"]
        ]
        .drop_duplicates()
        .assign(_group_order=lambda df: df["group"].map(_GROUP_ORDER).fillna(99))
        .sort_values(["_group_order", "screen_order"], ascending=[True, True])
        .drop(columns="_group_order")
        .reset_index(drop=True)
    )
    plot_wide = (
        plot_df.pivot_table(
            index="dataset_name",
            columns="method",
            values="adjusted_ndcg@100",
            aggfunc="first",
        )
        .reset_index()
    )
    dataset_rows = dataset_rows.merge(plot_wide, on="dataset_name", how="left")

    apply_style()
    height = max(5.5, 0.42 * len(dataset_rows) + 1.3)
    fig, ax = plt.subplots(figsize=(12, height))

    y_positions = list(range(len(dataset_rows)))
    bar_height = 0.36 if len(method_order) > 1 else 0.68
    if len(method_order) == 1:
        offsets = {method_order[0]: 0.0}
    else:
        offsets = {
            method_order[0]: -bar_height / 2,
            method_order[1]: bar_height / 2,
        }

    for method_name in method_order:
        if method_name not in dataset_rows.columns:
            continue
        method_values = dataset_rows[method_name]
        valid_mask = method_values.notna()
        if not valid_mask.any():
            continue
        method_y = [
            y_positions[idx] + offsets.get(method_name, 0.0)
            for idx, is_valid in enumerate(valid_mask)
            if is_valid
        ]
        ax.barh(
            method_y,
            method_values[valid_mask],
            color=_METHOD_COLORS.get(method_name, "#999999"),
            edgecolor="white",
            height=bar_height,
            label=short_label(method_name),
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(dataset_rows["dataset_label"])
    ax.invert_yaxis()
    ax.set_xlabel("AnDCG@100")
    ax.set_ylabel("Internal dataset")
    if "CT" in method_order:
        title = f"Internal ongoing-screen performance: {short_label(model_name)} vs CT"
    else:
        title = f"Internal ongoing-screen performance for {short_label(model_name)}"
    ax.set_title(title, fontweight="semibold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    max_value = float(plot_df["adjusted_ndcg@100"].max())
    x_padding = max(0.015, max_value * 0.04)
    ax.set_xlim(0.0, max(0.1, max_value + x_padding * 6))

    for method_name in method_order:
        if method_name not in dataset_rows.columns:
            continue
        for idx, value in enumerate(dataset_rows[method_name]):
            if pd.isna(value):
                continue
            ax.text(
                float(value) + x_padding,
                y_positions[idx] + offsets.get(method_name, 0.0),
                f"{float(value):.3f}",
                va="center",
                ha="left",
                fontsize=8,
            )

    group_ranges: Dict[str, tuple[int, int]] = {}
    for group_name, group_rows in dataset_rows.groupby("group", sort=False):
        indices = group_rows.index.tolist()
        group_ranges[group_name] = (min(indices), max(indices))

    ordered_groups = list(group_ranges.keys())
    for boundary_idx in range(len(ordered_groups) - 1):
        _, upper_max = group_ranges[ordered_groups[boundary_idx]]
        ax.axhline(upper_max + 0.5, color="#555555", linewidth=1.0, alpha=0.7)

    yaxis_transform = ax.get_yaxis_transform()
    for group_name in ordered_groups:
        group_min, group_max = group_ranges[group_name]
        group_center = (group_min + group_max) / 2
        ax.text(
            1.01,
            group_center,
            group_name,
            transform=yaxis_transform,
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=_GROUP_COLORS.get(group_name, "#444444"),
            clip_on=False,
        )

    if len(method_order) > 1:
        ax.legend(loc="lower right", framealpha=0.95)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{FILENAME_STEM}__{model_name.replace('-', '_').replace('/', '_')}"
    for fmt in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean adjusted_ndcg@100 across runs for internal ongoing screens "
            "and render one bar per dataset, with optional CT comparison bars."
        ),
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--predictions-base-dir", type=str, default=str(DEFAULT_PREDICTIONS_BASE_DIR))
    parser.add_argument(
        "--ct-predictions-json",
        action="append",
        default=None,
        help=(
            "Path to a CT predictions JSON file. Can be passed multiple times; "
            "all provided files are merged into one CT series."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    predictions_base_dir = Path(args.predictions_base_dir)
    if args.ct_predictions_json is None:
        ct_predictions_jsons = list(DEFAULT_CT_PREDICTIONS_JSONS)
    else:
        ct_predictions_jsons = [Path(path) for path in args.ct_predictions_json]

    results_df = _evaluate_adjusted_ndcg_per_dataset(
        model_name=args.model,
        predictions_base_dir=predictions_base_dir,
        ct_predictions_jsons=ct_predictions_jsons,
    )
    if results_df.empty:
        raise SystemExit("No internal datasets were found in ongoing_screens ground truth.")

    stem = f"{FILENAME_STEM}__{args.model.replace('-', '_').replace('/', '_')}"
    csv_path = output_dir / f"{stem}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)

    _build_plot(
        results_df,
        model_name=args.model,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    plotted = int(results_df["adjusted_ndcg@100"].notna().sum())
    missing = int(results_df["adjusted_ndcg@100"].isna().sum())
    methods_plotted = ", ".join(sorted(results_df.loc[results_df["adjusted_ndcg@100"].notna(), "method"].unique()))
    print(f"Saved dataset metrics to {csv_path}")
    print(f"Plotted {plotted} scored method-dataset rows ({methods_plotted})")
    if missing:
        print(f"{missing} internal datasets were skipped due to missing predictions or scores")


if __name__ == "__main__":
    main()
