#!/usr/bin/env python3
"""Create report-ready comparison graphs from base and structure result tables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RETRIEVAL_METRICS = ["Flickr30 I2T", "Flickr30 T2I", "COCO I2T", "COCO T2I"]
SOURCE_TO_IN_DOMAIN_METRIC = {
    "aircraft": "Aircraft",
    "pets": "Pets",
    "ucf101": "UCF101",
}
SOURCE_DISPLAY = {
    "aircraft": "Aircraft",
    "pets": "Pets",
    "ucf101": "UCF101",
    "coco": "COCO",
}
METHOD_DISPLAY = {
    "base": "CLIP Loss Only",
    "structure": "CLIP Loss + Structure",
}
METHOD_COLOR = {
    "base": "#6B7280",
    "structure": "#0F766E",
}
HEATMAP_ORDER = [
    "STL10",
    "Food101",
    "CIFAR10",
    "CIFAR100",
    "UCF101",
    "MNIST",
    "DTD",
    "GTSRB",
    "Country211",
    "Aircraft",
    "Pets",
    "Flowers",
    "Flickr30 I2T",
    "Flickr30 T2I",
    "COCO I2T",
    "COCO T2I",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create comparison graphs from result tables.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=project_root / "results" / "base",
        help="Directory containing CLIP-loss-only tables.",
    )
    parser.add_argument(
        "--structure-dir",
        type=Path,
        default=project_root / "results" / "tables",
        help="Directory containing structure-regularised tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "plots",
        help="Directory where figures and summary CSVs will be written.",
    )
    parser.add_argument(
        "--metric-kind",
        choices=("top1_acc_micro", "top1_acc_macro"),
        default="top1_acc_micro",
        help="Accuracy table variant to use for the comparison figures.",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
        }
    )


def clean_cell(cell: str) -> str:
    return (
        cell.replace("\\_", "_")
        .replace("\\\\", "")
        .replace("\t", " ")
        .strip()
    )


def parse_latex_table(path: Path) -> dict[str, str]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "&" not in line:
            continue
        if "\\multicolumn" in line:
            continue
        if line.startswith("\\"):
            continue

        parts = [clean_cell(part) for part in line.split("&")]
        if parts[0] == "Model":
            header = parts
            continue
        if header is None:
            continue

        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))

    if header is None or not rows:
        raise ValueError(f"Could not parse a data row from '{path}'.")
    return rows[0]


def parse_numeric(value: str) -> float:
    value = value.strip()
    if value.lower() == "nan":
        return math.nan
    return float(value)


def infer_source_dataset(path: Path) -> str:
    return path.name.split("_", 1)[0]


def load_accuracy_tables(directory: Path, method: str, metric_kind: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in sorted(directory.glob(f"*_{metric_kind}.tex")):
        source_dataset = infer_source_dataset(path)
        row = parse_latex_table(path)
        for metric_name, value in row.items():
            if metric_name == "Model":
                continue
            records.append(
                {
                    "source_dataset": source_dataset,
                    "method": method,
                    "metric_kind": metric_kind,
                    "eval_metric": metric_name.strip(),
                    "value": parse_numeric(value),
                }
            )
    return pd.DataFrame.from_records(records)


def load_loss_tables(directory: Path, method: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.tex")):
        if any(
            path.name.endswith(suffix)
            for suffix in ("_top1_acc_micro.tex", "_top1_acc_macro.tex")
        ):
            continue

        source_dataset = infer_source_dataset(path)
        row = parse_latex_table(path)
        for metric_name, value in row.items():
            if metric_name == "Model":
                continue
            records.append(
                {
                    "source_dataset": source_dataset,
                    "method": method,
                    "loss_metric": metric_name.strip(),
                    "value": parse_numeric(value),
                }
            )
    return pd.DataFrame.from_records(records)


def annotate_bars(ax: plt.Axes, bars, digits: int = 1) -> None:
    y_min, y_max = ax.get_ylim()
    pad = (y_max - y_min) * 0.015
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + pad,
            f"{height:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_grouped_method_bars(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    digits: int = 1,
    methods: list[str] | None = None,
    sources: list[str] | None = None,
) -> None:
    if methods is None:
        methods = ["base", "structure"]
    if sources is None:
        sources = [source for source in SOURCE_DISPLAY if source in summary_df["source_dataset"].unique()]

    filtered_sources = []
    for source in sources:
        source_values = summary_df.loc[
            summary_df["source_dataset"] == source, value_column
        ]
        if source_values.notna().any():
            filtered_sources.append(source)
    sources = filtered_sources

    if not sources:
        ax.set_axis_off()
        return

    x = np.arange(len(sources))
    width = 0.34

    max_value = summary_df[value_column].max()
    max_value = 1.0 if pd.isna(max_value) else float(max_value)

    for offset_index, method in enumerate(methods):
        subset = (
            summary_df[summary_df["method"] == method]
            .set_index("source_dataset")
            .reindex(sources)
            .reset_index()
        )
        offset = (-0.5 + offset_index) * width
        bars = ax.bar(
            x + offset,
            subset[value_column],
            width=width,
            label=METHOD_DISPLAY[method],
            color=METHOD_COLOR[method],
        )
        annotate_bars(ax, bars, digits=digits)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, [SOURCE_DISPLAY[source] for source in sources])
    ax.set_ylim(0, max_value * 1.22 + 0.1)
    ax.legend(frameon=False)


def build_summary_tables(accuracy_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accuracy_wide = (
        accuracy_df.pivot_table(
            index=["source_dataset", "method", "metric_kind"],
            columns="eval_metric",
            values="value",
        )
        .reset_index()
    )

    zero_shot_metrics = [
        column
        for column in accuracy_wide.columns
        if column not in {"source_dataset", "method", "metric_kind"}
        and column not in RETRIEVAL_METRICS
    ]

    summary_rows: list[dict[str, object]] = []
    for _, row in accuracy_wide.iterrows():
        source_dataset = row["source_dataset"]
        in_domain_metric = SOURCE_TO_IN_DOMAIN_METRIC.get(source_dataset)
        transfer_metrics = [
            metric for metric in zero_shot_metrics if metric != in_domain_metric
        ]
        retrieval_values = [row[metric] for metric in RETRIEVAL_METRICS if metric in row]
        flickr_retrieval_values = [
            row[metric]
            for metric in ("Flickr30 I2T", "Flickr30 T2I")
            if metric in row
        ]
        coco_retrieval_values = [
            row[metric]
            for metric in ("COCO I2T", "COCO T2I")
            if metric in row
        ]
        summary_rows.append(
            {
                "source_dataset": source_dataset,
                "method": row["method"],
                "metric_kind": row["metric_kind"],
                "in_domain_accuracy": row[in_domain_metric] if in_domain_metric else math.nan,
                "transfer_accuracy_mean": float(np.nanmean([row[m] for m in transfer_metrics])),
                "flickr_retrieval_mean": float(np.nanmean(flickr_retrieval_values)),
                "coco_retrieval_mean": float(np.nanmean(coco_retrieval_values)),
                "retrieval_r1_mean": float(np.nanmean(retrieval_values)),
            }
        )

    return accuracy_wide, pd.DataFrame.from_records(summary_rows)


def create_in_domain_plot(summary_df: pd.DataFrame, output_dir: Path, metric_kind: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plot_grouped_method_bars(
        ax=ax,
        summary_df=summary_df,
        value_column="in_domain_accuracy",
        title="In-Domain Zero-Shot Accuracy",
        ylabel="Top-1 Accuracy (%)",
    )
    fig.tight_layout()
    save_figure(fig, output_dir / f"in_domain_{metric_kind}")


def create_transfer_retrieval_plot(
    summary_df: pd.DataFrame,
    output_dir: Path,
    metric_kind: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    plot_grouped_method_bars(
        ax=axes[0],
        summary_df=summary_df,
        value_column="transfer_accuracy_mean",
        title="Mean Out-of-Domain Zero-Shot Accuracy",
        ylabel="Top-1 Accuracy (%)",
    )
    plot_grouped_method_bars(
        ax=axes[1],
        summary_df=summary_df,
        value_column="retrieval_r1_mean",
        title="Mean Retrieval Recall@1",
        ylabel="Recall@1 (%)",
    )
    fig.tight_layout()
    save_figure(fig, output_dir / f"transfer_retrieval_summary_{metric_kind}")


def create_structure_source_summary(
    summary_df: pd.DataFrame,
    output_dir: Path,
    metric_kind: str,
) -> None:
    structure_df = summary_df[summary_df["method"] == "structure"].copy()
    ordered_sources = [
        source for source in SOURCE_DISPLAY if source in structure_df["source_dataset"].unique()
    ]
    if structure_df.empty or not ordered_sources:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
    plot_grouped_method_bars(
        ax=axes[0],
        summary_df=structure_df,
        value_column="transfer_accuracy_mean",
        title="Structure Models: Mean Zero-Shot Transfer",
        ylabel="Top-1 Accuracy (%)",
        methods=["structure"],
        sources=ordered_sources,
    )
    plot_grouped_method_bars(
        ax=axes[1],
        summary_df=structure_df,
        value_column="flickr_retrieval_mean",
        title="Structure Models: Flickr30k Retrieval",
        ylabel="Recall@1 (%)",
        methods=["structure"],
        sources=ordered_sources,
    )
    plot_grouped_method_bars(
        ax=axes[2],
        summary_df=structure_df,
        value_column="coco_retrieval_mean",
        title="Structure Models: COCO Retrieval",
        ylabel="Recall@1 (%)",
        methods=["structure"],
        sources=ordered_sources,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / f"structure_source_summary_{metric_kind}")


def create_delta_heatmap(
    accuracy_wide: pd.DataFrame,
    output_dir: Path,
    metric_kind: str,
) -> None:
    base_df = (
        accuracy_wide[accuracy_wide["method"] == "base"]
        .drop(columns=["method", "metric_kind"])
        .set_index("source_dataset")
    )
    structure_df = (
        accuracy_wide[accuracy_wide["method"] == "structure"]
        .drop(columns=["method", "metric_kind"])
        .set_index("source_dataset")
    )

    common_sources = [source for source in SOURCE_DISPLAY if source in base_df.index and source in structure_df.index]
    common_metrics = [
        metric
        for metric in HEATMAP_ORDER
        if metric in base_df.columns and metric in structure_df.columns
    ]
    delta_df = structure_df.loc[common_sources, common_metrics] - base_df.loc[common_sources, common_metrics]
    delta_df.index = [SOURCE_DISPLAY[source] for source in delta_df.index]

    values = delta_df.to_numpy(dtype=float)
    max_abs = np.nanmax(np.abs(values))
    max_abs = max(max_abs, 0.5)

    fig, ax = plt.subplots(figsize=(14.5, 4.8))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_title("Structure Minus CLIP-Loss-Only Performance Delta")
    ax.set_xticks(np.arange(len(common_metrics)), common_metrics, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(delta_df.index)), delta_df.index)

    zero_shot_count = len([metric for metric in common_metrics if metric not in RETRIEVAL_METRICS])
    if 0 < zero_shot_count < len(common_metrics):
        ax.axvline(zero_shot_count - 0.5, color="black", linewidth=1.0, alpha=0.5)

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            text_color = "white" if abs(value) > max_abs * 0.55 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:+.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Percentage-point delta")
    fig.tight_layout()
    save_figure(fig, output_dir / f"structure_minus_base_heatmap_{metric_kind}")

    delta_df.reset_index(names="source_dataset").to_csv(
        output_dir / f"structure_minus_base_heatmap_{metric_kind}.csv",
        index=False,
    )


def create_loss_plot(loss_df: pd.DataFrame, output_dir: Path) -> None:
    loss_wide = (
        loss_df.pivot_table(
            index=["source_dataset", "method"],
            columns="loss_metric",
            values="value",
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    plot_grouped_method_bars(
        ax=axes[0],
        summary_df=loss_wide,
        value_column="val_overall_loss",
        title="Validation Overall Loss",
        ylabel="Loss",
        digits=3,
    )
    plot_grouped_method_bars(
        ax=axes[1],
        summary_df=loss_wide,
        value_column="train_loss",
        title="Training Loss",
        ylabel="Loss",
        digits=3,
    )

    structure_only = (
        loss_wide[loss_wide["method"] == "structure"]
        .set_index("source_dataset")
        .reindex(list(SOURCE_DISPLAY))
        .dropna(subset=["val_structure_loss"], how="all")
        .reset_index()
    )
    x = np.arange(len(structure_only))
    bars = axes[2].bar(
        x,
        structure_only["val_structure_loss"],
        color="#C2410C",
        width=0.55,
    )
    annotate_bars(axes[2], bars, digits=3)
    axes[2].set_title("Validation Structure Loss")
    axes[2].set_ylabel("Loss")
    axes[2].set_xticks(x, [SOURCE_DISPLAY[source] for source in structure_only["source_dataset"]])
    max_structure = float(structure_only["val_structure_loss"].max())
    axes[2].set_ylim(0, max_structure * 1.25 + 0.01)

    fig.tight_layout()
    save_figure(fig, output_dir / "loss_overview")
    loss_wide.to_csv(output_dir / "loss_summary.csv", index=False)


def create_structure_performance_heatmap(
    accuracy_wide: pd.DataFrame,
    output_dir: Path,
    metric_kind: str,
) -> None:
    structure_df = (
        accuracy_wide[accuracy_wide["method"] == "structure"]
        .drop(columns=["method", "metric_kind"])
        .set_index("source_dataset")
    )
    structure_sources = [
        source for source in SOURCE_DISPLAY if source in structure_df.index
    ]
    if not structure_sources:
        return

    common_metrics = [metric for metric in HEATMAP_ORDER if metric in structure_df.columns]
    values_df = structure_df.loc[structure_sources, common_metrics].copy()
    values_df.index = [SOURCE_DISPLAY[source] for source in values_df.index]

    values = values_df.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(14.8, 5.2))
    im = ax.imshow(values, cmap="YlGnBu", aspect="auto")
    ax.set_title("Structure-Regularised Performance by Source Dataset")
    ax.set_xticks(np.arange(len(common_metrics)), common_metrics, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(values_df.index)), values_df.index)

    zero_shot_count = len([metric for metric in common_metrics if metric not in RETRIEVAL_METRICS])
    if 0 < zero_shot_count < len(common_metrics):
        ax.axvline(zero_shot_count - 0.5, color="black", linewidth=1.0, alpha=0.45)

    max_value = np.nanmax(values)
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            text_color = "white" if value > max_value * 0.55 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Performance (%)")
    fig.tight_layout()
    save_figure(fig, output_dir / f"structure_performance_heatmap_{metric_kind}")
    values_df.reset_index(names="source_dataset").to_csv(
        output_dir / f"structure_performance_heatmap_{metric_kind}.csv",
        index=False,
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(f"{output_base}.png", bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_df = pd.concat(
        [
            load_accuracy_tables(args.base_dir, method="base", metric_kind=args.metric_kind),
            load_accuracy_tables(args.structure_dir, method="structure", metric_kind=args.metric_kind),
        ],
        ignore_index=True,
    )
    loss_df = pd.concat(
        [
            load_loss_tables(args.base_dir, method="base"),
            load_loss_tables(args.structure_dir, method="structure"),
        ],
        ignore_index=True,
    )

    accuracy_wide, summary_df = build_summary_tables(accuracy_df)

    accuracy_df.to_csv(args.output_dir / f"accuracy_{args.metric_kind}_long.csv", index=False)
    accuracy_wide.to_csv(args.output_dir / f"accuracy_{args.metric_kind}_wide.csv", index=False)
    summary_df.to_csv(args.output_dir / f"summary_{args.metric_kind}.csv", index=False)

    create_in_domain_plot(summary_df, args.output_dir, args.metric_kind)
    create_transfer_retrieval_plot(summary_df, args.output_dir, args.metric_kind)
    create_structure_source_summary(summary_df, args.output_dir, args.metric_kind)
    create_delta_heatmap(accuracy_wide, args.output_dir, args.metric_kind)
    create_structure_performance_heatmap(accuracy_wide, args.output_dir, args.metric_kind)
    create_loss_plot(loss_df, args.output_dir)

    print(f"Wrote figures and summary CSVs to: {args.output_dir}")


if __name__ == "__main__":
    main()
