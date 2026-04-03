#!/usr/bin/env python3
"""Create a paper-style LaTeX ablation table from existing result tables."""

from __future__ import annotations

import argparse
from pathlib import Path


DISPLAY_NAME = {
    "aircraft": "Aircraft",
    "pets": "Pets",
    "ucf101": "UCF101",
    "coco": "COCO",
}

SECTION_ROWS = [
    {
        "title": "Aircraft source dataset",
        "rows": [
            ("CLIP loss only", "base", "aircraft"),
            ("CLIP loss + STRUCTURE", "tables", "aircraft"),
        ],
    },
    {
        "title": "Pets source dataset",
        "rows": [
            ("CLIP loss only", "base", "pets"),
            ("CLIP loss + STRUCTURE", "tables", "pets"),
        ],
    },
    {
        "title": "UCF101 source dataset",
        "rows": [
            ("CLIP loss only", "base", "ucf101"),
            ("CLIP loss + STRUCTURE", "tables", "ucf101"),
        ],
    },
    {
        "title": "Captioned source dataset",
        "rows": [
            ("COCO + STRUCTURE", "tables", "coco"),
        ],
    },
]

TARGET_COLUMNS = ["STL10", "CIFAR10", "Food101", "CIFAR100"]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate a paper-style ablation table from result tables.",
    )
    parser.add_argument(
        "--metric-kind",
        choices=("top1_acc_micro", "top1_acc_macro"),
        default="top1_acc_micro",
        help="Which accuracy variant to read from the result tables.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to write the generated LaTeX table.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Repository root.",
    )
    return parser.parse_args()


def clean_cell(cell: str) -> str:
    return cell.replace("\\_", "_").replace("\\\\", "").strip()


def parse_latex_row(path: Path) -> dict[str, float]:
    header = None
    values = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "&" not in line or "\\multicolumn" in line or line.startswith("\\"):
            continue

        parts = [clean_cell(part) for part in line.split("&")]
        if parts[0] == "Model":
            header = parts
            continue
        if header is not None:
            values = parts
            break

    if header is None or values is None:
        raise ValueError(f"Could not parse data row from '{path}'.")

    row = {}
    for key, value in zip(header[1:], values[1:]):
        row[key] = float(value)
    return row


def load_metric_row(project_root: Path, directory_name: str, dataset: str, metric_kind: str) -> dict[str, float]:
    table_path = (
        project_root
        / "results"
        / directory_name
        / f"{dataset}_sentence-transformers_all-roberta-large-v1_vit_large_patch14_dinov2.lvd142m_{metric_kind}.tex"
    )
    if not table_path.exists():
        raise FileNotFoundError(f"Missing result table: {table_path}")
    return parse_latex_row(table_path)


def format_value(value: float) -> str:
    return f"{value:.1f}"


def render_section(title: str, rows: list[tuple[str, str, str]], metrics_by_key: dict[tuple[str, str], dict[str, float]]) -> list[str]:
    lines = [rf"\emph{{{title}}} \\"]

    section_values = {
        column: max(metrics_by_key[(directory_name, dataset)][column] for _, directory_name, dataset in rows)
        for column in TARGET_COLUMNS
    }

    for label, directory_name, dataset in rows:
        metrics = metrics_by_key[(directory_name, dataset)]
        formatted_metrics = []
        for column in TARGET_COLUMNS:
            value = metrics[column]
            value_str = format_value(value)
            if value == section_values[column]:
                value_str = rf"\textbf{{{value_str}}}"
            formatted_metrics.append(value_str)
        lines.append(f"{label} & " + " & ".join(formatted_metrics) + r" \\")
    return lines


def build_table(metric_kind: str, project_root: Path) -> str:
    metrics_by_key = {}
    for section in SECTION_ROWS:
        for _, directory_name, dataset in section["rows"]:
            key = (directory_name, dataset)
            if key not in metrics_by_key:
                metrics_by_key[key] = load_metric_row(
                    project_root=project_root,
                    directory_name=directory_name,
                    dataset=dataset,
                    metric_kind=metric_kind,
                )

    caption_metric = "micro-averaged" if metric_kind == "top1_acc_micro" else "macro-averaged"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Ablation} & \textbf{STL10} & \textbf{CIFAR10} & \textbf{Food101} & \textbf{CIFAR100} \\",
        r"\midrule",
    ]

    for section_index, section in enumerate(SECTION_ROWS):
        lines.extend(render_section(section["title"], section["rows"], metrics_by_key))
        if section_index != len(SECTION_ROWS) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Ablation-style comparison using the available real experiments. "
                rf"Values report {caption_metric} top-1 accuracy (\%) on four benchmark datasets. "
                r"Within each section, the best value is shown in bold.}"
            ),
            rf"\label{{tab:ablation_{metric_kind}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_path = args.output_path
    if output_path is None:
        output_path = (
            args.project_root
            / "results"
            / "tables"
            / f"ablation_style_{args.metric_kind}.tex"
        )

    table_str = build_table(metric_kind=args.metric_kind, project_root=args.project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_str, encoding="utf-8")
    print(f"Wrote LaTeX table to: {output_path}")


if __name__ == "__main__":
    main()
