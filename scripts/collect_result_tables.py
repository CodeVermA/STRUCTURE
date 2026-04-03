#!/usr/bin/env python3
"""Collect all result tables into a single labelled LaTeX file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MODEL_STEM = "sentence-transformers_all-roberta-large-v1_vit_large_patch14_dinov2.lvd142m"
SOURCE_DISPLAY = {
    "aircraft": "Aircraft",
    "pets": "Pets",
    "ucf101": "UCF101",
    "coco": "COCO",
}
DIRECTORY_DISPLAY = {
    "base": "CLIP loss only results",
    "tables": "CLIP loss + STRUCTURE results",
}
DIRECTORY_CAPTION_STYLE = {
    "base": "CLIP-loss-only",
    "tables": "CLIP loss + STRUCTURE",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Collect all result tables into one LaTeX file with captions and labels.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Repository root.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_root / "results" / "tables" / "all_result_tables.tex",
        help="Path for the combined LaTeX output.",
    )
    return parser.parse_args()


def prettify_source(name: str) -> str:
    return SOURCE_DISPLAY.get(name, name.replace("_", " ").title())


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def classify_table(path: Path) -> tuple[str, str]:
    name = path.stem
    if name.startswith("ablation_style_"):
        metric_kind = name.removeprefix("ablation_style_")
        return "ablation", metric_kind
    if name.endswith("_top1_acc_micro"):
        return "accuracy_micro", name.removesuffix("_top1_acc_micro")
    if name.endswith("_top1_acc_macro"):
        return "accuracy_macro", name.removesuffix("_top1_acc_macro")
    return "loss", name


def extract_source_dataset(stem: str) -> str:
    if stem.startswith("ablation_style_"):
        return "ablation"
    if f"_{MODEL_STEM}" in stem:
        return stem.split(f"_{MODEL_STEM}", 1)[0]
    return stem.split("_", 1)[0]


def build_caption(path: Path, table_type: str, source_dataset: str, suffix: str) -> str:
    source_display = prettify_source(source_dataset)
    directory_label = DIRECTORY_CAPTION_STYLE.get(path.parent.name, path.parent.name)

    if table_type == "loss":
        return f"{directory_label} loss summary for the {source_display} source dataset."
    if table_type == "accuracy_micro":
        return (
            f"{directory_label} zero-shot classification and retrieval results for the "
            f"{source_display} source dataset using micro-averaged top-1 accuracy."
        )
    if table_type == "accuracy_macro":
        return (
            f"{directory_label} zero-shot classification and retrieval results for the "
            f"{source_display} source dataset using macro-averaged top-1 accuracy."
        )
    if table_type == "ablation":
        metric_label = "micro-averaged" if suffix == "top1_acc_micro" else "macro-averaged"
        return (
            "Ablation-style comparison derived from the available experiments using "
            f"{metric_label} top-1 accuracy."
        )
    return f"Results table for {source_display}."


def build_label(path: Path, table_type: str, source_dataset: str, suffix: str) -> str:
    if table_type == "ablation":
        return f"tab:{slugify(path.stem)}"
    return f"tab:{path.parent.name}_{slugify(source_dataset)}_{slugify(table_type)}"


def wrap_tabular(content: str, caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            content.strip(),
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )


def ensure_label(table_tex: str, desired_label: str) -> str:
    if r"\label{" in table_tex:
        table_tex = re.sub(r"\\label\{[^}]+\}", rf"\\label{{{desired_label}}}", table_tex, count=1)
        return table_tex
    return table_tex.replace(r"\end{table}", rf"\label{{{desired_label}}}" + "\n" + r"\end{table}", 1)


def sort_key(path: Path) -> tuple[int, str, str]:
    directory_order = {"base": 0, "tables": 1}.get(path.parent.name, 2)
    table_type, _ = classify_table(path)
    type_order = {"loss": 0, "accuracy_micro": 1, "accuracy_macro": 2, "ablation": 3}.get(table_type, 4)
    return directory_order, type_order, path.name


def collect_tables(project_root: Path, output_path: Path) -> list[Path]:
    table_paths = list((project_root / "results" / "base").glob("*.tex"))
    table_paths.extend((project_root / "results" / "tables").glob("*.tex"))
    output_path = output_path.resolve()
    filtered_paths = [
        path for path in table_paths if path.resolve() != output_path
    ]
    return sorted(filtered_paths, key=sort_key)


def build_combined_file(project_root: Path, output_path: Path) -> str:
    lines = [
        "% Auto-generated by scripts/collect_result_tables.py",
        "% This file groups all tables from results/base and results/tables with captions and labels.",
        "",
    ]

    current_directory = None
    for path in collect_tables(project_root, output_path):
        directory_name = path.parent.name
        if directory_name != current_directory:
            section_title = DIRECTORY_DISPLAY.get(directory_name, directory_name.title())
            lines.extend([rf"\section*{{{section_title}}}", ""])
            current_directory = directory_name

        table_type, suffix = classify_table(path)
        source_dataset = extract_source_dataset(path.stem)
        caption = build_caption(path, table_type, source_dataset, suffix)
        label = build_label(path, table_type, source_dataset, suffix)
        content = path.read_text(encoding="utf-8").strip()

        if content.lstrip().startswith(r"\begin{table}"):
            content = ensure_label(content, label)
            if r"\caption{" not in content:
                content = content.replace(r"\end{tabular}", r"\end{tabular}" + "\n" + rf"\caption{{{caption}}}", 1)
            lines.extend([content, ""])
        else:
            lines.extend([wrap_tabular(content, caption, label), ""])

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    combined_tex = build_combined_file(args.project_root, args.output_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(combined_tex, encoding="utf-8")
    print(f"Wrote combined LaTeX tables to: {args.output_path}")


if __name__ == "__main__":
    main()
