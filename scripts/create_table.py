#!/usr/bin/env python3
"""Create report-friendly tables from an offline W&B run.

Example:
    python scripts/create_table.py 20260401_002158-kki0jhvh
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


SUMMARY_FILE = "wandb-summary.json"
CONFIG_FILE = "config.yaml"
REQUIRED_METRICS = (
    "val_loss",
    "val_loss_avg",
    "val_overall_loss",
    "train_loss",
    "train_loss_avg",
)
OPTIONAL_METRICS = ("val_structure_loss",)
DISPLAY_NAME_MAP = {
    "aircraft": "Aircraft",
    "birdsnap": "Birdsnap",
    "caltech101": "Caltech101",
    "cars": "Cars",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "clevr": "CLEVR",
    "coco": "COCO",
    "country211": "Country211",
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "fer2013": "FER2013",
    "flickr30": "Flickr30",
    "flowers": "Flowers",
    "food101": "Food101",
    "gtsrb": "GTSRB",
    "hatefulmemes": "HatefulMemes",
    "imagenet": "ImageNet",
    "kinetics700": "Kinetics700",
    "kitti": "KITTI",
    "mnist": "MNIST",
    "pcam": "PCAM",
    "pets": "Pets",
    "resisc45": "RESISC45",
    "sst": "SST",
    "stl10": "STL10",
    "sun397": "SUN397",
    "ucf101": "UCF101",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create summary tables from a W&B offline run.",
    )
    parser.add_argument(
        "job_id",
        help=(
            "W&B offline job id such as '20260401_002158-kki0jhvh'. "
            "A full run path, files directory, or summary file path is also accepted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "tables",
        help="Directory where table files will be written.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("csv", "md", "tex"),
        default=("tex",),
        help="Table formats to generate. Defaults to 'tex'.",
    )
    return parser.parse_args()


def sanitize_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())


def get_display_name(dataset_name: str) -> str:
    if dataset_name in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[dataset_name]
    return dataset_name.replace("_", " ").replace("-", " ").title()


def infer_job_id_from_path(files_dir: Path) -> str:
    run_dir = files_dir.parent
    if run_dir.name.startswith("offline-run-"):
        return run_dir.name.removeprefix("offline-run-")
    return run_dir.name


def resolve_files_dir(job_id_or_path: str, project_root: Path) -> tuple[Path, str]:
    candidate = Path(job_id_or_path).expanduser()
    if candidate.exists():
        resolved = candidate.resolve()
        if resolved.is_file():
            if resolved.name not in {SUMMARY_FILE, CONFIG_FILE}:
                raise ValueError(
                    f"Unsupported file path '{resolved}'. Expected {SUMMARY_FILE} or {CONFIG_FILE}."
                )
            files_dir = resolved.parent
            return files_dir, infer_job_id_from_path(files_dir)

        if (resolved / SUMMARY_FILE).exists() and (resolved / CONFIG_FILE).exists():
            return resolved, infer_job_id_from_path(resolved)

        files_dir = resolved / "files"
        if (files_dir / SUMMARY_FILE).exists() and (files_dir / CONFIG_FILE).exists():
            return files_dir, infer_job_id_from_path(files_dir)

        raise ValueError(
            f"Could not find both {SUMMARY_FILE} and {CONFIG_FILE} under '{resolved}'."
        )

    files_dir = project_root / "wandb" / f"offline-run-{job_id_or_path}" / "files"
    if files_dir.exists():
        return files_dir, str(job_id_or_path)

    matches = sorted((project_root / "wandb").glob(f"*{job_id_or_path}*/files"))
    if len(matches) == 1:
        return matches[0], infer_job_id_from_path(matches[0])
    if len(matches) > 1:
        raise ValueError(
            f"Multiple W&B runs matched '{job_id_or_path}'. Please pass a more specific job id."
        )
    raise FileNotFoundError(
        f"Unable to find offline W&B run for job id '{job_id_or_path}'."
    )


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def unwrap_wandb_config(raw_config: dict) -> dict:
    cleaned = {}
    for key, value in raw_config.items():
        if isinstance(value, dict) and "value" in value:
            cleaned[key] = value["value"]
        else:
            cleaned[key] = value
    return cleaned


def detect_metric_prefix(summary: dict) -> str:
    if all(metric in summary for metric in REQUIRED_METRICS):
        return ""

    candidates: set[str] = set()
    for key in summary:
        for metric in REQUIRED_METRICS:
            suffix = f"/{metric}"
            if key.endswith(suffix):
                candidates.add(key[: -len(suffix)])

    valid_prefixes = [
        prefix
        for prefix in sorted(candidates)
        if all(f"{prefix}/{metric}" in summary for metric in REQUIRED_METRICS)
    ]
    if not valid_prefixes:
        raise KeyError(
            "Could not find all requested metrics in the W&B summary file. "
            f"Expected keys: {', '.join(REQUIRED_METRICS)}."
        )

    if len(valid_prefixes) == 1:
        return valid_prefixes[0]

    # Prefer the prefix that owns the most summary entries when several are present.
    valid_prefixes.sort(
        key=lambda prefix: (-sum(key.startswith(f"{prefix}/") for key in summary), prefix)
    )
    return valid_prefixes[0]


def extract_metrics(summary: dict, prefix: str) -> dict[str, float]:
    metrics = {}
    missing = []
    for metric in REQUIRED_METRICS:
        key = metric if not prefix else f"{prefix}/{metric}"
        if key not in summary:
            missing.append(key)
            continue
        metrics[metric] = summary[key]

    if missing:
        raise KeyError(
            "The summary file is missing required metrics: " + ", ".join(missing)
        )

    for metric in OPTIONAL_METRICS:
        key = metric if not prefix else f"{prefix}/{metric}"
        metrics[metric] = summary.get(key)
    return metrics


def build_flat_table(model_name: str, metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": model_name,
                "val_loss": metrics["val_loss"],
                "val_loss_avg": metrics["val_loss_avg"],
                "val_overall_loss": metrics["val_overall_loss"],
                "val_structure_loss": metrics.get("val_structure_loss"),
                "train_loss": metrics["train_loss"],
                "train_loss_avg": metrics["train_loss_avg"],
            }
        ]
    )


def build_latex_table(model_name: str, metrics: dict[str, float]) -> pd.DataFrame:
    columns = pd.MultiIndex.from_tuples(
        [
            ("", "Model"),
            ("Validation", "val_loss"),
            ("Validation", "val_loss_avg"),
            ("Validation", "val_overall_loss"),
            ("Validation", "val_structure_loss"),
            ("Training", "train_loss"),
            ("Training", "train_loss_avg"),
        ]
    )
    row = [
        model_name,
        metrics["val_loss"],
        metrics["val_loss_avg"],
        metrics["val_overall_loss"],
        metrics.get("val_structure_loss"),
        metrics["train_loss"],
        metrics["train_loss_avg"],
    ]
    return pd.DataFrame([row], columns=columns)


def get_summary_value(summary: dict, prefix: str, suffix: str) -> float | None:
    key = suffix if not prefix else f"{prefix}/{suffix}"
    return summary.get(key)


def collect_accuracy_metrics(
    summary: dict,
    prefix: str,
    config: dict,
    metric_name: str,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    zero_shot_metrics = []
    retrieval_metrics = []

    for dataset_name in config["evaluation"]["zero_shot_datasets"]:
        metric_value = get_summary_value(summary, prefix, f"{dataset_name}/{metric_name}")
        if metric_value is None:
            continue
        zero_shot_metrics.append(
            (get_display_name(dataset_name), float(metric_value) * 100)
        )

    for dataset_name in config["evaluation"]["retrieval_datasets"]:
        i2t_value = get_summary_value(summary, prefix, f"{dataset_name}/I2T-R@1")
        if i2t_value is not None:
            retrieval_metrics.append(
                (f"{get_display_name(dataset_name)} I2T", float(i2t_value) * 100)
            )

        t2i_value = get_summary_value(summary, prefix, f"{dataset_name}/T2I-R@1")
        if t2i_value is not None:
            retrieval_metrics.append(
                (f"{get_display_name(dataset_name)} T2I", float(t2i_value) * 100)
            )

    return zero_shot_metrics, retrieval_metrics


def build_accuracy_flat_table(
    model_name: str,
    zero_shot_metrics: list[tuple[str, float]],
    retrieval_metrics: list[tuple[str, float]],
) -> pd.DataFrame:
    row = {"Model": model_name}
    for label, value in zero_shot_metrics + retrieval_metrics:
        row[label] = value
    return pd.DataFrame([row])


def build_accuracy_latex_table(
    model_name: str,
    zero_shot_metrics: list[tuple[str, float]],
    retrieval_metrics: list[tuple[str, float]],
) -> pd.DataFrame:
    columns = [("", "Model")]
    columns.extend(
        ("Zero-shot Classification (Accuracy)", label) for label, _ in zero_shot_metrics
    )
    columns.extend(("Retrieval (R@1)", label) for label, _ in retrieval_metrics)

    row = [model_name]
    row.extend(value for _, value in zero_shot_metrics)
    row.extend(value for _, value in retrieval_metrics)
    return pd.DataFrame([row], columns=pd.MultiIndex.from_tuples(columns))


def format_value(value: object, float_digits: int) -> str:
    if isinstance(value, float):
        return f"{value:.{float_digits}f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, float_digits: int) -> str:
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(format_value(value, float_digits=float_digits) for value in row)
            + " |"
        )
    return "\n".join(lines) + "\n"


def normalize_formats(formats: Iterable[str] | str) -> list[str]:
    if isinstance(formats, str):
        return [formats]
    return list(formats)


def write_outputs(
    base_path: Path,
    flat_table: pd.DataFrame,
    latex_table: pd.DataFrame,
    formats: Iterable[str],
    float_digits: int,
) -> list[Path]:
    output_paths = []
    for fmt in normalize_formats(formats):
        output_path = base_path.parent / f"{base_path.name}.{fmt}"
        if fmt == "csv":
            flat_table.to_csv(
                output_path,
                index=False,
                float_format=f"%.{float_digits}f",
            )
        elif fmt == "md":
            output_path.write_text(
                dataframe_to_markdown(flat_table, float_digits=float_digits),
                encoding="utf-8",
            )
        elif fmt == "tex":
            latex_str = latex_table.to_latex(
                index=False,
                escape=True,
                multicolumn=True,
                multicolumn_format="c",
                float_format=lambda value: f"{value:.{float_digits}f}",
            )
            output_path.write_text(latex_str, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported format '{fmt}'.")
        output_paths.append(output_path)
    return output_paths


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    files_dir, resolved_job_id = resolve_files_dir(args.job_id, project_root)

    summary = load_json(files_dir / SUMMARY_FILE)
    config = unwrap_wandb_config(load_yaml(files_dir / CONFIG_FILE))

    dataset = config["features"]["dataset"]
    llm_model = config["alignment"]["llm_model_name"]
    lvm_model = config["alignment"]["lvm_model_name"]

    metric_prefix = detect_metric_prefix(summary)
    metrics = extract_metrics(summary, metric_prefix)

    loss_flat_table = build_flat_table(dataset, metrics)
    loss_latex_table = build_latex_table(dataset, metrics)
    macro_zero_shot, macro_retrieval = collect_accuracy_metrics(
        summary=summary,
        prefix=metric_prefix,
        config=config,
        metric_name="top1_acc_macro",
    )
    micro_zero_shot, micro_retrieval = collect_accuracy_metrics(
        summary=summary,
        prefix=metric_prefix,
        config=config,
        metric_name="top1_acc_micro",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = (
        f"{sanitize_filename_part(dataset)}_"
        f"{sanitize_filename_part(llm_model)}_"
        f"{sanitize_filename_part(lvm_model)}"
    )
    output_paths = []

    loss_output_base = args.output_dir / file_stem
    output_paths.extend(
        write_outputs(
            loss_output_base,
            loss_flat_table,
            loss_latex_table,
            args.formats,
            float_digits=4,
        )
    )

    if macro_zero_shot or macro_retrieval:
        macro_output_base = args.output_dir / f"{file_stem}_top1_acc_macro"
        output_paths.extend(
            write_outputs(
                macro_output_base,
                build_accuracy_flat_table(dataset, macro_zero_shot, macro_retrieval),
                build_accuracy_latex_table(dataset, macro_zero_shot, macro_retrieval),
                args.formats,
                float_digits=1,
            )
        )

    if micro_zero_shot or micro_retrieval:
        micro_output_base = args.output_dir / f"{file_stem}_top1_acc_micro"
        output_paths.extend(
            write_outputs(
                micro_output_base,
                build_accuracy_flat_table(dataset, micro_zero_shot, micro_retrieval),
                build_accuracy_latex_table(dataset, micro_zero_shot, micro_retrieval),
                args.formats,
                float_digits=1,
            )
        )

    print(f"Resolved job id: {resolved_job_id}")
    print(f"Loaded files from: {files_dir}")
    if metric_prefix:
        print(f"Using summary metric prefix: {metric_prefix}")
    else:
        print("Using unprefixed summary metrics.")
    print("Generated:")
    for output_path in output_paths:
        print(f"  - {output_path}")


if __name__ == "__main__":
    main()
