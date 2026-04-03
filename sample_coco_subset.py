#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLE_SIZE = 3600


def resolve_coco_root(project_root: Path) -> Path:
    for candidate in (
        project_root / "data" / "processed" / "COCO",
        project_root / "data" / "processed" / "coco",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a COCO data directory under "
        f"'{project_root / 'data' / 'processed'}'."
    )


def get_split_defaults(split: str) -> Tuple[Path, Path, Path]:
    coco_root = resolve_coco_root(PROJECT_ROOT)
    annotations_root = coco_root / "annotations"
    output_root = PROJECT_ROOT / "data" / "subsampled"

    if split == "train":
        return (
            coco_root / "train2014",
            annotations_root / "captions_train2014.json",
            output_root / f"coco_{DEFAULT_SAMPLE_SIZE}.json",
        )

    if split == "val":
        return (
            coco_root / "val2014",
            annotations_root / "captions_val2014.json",
            output_root / f"coco_val_{DEFAULT_SAMPLE_SIZE}.json",
        )

    raise ValueError(f"Unknown COCO split '{split}'.")


def load_coco_records(
    annotation_file: Path,
    image_dir: Path,
) -> Tuple[List[dict], Dict[int, List[str]], Dict[int, str]]:
    with annotation_file.open("r") as file:
        coco_data = json.load(file)

    image_id_to_file = {
        image["id"]: image["file_name"] for image in coco_data.get("images", [])
    }

    pair_records = []
    captions_by_image = defaultdict(list)
    image_id_to_path = {}

    for annotation in coco_data.get("annotations", []):
        image_id = annotation["image_id"]
        caption = annotation.get("caption", "").strip()

        if not caption or image_id not in image_id_to_file:
            continue

        image_path = image_dir / image_id_to_file[image_id]

        record = {
            "image_path": str(image_path),
            "text": caption,
            "label": None,
        }
        pair_records.append(record)
        captions_by_image[image_id].append(caption)
        image_id_to_path[image_id] = str(image_path)

    return pair_records, captions_by_image, image_id_to_path


def create_coco_subsampled_dataset(
    image_dir: Path,
    annotation_file: Path,
    output_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 42,
    sampling_unit: str = "image",
    split_name: str = "train",
) -> None:
    random_generator = random.Random(seed)
    image_dir = Path(image_dir).resolve()
    annotation_file = Path(annotation_file).resolve()
    output_path = Path(output_path)

    pair_records, captions_by_image, image_id_to_path = load_coco_records(
        annotation_file=annotation_file,
        image_dir=image_dir,
    )

    if sampling_unit == "pair":
        available_count = len(pair_records)
        if sample_size > available_count:
            raise ValueError(
                f"Requested {sample_size} pairs, but only found {available_count} valid image-caption pairs."
            )
        sampled_data = random_generator.sample(pair_records, sample_size)
    else:
        available_image_ids = sorted(captions_by_image)
        available_count = len(available_image_ids)
        if sample_size > available_count:
            raise ValueError(
                f"Requested {sample_size} images, but only found {available_count} valid COCO {split_name} images."
            )

        sampled_image_ids = random_generator.sample(available_image_ids, sample_size)
        sampled_data = []

        for image_id in sampled_image_ids:
            sampled_data.append(
                {
                    "image_path": image_id_to_path[image_id],
                    "text": random_generator.choice(captions_by_image[image_id]),
                    "label": None,
                }
            )

    missing_paths = [record["image_path"] for record in sampled_data if not Path(record["image_path"]).exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"Found {len(missing_paths)} sampled records with missing image files. "
            f"First missing path: {missing_paths[0]}"
        )

    random_generator.shuffle(sampled_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(sampled_data, file, indent=4)

    print(
        f"Saved {len(sampled_data)} COCO {split_name} {sampling_unit} samples to {output_path} "
        f"(seed={seed})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a subsampled COCO split for alignment training or validation.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Which COCO split to sample from.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Path to the COCO image directory for the selected split.",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help="Path to the COCO caption annotation file for the selected split.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to write the sampled JSON file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of records to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--sampling-unit",
        choices=["image", "pair"],
        default="image",
        help="Sample unique images with one caption each, or raw image-caption pairs.",
    )
    args = parser.parse_args()

    default_image_dir, default_annotation_file, default_output_path = get_split_defaults(
        args.split
    )

    image_dir = args.image_dir or default_image_dir
    annotation_file = args.annotation_file or default_annotation_file
    output_path = args.output_path or default_output_path

    create_coco_subsampled_dataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        output_path=output_path,
        sample_size=args.sample_size,
        seed=args.seed,
        sampling_unit=args.sampling_unit,
        split_name=args.split,
    )
