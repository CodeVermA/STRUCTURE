#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/data/processed"
DOWNLOAD_ROOT="${DATA_ROOT}/_downloads"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") caltech101
  $(basename "$0") flickr30k
  $(basename "$0") public
  $(basename "$0") imagenet /path/to/imagenet_root
  $(basename "$0") verify

Commands:
  caltech101  Download and prepare Caltech-101 under data/processed/caltech-101
  flickr30k   Download and prepare Flickr30k under data/processed/flickr30k_images
  public      Run both public dataset steps above
  imagenet    Link an existing ImageNet copy into data/processed/imagenet
  verify      Print the dataset paths this repo currently expects

ImageNet source layouts supported:
  1. Prepared layout:
     <root>/train
     <root>/val
     <root>/LOC_synset_mapping.txt

  2. Kaggle / CLS-LOC style layout:
     <root>/ILSVRC/Data/CLS-LOC/train
     <root>/ILSVRC/Data/CLS-LOC/val
     <root>/LOC_synset_mapping.txt
     <root>/LOC_val_solution.csv

Notes:
  - Caltech-101 is fetched from official CaltechDATA.
  - Flickr30k is fetched from a Hugging Face mirror, then converted into the
    exact file layout this repo's loader expects.
  - Full ImageNet download is not automated here because it requires official
    access. Use the 'imagenet' command after obtaining a local copy.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

ensure_common_tools() {
  require_cmd wget
  require_cmd unzip
  require_cmd python
}

download_caltech101() {
  ensure_common_tools
  local target_dir="${DATA_ROOT}/caltech-101/101_ObjectCategories"
  local zip_path="${DOWNLOAD_ROOT}/caltech-101.zip"
  local extract_dir="${DOWNLOAD_ROOT}/caltech_extract"

  mkdir -p "${DOWNLOAD_ROOT}" "${DATA_ROOT}/caltech-101"
  if [ -d "${target_dir}" ]; then
    echo "Caltech-101 already present at ${target_dir}"
    return
  fi

  wget -O "${zip_path}" \
    "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"

  rm -rf "${extract_dir}"
  mkdir -p "${extract_dir}"
  unzip -q -o "${zip_path}" -d "${extract_dir}"

  if [ -d "${extract_dir}/caltech-101/101_ObjectCategories" ]; then
    mv "${extract_dir}/caltech-101/101_ObjectCategories" "${DATA_ROOT}/caltech-101/"
  elif [ -d "${extract_dir}/101_ObjectCategories" ]; then
    mv "${extract_dir}/101_ObjectCategories" "${DATA_ROOT}/caltech-101/"
  else
    echo "Could not find 101_ObjectCategories after extracting Caltech-101." >&2
    exit 1
  fi

  echo "Prepared ${target_dir}"
}

download_flickr30k() {
  ensure_common_tools
  local dataset_root="${DATA_ROOT}/flickr30k_images"
  local images_zip="${DOWNLOAD_ROOT}/flickr30k-images.zip"
  local annotations_csv="${DOWNLOAD_ROOT}/flickr_annotations_30k.csv"
  local extract_dir="${DOWNLOAD_ROOT}/flickr_extract"

  mkdir -p "${DOWNLOAD_ROOT}" "${dataset_root}"
  if [ -f "${dataset_root}/results.csv" ] && [ -d "${dataset_root}/flickr30k_images" ]; then
    echo "Flickr30k already present at ${dataset_root}"
    return
  fi

  wget -O "${images_zip}" \
    "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip"
  wget -O "${annotations_csv}" \
    "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr_annotations_30k.csv"

  rm -rf "${extract_dir}" "${dataset_root}/flickr30k_images"
  mkdir -p "${extract_dir}"
  unzip -q -o "${images_zip}" -d "${extract_dir}"

  if [ -d "${extract_dir}/flickr30k-images" ]; then
    mv "${extract_dir}/flickr30k-images" "${dataset_root}/flickr30k_images"
  elif [ -d "${extract_dir}/flickr30k_images" ]; then
    mv "${extract_dir}/flickr30k_images" "${dataset_root}/flickr30k_images"
  else
    echo "Could not find extracted Flickr30k image directory." >&2
    exit 1
  fi

  FLICKR_ROOT="${dataset_root}" FLICKR_SOURCE="${annotations_csv}" python - <<'PY'
import ast
import csv
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["FLICKR_ROOT"])
source = Path(os.environ["FLICKR_SOURCE"])
df = pd.read_csv(source)

rows = []
for _, row in df.iterrows():
    captions = row["raw"]
    if isinstance(captions, str):
        try:
            captions = ast.literal_eval(captions)
        except Exception:
            captions = [captions]
    for index, caption in enumerate(captions):
        rows.append(
            {
                "image_name": row["filename"],
                "comment_number": index,
                "comment": caption,
            }
        )

pd.DataFrame(rows).to_csv(
    root / "results.csv",
    sep="|",
    index=False,
    quoting=csv.QUOTE_MINIMAL,
)

for split in ("train", "val", "test"):
    names = (
        df.loc[df["split"] == split, "filename"]
        .astype(str)
        .str.replace(".jpg", "", regex=False)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    (root / f"{split}.txt").write_text("\n".join(names) + "\n")
PY

  echo "Prepared ${dataset_root}"
}

prepare_imagenet() {
  if [ $# -lt 1 ]; then
    echo "ImageNet preparation requires a source directory." >&2
    usage
    exit 1
  fi

  local source_root
  source_root="$(cd "$1" && pwd)"
  local target_root="${DATA_ROOT}/imagenet"

  mkdir -p "${DATA_ROOT}"
  rm -rf "${target_root}"
  mkdir -p "${target_root}"

  if [ -d "${source_root}/train" ] && [ -d "${source_root}/val" ] && [ -f "${source_root}/LOC_synset_mapping.txt" ]; then
    ln -s "${source_root}/train" "${target_root}/train"
    ln -s "${source_root}/val" "${target_root}/val"
    ln -s "${source_root}/LOC_synset_mapping.txt" "${target_root}/LOC_synset_mapping.txt"
    echo "Linked prepared ImageNet layout from ${source_root}"
    return
  fi

  local cls_loc_root="${source_root}/ILSVRC/Data/CLS-LOC"
  local train_root="${cls_loc_root}/train"
  local val_root="${cls_loc_root}/val"
  local mapping_file="${source_root}/LOC_synset_mapping.txt"
  local val_solution="${source_root}/LOC_val_solution.csv"

  if [ ! -d "${train_root}" ] || [ ! -d "${val_root}" ] || [ ! -f "${mapping_file}" ]; then
    echo "Unsupported ImageNet source layout: ${source_root}" >&2
    usage
    exit 1
  fi

  ln -s "${train_root}" "${target_root}/train"
  ln -s "${mapping_file}" "${target_root}/LOC_synset_mapping.txt"

  if find "${val_root}" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    ln -s "${val_root}" "${target_root}/val"
    echo "Linked class-organized ImageNet validation directory from ${source_root}"
    return
  fi

  if [ ! -f "${val_solution}" ]; then
    echo "Validation images are flat and LOC_val_solution.csv is missing at ${source_root}" >&2
    exit 1
  fi

  mkdir -p "${target_root}/val"
  python - "${val_root}" "${val_solution}" "${target_root}/val" <<'PY'
import csv
import os
import sys
from pathlib import Path

val_images = Path(sys.argv[1])
solution_file = Path(sys.argv[2])
target_root = Path(sys.argv[3])

with solution_file.open(newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        image_id = row["ImageId"]
        prediction = row["PredictionString"].split()[0]
        source = val_images / image_id
        target_dir = target_root / prediction
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / image_id
        if not target.exists():
            os.symlink(source, target)
PY

  echo "Prepared ImageNet validation symlinks under ${target_root}/val"
}

verify_layout() {
  printf '%s\n' \
    "${DATA_ROOT}/caltech-101/101_ObjectCategories" \
    "${DATA_ROOT}/flickr30k_images/results.csv" \
    "${DATA_ROOT}/flickr30k_images/train.txt" \
    "${DATA_ROOT}/flickr30k_images/val.txt" \
    "${DATA_ROOT}/flickr30k_images/test.txt" \
    "${DATA_ROOT}/flickr30k_images/flickr30k_images" \
    "${DATA_ROOT}/imagenet/train" \
    "${DATA_ROOT}/imagenet/val" \
    "${DATA_ROOT}/imagenet/LOC_synset_mapping.txt"
}

main() {
  if [ $# -lt 1 ]; then
    usage
    exit 1
  fi

  case "$1" in
    caltech101)
      download_caltech101
      ;;
    flickr30k)
      download_flickr30k
      ;;
    public)
      download_caltech101
      download_flickr30k
      ;;
    imagenet)
      shift
      prepare_imagenet "$@"
      ;;
    verify)
      verify_layout
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "Unknown command: $1" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
