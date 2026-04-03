import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torchvision.transforms as transforms
import yaml
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.core.src.datasets.image_text_dataset import ImageTextDataset
from src.core.src.utils.loader import Loader, merge_dicts
from src.dataset_preparation.data_utils import get_datasets, get_default_transforms
from src.trainers.alignment_trainer import AlignmentTrainer
from src.trainers.clip_eval_trainer import CLIPEvalTrainer
from src.trainers.csa_trainer import CSATrainer


class SubsampledImageTextDataset(Dataset):
    def __init__(
        self,
        dataset_file: Path,
        transform=None,
    ):
        super().__init__()
        self.dataset_file = Path(dataset_file)
        with open(self.dataset_file, "r") as f:
            records = json.load(f)

        self.df = pd.DataFrame(records)
        missing_columns = {"image_path", "text"} - set(self.df.columns)
        if missing_columns:
            missing_columns = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Subsampled dataset '{self.dataset_file}' is missing required columns: {missing_columns}"
            )

        self.df = self.df.rename(columns={"text": "captions"})
        if "label" not in self.df.columns:
            self.df["label"] = None
        self.df["image_path"] = self.df["image_path"].astype(str)

        self.transform = transform
        self.name = self.dataset_file.stem
        self.tokenizer = None
        self.tokens = None

    def apply_tokenizer(self) -> None:
        if self.tokenizer:
            self.tokens = self.tokenizer(
                list(self.df["captions"].values),
                padding="longest",
                return_tensors="pt",
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.tokenizer:
            caption = {k: v[index] for (k, v) in self.tokens.items()}
            return image, caption
        return image, row["captions"]


def _get_subsampled_search_dirs(
    data_path: Path,
    subsampled_data_path: Optional[Path] = None,
) -> List[Path]:
    if subsampled_data_path is not None:
        return [Path(subsampled_data_path)]

    search_dirs = [data_path / "subsampled", data_path.parent / "subsampled"]
    unique_dirs = []
    for directory in search_dirs:
        if directory not in unique_dirs:
            unique_dirs.append(directory)
    return unique_dirs


def _get_subsampled_sample_count(dataset_file: Path) -> int:
    suffix = dataset_file.stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else -1


def resolve_subsampled_dataset_file(
    dataset_name: str,
    data_path: Path,
    subsampled_data_path: Optional[Path] = None,
    subsampled_dataset_file: Optional[str] = None,
) -> Optional[Path]:
    search_dirs = _get_subsampled_search_dirs(
        data_path=data_path,
        subsampled_data_path=subsampled_data_path,
    )

    if subsampled_dataset_file is not None:
        return resolve_subsampled_dataset_path(
            dataset_file=subsampled_dataset_file,
            data_path=data_path,
            subsampled_data_path=subsampled_data_path,
        )

    candidates = []
    for search_dir in search_dirs:
        candidates.extend(search_dir.glob(f"{dataset_name}_*.json"))
    if not candidates:
        return None

    candidates = sorted(
        set(candidates),
        key=lambda path: (_get_subsampled_sample_count(path), path.name),
    )
    selected_file = candidates[-1]
    if len(candidates) > 1:
        logger.info(
            f"Found multiple subsampled datasets for '{dataset_name}', using '{selected_file.name}'."
        )
    return selected_file


def resolve_subsampled_dataset_path(
    dataset_file: str,
    data_path: Path,
    subsampled_data_path: Optional[Path] = None,
) -> Path:
    search_dirs = _get_subsampled_search_dirs(
        data_path=data_path,
        subsampled_data_path=subsampled_data_path,
    )

    dataset_path = Path(dataset_file)
    if dataset_path.is_absolute():
        if not dataset_path.exists():
            raise ValueError(f"Unable to find subsampled dataset file: {dataset_path}")
        return dataset_path

    for search_dir in search_dirs:
        candidate = search_dir / dataset_path
        if candidate.exists():
            return candidate

    searched_dirs = ", ".join(str(path) for path in search_dirs)
    raise ValueError(
        f"Unable to find subsampled dataset file '{dataset_path}' in: {searched_dirs}"
    )


def load_dataset(
    dataset_name: str,
    data_path: Path,
    batch_size: int = 16,
    num_workers: int = 1,
    label_templates: List[str] = ["a photo of a {label}"],
    template_key: str = "label",
    precompute_captions: bool = True,
    subsampled_data_path: Optional[Path] = None,
    subsampled_dataset_file: Optional[str] = None,
    subsampled_val_dataset_file: Optional[str] = None,
):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    subsampled_dataset_path = resolve_subsampled_dataset_file(
        dataset_name=dataset_name,
        data_path=data_path,
        subsampled_data_path=subsampled_data_path,
        subsampled_dataset_file=subsampled_dataset_file,
    )
    subsampled_val_dataset_path = None
    if subsampled_val_dataset_file is not None:
        subsampled_val_dataset_path = resolve_subsampled_dataset_path(
            dataset_file=subsampled_val_dataset_file,
            data_path=data_path,
            subsampled_data_path=subsampled_data_path,
        )

    if subsampled_dataset_path is not None:
        logger.info(f"Using subsampled training data from '{subsampled_dataset_path}'.")
        train_dataset = SubsampledImageTextDataset(
            dataset_file=subsampled_dataset_path,
            transform=transform,
        )
        if subsampled_val_dataset_path is not None:
            logger.info(
                f"Using subsampled validation data from '{subsampled_val_dataset_path}'."
            )
            val_dataset = SubsampledImageTextDataset(
                dataset_file=subsampled_val_dataset_path,
                transform=transform,
            )
        else:
            _, val_dataset = get_datasets(
                dataset=dataset_name,
                transform=transform,
                root_dir=data_path,
            )
            if dataset_name != "coco" and dataset_name != "flickr30":
                val_dataset = ImageTextDataset(
                    dataset=val_dataset,
                    label_templates=label_templates,
                    template_key=template_key,
                    precompute_captions=precompute_captions,
                )
                val_dataset.name = dataset_name
    else:
        train_dataset, val_dataset = get_datasets(
            dataset=dataset_name,
            transform=transform,
            root_dir=data_path,
        )

        if dataset_name != "coco" and dataset_name != "flickr30":
            train_dataset = ImageTextDataset(
                dataset=train_dataset,
                label_templates=label_templates,
                template_key=template_key,
                precompute_captions=precompute_captions,
            )
            val_dataset = ImageTextDataset(
                dataset=val_dataset,
                label_templates=label_templates,
                template_key=template_key,
                precompute_captions=precompute_captions,
            )
            train_dataset.name = dataset_name
            val_dataset.name = dataset_name

    # since we're purely using the train dataset
    # for obtaining the embeddings we don't need to shuffle
    train_dataset = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=ImageTextDataset.collate_fn,
    )
    val_dataset = DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=ImageTextDataset.collate_fn,
    )

    return train_dataset, val_dataset


parser = argparse.ArgumentParser(
    description="Experiments for the Representation Alignment.",
)
parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the config yaml.",
)
parser.add_argument(
    "--wandb_notes",
    type=str,
    help="Notes for the wandb run.",
)
args = parser.parse_args()

if __name__ == "__main__":
    args.config_path = Path(args.config_path)
    if not args.config_path.exists():
        raise ValueError(f"Unable to find config yaml file: {args.config_path}")
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=Loader)
    # merge defaults with overrides (overrides take precedence)
    config = merge_dicts(config.get("defaults", {}), config.get("overrides", {}))

    data_path = Path(config["paths"]["data_path"])
    train_dataset, val_dataset = load_dataset(
        dataset_name=config["features"]["dataset"],
        data_path=data_path,
        batch_size=config["features"]["batch_size"],
        num_workers=config["features"]["num_workers"],
        label_templates=config["features"]["label_templates"],
        template_key=config["features"]["template_key"],
        precompute_captions=config["features"]["precompute_captions"],
        subsampled_data_path=config["paths"].get("subsampled_data_path"),
        subsampled_dataset_file=config["features"].get("subsampled_dataset_file"),
        subsampled_val_dataset_file=config["features"].get("subsampled_val_dataset_file"),
    )

    # additional unimodal data
    additional_unimodal_data = None
    text_unimodal_data = []
    image_unimodal_data = []
    if config["training"]["unimodal_data"]["use"]:
        for modality in ["text", "image"]:
            if config["training"]["unimodal_data"][modality] is None:
                continue
            for dataset_name in config["training"]["unimodal_data"][modality]:
                orig_dataset_name = dataset_name
                use_val_set = False
                range_from = None
                range_to = None
                if "_val" in dataset_name:
                    dataset_name = dataset_name.replace("_val", "")
                    use_val_set = True
                if "_" in dataset_name:
                    range_from = int(dataset_name.split("_")[1])
                    range_to = int(dataset_name.split("_")[2])
                    dataset_name = dataset_name.split("_")[0]
                try:
                    ds_train, ds_val = get_datasets(
                        dataset=dataset_name,
                        transform=get_default_transforms(),
                        root_dir=data_path,
                    )
                    if use_val_set:
                        ds_train = ds_val
                    if range_from is not None and range_to is not None:
                        ds_train.df = ds_train.df.iloc[range_from:range_to]
                        ds_train.df.reset_index(drop=True, inplace=True)
                    if config["training"]["unimodal_data"].get("samples", None):
                        ds_train.df = ds_train.df.sample(
                            n=config["training"]["unimodal_data"]["samples"]
                        )
                        ds_train.df.reset_index(drop=True, inplace=True)
                    train_loader = DataLoader(
                        ds_train,
                        batch_size=config["features"]["batch_size"],
                        drop_last=False,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=config["features"]["num_workers"],
                    )
                    if modality == "text":
                        text_unimodal_data.append((orig_dataset_name, train_loader))
                    else:
                        image_unimodal_data.append((orig_dataset_name, train_loader))
                    logger.info(
                        f"Successfully loaded unimodal data '{orig_dataset_name}', train size: {len(ds_train)}"
                    )
                except Exception as e:
                    logger.error(f"Error on {dataset_name}: {e}")
        additional_unimodal_data = {}
        additional_unimodal_data["text"] = text_unimodal_data
        additional_unimodal_data["image"] = image_unimodal_data

    # our evaluation datasets
    eval_zero_shot_datasets = []
    eval_retrieval_datasets = []
    for d_name, l_data in [
        ("zero_shot_datasets", eval_zero_shot_datasets),
        ("retrieval_datasets", eval_retrieval_datasets),
    ]:
        for dataset_name in config["evaluation"][d_name]:
            try:
                _, ds_val = get_datasets(
                    dataset=dataset_name,
                    transform=get_default_transforms(),
                    root_dir=data_path,
                )
                l_data.append((dataset_name, ds_val))
                logger.info(f"Successfully loaded '{dataset_name}', test size: {len(ds_val)}")
            except Exception as e:
                print(f"Error on {dataset_name}: {e}")
    
    trainer_kwargs = {
        "config": config,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "eval_zero_shot_datasets": eval_zero_shot_datasets,
        "eval_retrieval_datasets": eval_retrieval_datasets,
        "wandb_notes": args.wandb_notes,
    }
    trainer_kwargs = trainer_kwargs | config["alignment"]
    if "cca" in config["training"].keys() and config["training"]["cca"]:
        trainer = CSATrainer(**trainer_kwargs)
    elif "clip" in config["training"].keys() and config["training"]["clip"]:
        trainer = CLIPEvalTrainer(**trainer_kwargs)
    else:
        trainer = AlignmentTrainer(**trainer_kwargs)
    trainer.fit(additional_unimodal_data=additional_unimodal_data)
    del trainer
