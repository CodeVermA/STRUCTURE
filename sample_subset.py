import os
import json
import random
from pathlib import Path
from collections import defaultdict


def build_record(img_path, class_name, prompt_template):
    return {
        "image_path": img_path,
        "text": prompt_template.format(class_name),
        "label": class_name,
    }


def create_subsampled_dataset(
    train_dir,
    output_dir,
    sample_size=5000,
    prompt_template="A photo of a {}",
    validation_output_stem=None,
    seed=42,
):
    """
    Randomly samples images from an ImageFolder directory using stratified sampling
    and generates synthetic text-image pairs.
    """
    rng = random.Random(seed)
    
    # 1. Collect all images grouped by class
    class_to_images = defaultdict(list)
    
    for class_name in sorted(os.listdir(train_dir)):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            clean_class_name = class_name.replace("_", " ").replace("-", " ")
            for img_file in sorted(os.listdir(class_dir)):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    class_to_images[clean_class_name].append(img_path)
                    
    # 2. Calculate stratified sample size per class
    num_classes = len(class_to_images)
    samples_per_class = sample_size // num_classes
    remainder = sample_size % num_classes
    
    sampled_data = []
    remaining_data = []
    
    # 3. Sample images
    for idx, (class_name, images) in enumerate(class_to_images.items()):
        # Handle cases where a class has fewer images than the target quota
        quota = samples_per_class + (1 if idx < remainder else 0)
        
        if len(images) >= quota:
            sampled_imgs = rng.sample(images, quota)
        else:
            # If a class is tiny, take all and warn (you should document this if it happens!)
            print(f"WARNING: Class {class_name} only has {len(images)} images (quota: {quota})")
            sampled_imgs = list(images)

        sampled_img_set = set(sampled_imgs)
        remaining_imgs = [img_path for img_path in images if img_path not in sampled_img_set]
            
        for img_path in sampled_imgs:
            sampled_data.append(build_record(img_path, class_name, prompt_template))

        for img_path in remaining_imgs:
            remaining_data.append(build_record(img_path, class_name, prompt_template))
            
    # 4. Shuffle the final dataset to ensure mixed batches during training
    rng.shuffle(sampled_data)
    rng.shuffle(remaining_data)
        
    dataset_name = Path(train_dir).parent.name
    output_json = Path(output_dir) / f"{dataset_name}_{len(sampled_data)}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # 5. Save to JSON
    with open(output_json, 'w') as f:
        json.dump(sampled_data, f, indent=4)
        
    print(f"Successfully saved {len(sampled_data)} pairs to {output_json}\n")

    if validation_output_stem is not None:
        validation_output_json = (
            Path(output_dir) / f"{validation_output_stem}_{len(remaining_data)}.json"
        )
        with open(validation_output_json, "w") as f:
            json.dump(remaining_data, f, indent=4)

        print(
            "Successfully saved "
            f"{len(remaining_data)} validation pairs to {validation_output_json}\n"
        )

# --- Execution ---
if __name__ == "__main__":
    SAMPLE_SIZE = 3600
    
    # 1. Aircraft (Using trainval):
    create_subsampled_dataset(
        train_dir="/home/s2412780/MLP/STRUCTURE/data/processed/aircraft/trainval",
        output_dir="/home/s2412780/MLP/STRUCTURE/data/subsampled",
        prompt_template="A photo of a {}, a type of aircraft.",
        sample_size=SAMPLE_SIZE,
        validation_output_stem="aircraft_val",
    )

    # 2. Pets (Using train):
    create_subsampled_dataset(
        train_dir="/home/s2412780/MLP/STRUCTURE/data/processed/pets/train",
        output_dir="/home/s2412780/MLP/STRUCTURE/data/subsampled",
        prompt_template="A photo of a {}, a type of pet.",
        sample_size=SAMPLE_SIZE
    )

    #3. UCF101 (Using train)
    create_subsampled_dataset(
        train_dir="/home/s2412780/MLP/STRUCTURE/data/processed/ucf101/train",
        output_dir="/home/s2412780/MLP/STRUCTURE/data/subsampled",
        prompt_template="A photo of a person doing {}.",
        sample_size=SAMPLE_SIZE
    )
