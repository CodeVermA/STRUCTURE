import wandb
import pandas as pd

api = wandb.Api()

runs = api.runs("vasuverma-the-university-of-edinburgh/representation-alignment")

rows = []

def get_method(cfg):
    name = cfg.get("training", {}).get("value", {}).get("alignment_layer_name", "")
    if "Linear" in name:
        return "Linear"
    if "MLP" in name:
        return "MLP"
    if "CSA" in name:
        return "CSA"
    return name or "Unknown"

def get_layer_type(cfg):
    ls = cfg.get("layer_selection", {}).get("value", {})
    if ls.get("last_only", False):
        return "Last"
    if ls.get("best_only", False):
        return "Similar"
    return "Unknown"

def has_rs(cfg):
    lam = (
        cfg.get("training", {})
           .get("value", {})
           .get("clip_loss", {})
           .get("structure_lambda", 0)
    )
    return lam and lam > 0

for run in runs:
    cfg = run.config
    summary = run.summary

    method = get_method(cfg)
    layer = get_layer_type(cfg)
    use_rs = has_rs(cfg)

    row_name = f"{method} + {layer}"
    if use_rs:
        row_name += " + RS"

    row = {
        "Method": row_name,
        "STL10": summary.get("img_10_txt_10/stl10/top1_acc_micro"),
        "CIFAR10": summary.get("img_10_txt_10/cifar10/top1_acc_micro"),
        "Caltech101": summary.get("img_10_txt_10/caltech101/top1_acc_micro"),
        "Food101": summary.get("img_10_txt_10/food101/top1_acc_micro"),
        "CIFAR100": summary.get("img_10_txt_10/cifar100/top1_acc_micro"),
        "ImageNet": summary.get("img_10_txt_10/imagenet/top1_acc_micro"),
        "Pets": summary.get("img_10_txt_10/pets/top1_acc_micro"),
        "Flickr30 I2T": summary.get("img_10_txt_10/flickr30/I2T-R@1"),
        "Flickr30 T2I": summary.get("img_10_txt_10/flickr30/T2I-R@1"),
        "COCO I2T": summary.get("img_10_txt_10/coco/I2T-R@1"),
        "COCO T2I": summary.get("img_10_txt_10/coco/T2I-R@1"),
    }

    rows.append(row)

df = pd.DataFrame(rows)

# If you have repeated runs for the same setup, average them
df = df.groupby("Method", as_index=False).mean(numeric_only=True)

order = [
    "Linear + Last",
    "Linear + Similar",
    "Linear + Similar + RS",
    "MLP + Last",
    "MLP + Similar",
    "MLP + Similar + RS",
    "CSA + Last",
    "CSA + Similar",
    "CSA + Similar + RS",
]

df["Method"] = pd.Categorical(df["Method"], categories=order, ordered=True)
df = df.sort_values("Method")

for col in df.columns[1:]:
    df[col] = (df[col] * 100).round(1)  # convert from 0.1495 to 14.9

print(df.to_string(index=False))
df.to_csv("paper_style_table.csv", index=False)