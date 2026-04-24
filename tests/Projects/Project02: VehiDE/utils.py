import os

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images/files in '{dirpath}'.")

######################################################

import json
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image


def plot_random_images_with_labels(
    json_path,
    images_dir,
    num_samples=9,
    n_cols=3,
    figsize_per_image=(4, 4),
    seed=None,
):
    """
    Plots random images from the train folder with their damage labels as captions.

    Args:
        json_path   (str | Path): path to the VehiDE annotation JSON file
        images_dir  (str | Path): path to the folder containing the images
        num_samples (int):        number of images to sample and display (default: 9)
        n_cols      (int):        number of columns in the grid (default: 3)
        figsize_per_image (tuple): (width, height) in inches per subplot cell
        seed        (int | None): random seed for reproducibility

    Returns:
        None — displays a matplotlib figure
    """
    json_path = Path(json_path)
    images_dir = Path(images_dir)

    if not json_path.exists():
        print(f"[ERROR] JSON file not found: {json_path.resolve()}")
        return

    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir.resolve()}")
        return

    # Load annotations
    with open(json_path, "r") as f:
        annos = json.load(f)

    # Build mapping: filename -> list of unique damage classes
    image_to_labels = {}
    for img_key, meta in annos.items():
        filename = meta.get("name") or img_key
        regions = meta.get("regions", [])
        unique_classes = list(dict.fromkeys(
            r.get("class") for r in regions if r.get("class")
        ))
        if unique_classes:
            image_to_labels[filename] = unique_classes

    print(f"Annotated images found: {len(image_to_labels)}")

    if seed is not None:
        random.seed(seed)

    sample_filenames = random.sample(
        list(image_to_labels.keys()),
        k=min(num_samples, len(image_to_labels))
    )

    n_rows = (len(sample_filenames) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_image[0] * n_cols, figsize_per_image[1] * n_rows)
    )
    axes = axes.flatten()

    for i, fname in enumerate(sample_filenames):
        img_path = images_dir / fname
        if not img_path.exists():
            axes[i].set_visible(False)
            continue

        img = Image.open(img_path).convert("RGB")
        labels_str = ", ".join(image_to_labels[fname])

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(
            f"{fname}\n{labels_str}",
            fontsize=8,
            pad=4
        )

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
#######################################################################
import json
from pathlib import Path

def get_unique_labels(json_path):
    """
    Counts all unique damage labels from a VehiDE annotation JSON file.

    Args:
        json_path (str | Path): path to the annotation JSON file

    Returns:
        dict: {label: count} sorted by count descending
    """
    with open(Path(json_path), "r") as f:
        annos = json.load(f)

    label_counts = {}

    for meta in annos.values():
        for region in meta.get("regions", []):
            cls = region.get("class")
            if cls:
                label_counts[cls] = label_counts.get(cls, 0) + 1

    # Sort by frequency
    label_counts = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True))

    print(f"Total unique labels: {len(label_counts)}")
    print(f"{'Label':<20} {'Count':>8}")
    print("-" * 30)
    for label, count in label_counts.items():
        print(f"{label:<20} {count:>8}")

    return label_counts
####################################################
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_label_distribution(train_json, val_json):
    """
    Plots label distributions for train and validation sets.

    Args:
        train_json (str | Path): path to train annotation JSON
        val_json   (str | Path): path to val annotation JSON
    """

    def count_labels(json_path):
        with open(Path(json_path), "r") as f:
            annos = json.load(f)
        counts = {}
        for meta in annos.values():
            for region in meta.get("regions", []):
                cls = region.get("class")
                if cls:
                    counts[cls] = counts.get(cls, 0) + 1
        return counts

    train_counts = count_labels(train_json)
    val_counts   = count_labels(val_json)

    # Align labels
    all_labels = sorted(
        set(train_counts) | set(val_counts),
        key=lambda l: train_counts.get(l, 0),
        reverse=True
    )

    train_vals = [train_counts.get(l, 0) for l in all_labels]
    val_vals   = [val_counts.get(l, 0)   for l in all_labels]

    x = np.arange(len(all_labels))
    bar_width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("VehiDE Dataset — Label Distribution", fontsize=14, fontweight="bold")

    # --- Plot 1: Grouped bar chart train vs val ---
    ax = axes[0]
    ax.bar(x - bar_width/2, train_vals, bar_width, label="Train", color="#4C72B0")
    ax.bar(x + bar_width/2, val_vals,   bar_width, label="Val",   color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=30, ha="right")
    ax.set_ylabel("Region count")
    ax.set_title("Class Frequency: Train vs Val")
    ax.legend()
    for i, (tv, vv) in enumerate(zip(train_vals, val_vals)):
        ax.text(i - bar_width/2, tv + 50, str(tv), ha="center", fontsize=7)
        ax.text(i + bar_width/2, vv + 50, str(vv), ha="center", fontsize=7)

    # --- Plot 2: Train/Val split ratio per class ---
    ax = axes[1]
    total_vals = [t + v for t, v in zip(train_vals, val_vals)]
    train_ratio = [t / total * 100 if total else 0 for t, total in zip(train_vals, total_vals)]
    val_ratio   = [v / total * 100 if total else 0 for v, total in zip(val_vals, total_vals)]
    ax.barh(all_labels, train_ratio, color="#4C72B0", label="Train %")
    ax.barh(all_labels, val_ratio,   left=train_ratio, color="#DD8452", label="Val %")
    ax.axvline(80, color="gray", linestyle="--", linewidth=1, label="80% line")
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Train / Val Split per Class")
    ax.legend()

    # --- Plot 3: Pie chart of total class proportions ---
    ax = axes[2]
    total_counts = [t + v for t, v in zip(train_vals, val_vals)]
    ax.pie(
        total_counts,
        labels=all_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=plt.cm.Set2.colors
    )
    ax.set_title("Overall Class Proportions (Train + Val)")

    plt.tight_layout()
    plt.show()