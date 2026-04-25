import os
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def get_y_true_y_pred(model, dataset):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        preds_classes = np.argmax(preds, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds_classes)

    return np.array(y_true), np.array(y_pred)


##########
def make_confusion_matrix(y_true, y_pred, classes=None,
                          figsize=(10, 10), text_size=15,
                          norm=False, savefig=False, filename="confusion_matrix.png"):
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    labels = classes if classes is not None else np.arange(cm.shape[0])

    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels
    )

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    threshold = (cm.max() + cm.min()) / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            text = f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)"
        else:
            text = f"{cm[i, j]}"
        ax.text(
            j, i, text,
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=text_size
        )

    fig.tight_layout()

    if savefig:
        fig.savefig(filename, dpi=150, bbox_inches="tight")

    return fig, ax

def plot_random_images_per_class(
    csv_path,
    images_dir=None,
    samples_per_class: int = 1,
    n_cols: int = 3,
    figsize_per_image: tuple = (4, 4),
    seed: int = None,
    base_dir=None,
    image_col: str = "image",
    label_col: str = "classes",
    title: str = "Random Samples Per Class",
    class_order=None,
):
    """
    Plot random image samples for each class.
    """

    # Resolve paths
    base_dir = Path(base_dir) if base_dir else Path.cwd()
    csv_path = Path(csv_path)
    csv_path = csv_path if csv_path.is_absolute() else base_dir / csv_path

    if images_dir is None:
        images_dir = csv_path.parent / "image"
    else:
        images_dir = Path(images_dir)
        images_dir = images_dir if images_dir.is_absolute() else base_dir / images_dir

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Load data
    df = pd.read_csv(csv_path)

    if image_col not in df.columns:
        raise ValueError(f"Image column '{image_col}' not found in CSV")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV")

    # Drop duplicate image rows
    df = df.drop_duplicates(subset=[image_col]).copy()

    # Keep only classes that have enough rows if needed
    # For your case samples_per_class=1, so this is safe for all classes
    sampled_df = (
        df.groupby(label_col, group_keys=False)
          .sample(n=samples_per_class, random_state=seed)
          .reset_index(drop=True)
    )

    # Stable class order
    if class_order is not None:
        sampled_df[label_col] = pd.Categorical(
            sampled_df[label_col],
            categories=class_order,
            ordered=True
        )
        sampled_df = sampled_df.sort_values(label_col).reset_index(drop=True)
    else:
        sampled_df = sampled_df.sort_values(by=label_col).reset_index(drop=True)

    # Build subplot grid
    n = len(sampled_df)
    n_rows = math.ceil(n / n_cols)
    fig_w = figsize_per_image[0] * n_cols
    fig_h = figsize_per_image[1] * n_rows + 0.8

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, (_, row) in enumerate(sampled_df.iterrows()):
        ax = axes[idx]

        img_rel = Path(str(row[image_col]).lstrip("/"))
        img_path = images_dir / img_rel.name
        if not img_path.exists():
            img_path = images_dir / img_rel

        if img_path.exists():
            img = mpimg.imread(img_path)
            ax.imshow(img)
        else:
            ax.text(
                0.5, 0.5, f"Image not found\n{img_rel.name}",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="red",
            )

        ax.set_title(str(row[label_col]), fontsize=10, pad=4)
        ax.axis("off")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    return None
# ──────────────────────────────────────────────────────────────────────────────
# Directory utilities
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Image grid visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_random_images_with_labels(
    csv_path,
    images_dir=None,
    num_samples: int = 9,
    n_cols: int = 3,
    figsize_per_image: tuple = (4, 4),
    seed: int = None,
    base_dir=None,
    image_col: str = None,
    label_col: str = None,
    title: str = "Random Dataset Samples",
):
    """
    Plots random images with their labels from a Peltarion-style dataset.

    The dataset is expected to have:
      - A CSV file (data.csv) with at least one image-path column and one label column.
      - An image directory (default: <csv_parent>/image/) containing the image files.

    Column auto-detection (used when image_col / label_col are not provided):
      - Image column: first column whose name contains 'image', 'path', 'img', or 'file'.
        Falls back to the first column.
      - Label column: first column whose name contains 'label', 'class', 'target',
        'damage', or 'category'. Falls back to the last column.

    Args:
        csv_path  (str | Path): Path to data.csv (absolute or relative to base_dir).
        images_dir (str | Path, optional): Directory that contains the image files.
                                           Defaults to <csv_parent>/image/.
        num_samples (int):   Number of images to sample and display. Default 9.
        n_cols      (int):   Number of columns in the subplot grid. Default 3.
        figsize_per_image (tuple): (width, height) in inches per subplot. Default (4, 4).
        seed        (int, optional): Random seed for reproducibility.
        base_dir    (str | Path, optional): Root used to resolve relative paths.
                                            Defaults to the current working directory.
        image_col   (str, optional): Explicit name of the image-path column in the CSV.
        label_col   (str, optional): Explicit name of the label column in the CSV.
        title       (str): Figure super-title. Default "Random Dataset Samples".

    Returns:
        fig (matplotlib.figure.Figure): The generated figure.

    Example — Peltarion car-damage dataset
    ----------------------------------------
    Project layout::

        tests/Projects/Project03:Peltarion/dataset/
        └── datasets/hamzamanssor/car-damage-assessment/
            └── versions/1/
                ├── data.csv
                └── image/

    Usage::

        plot_random_images_with_labels(
            csv_path="datasets/hamzamanssor/car-damage-assessment/versions/1/data.csv",
            base_dir="tests/Projects/Project03:Peltarion/dataset",
            num_samples=9,
            seed=42,
        )
    """
    # ── Resolve paths ──────────────────────────────────────────────────────────
    base_dir = Path(base_dir) if base_dir else Path.cwd()
    csv_path = Path(csv_path)
    csv_path = csv_path if csv_path.is_absolute() else base_dir / csv_path

    if images_dir is None:
        images_dir = csv_path.parent / "image"
    else:
        images_dir = Path(images_dir)
        images_dir = images_dir if images_dir.is_absolute() else base_dir / images_dir

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # ── Load CSV ───────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    cols_lower = [c.lower() for c in df.columns]

    image_col = "image"      # or auto-detect if you coded that
    label_col = "classes"

    # drop duplicate image rows
    unique_df = df.drop_duplicates(subset=[image_col])

    # ── Sample rows ───────────────────────────────────────────────────────────
    n = min(num_samples, len(unique_df))
    sample_df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    # ── Build subplot grid ────────────────────────────────────────────────────
    n_rows = (n + n_cols - 1) // n_cols
    fig_w = figsize_per_image[0] * n_cols
    fig_h = figsize_per_image[1] * n_rows + 0.6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        ax = axes[idx]

        # Resolve image path: strip any leading 'image/' prefix then look inside images_dir
        img_rel = Path(str(row[image_col]).lstrip("/"))
        img_path = images_dir / img_rel.name          # bare filename inside images_dir
        if not img_path.exists():
            img_path = images_dir / img_rel            # keep sub-path if present

        if img_path.exists():
            img = mpimg.imread(img_path)
            ax.imshow(img)
        else:
            ax.text(
                0.5, 0.5, f"Image not found\n{img_rel.name}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="red",
            )

        ax.set_title(str(row[label_col]), fontsize=10, pad=4)
        ax.axis("off")

    for ax in axes[n:]:          # hide empty subplots
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return None

# ──────────────────────────────────────────────────────────────────────────────
# dataset structure
# ──────────────────────────────────────────────────────────────────────────────

import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_car_damage_dataset(
    csv_path,
    images_root,
    output_root="dataset1",
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    copy_files=True,
):
    if not abs(train_size + val_size + test_size - 1.0) < 1e-8:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    df = pd.read_csv(csv_path)

    if "image" not in df.columns or "classes" not in df.columns:
        raise ValueError("CSV must have 'image' and 'classes' columns")

    df = df[["image", "classes"]].dropna().copy()
    df["classes"] = df["classes"].astype(str)

    df["full_path"] = df["image"].apply(
        lambda x: str(Path(images_root) / x)
    )

    df = df[df["full_path"].apply(os.path.exists)].copy()

    if df.empty:
        raise RuntimeError("No valid image files found. Check paths.")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_size),
        stratify=df["classes"],
        random_state=random_state,
    )

    val_ratio_adjusted = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_df["classes"],
        random_state=random_state,
    )

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            cls = row["classes"]
            src = Path(row["full_path"])
            dst_dir = output_root / split_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            dst = dst_dir / src.name

            if copy_files:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

    for split_name, split_df in splits.items():
        print(f"\n{split_name.upper()} SET")
        print(split_df["classes"].value_counts())

    return splits
# ======================
import matplotlib.pyplot as plt
import pandas as pd


def plot_split_label_distribution(train_df: pd.DataFrame,
                                  val_df: pd.DataFrame,
                                  test_df: pd.DataFrame | None = None,
                                  label_col: str = "classes"):
    """Visualize class distribution for train / val (/ test) splits."""
    splits = {"Train": train_df, "Val": val_df}
    if test_df is not None:
        splits["Test"] = test_df

    counts = {
        name: df[label_col].value_counts().sort_index()
        for name, df in splits.items()
    }

    all_labels = sorted(set().union(*[c.index for c in counts.values()]))

    counts_df = pd.DataFrame(
        {name: c.reindex(all_labels, fill_value=0) for name, c in counts.items()}
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # 1) Bar chart: counts per split
    counts_df.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Class Frequency per Split")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Image count")
    axes[0].tick_params(axis="x", rotation=45)

    # Optional: also set alignment directly on labels
    for label in axes[0].get_xticklabels():
        label.set_horizontalalignment("right")

    # 2) Stacked bar of percentages per class
    perc_df = counts_df.div(counts_df.sum(axis=1), axis=0) * 100
    bottom = None
    for split_name in perc_df.columns:
        axes[1].barh(
            perc_df.index,
            perc_df[split_name],
            left=bottom,
            label=f"{split_name} %",
        )
        bottom = perc_df[split_name] if bottom is None else bottom + perc_df[split_name]

    axes[1].set_title("Split Percentage per Class")
    axes[1].set_xlabel("Percentage (%)")
    axes[1].legend()
    axes[1].axvline(80, color="gray", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.show()