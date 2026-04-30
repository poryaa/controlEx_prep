# utils.py

from pathlib import Path
import os
import shutil
from collections import Counter
import random

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# =========================================================================
# --------------------------------------------------------------------
# 1. Download Kaggle dataset once into a stable folder
# --------------------------------------------------------------------


def download_original_dataset(
    dataset_name: str = "prajwalbhamere/car-damage-severity-dataset",
    target_dir: str = "original_dataset",
):
    """
    Download a Kaggle dataset via kagglehub into `target_dir` if not already present.

    Args:
        dataset_name (str): Kaggle dataset id.
        target_dir (str): Local folder where the full dataset will live.

    Returns:
        Path: path to the final dataset folder (target_dir).
    """
    target_path = Path(target_dir)

    # If target_dir already exists and is non-empty, skip download
    if target_path.exists() and any(target_path.iterdir()):
        print(f"[INFO] Dataset already exists, skipping download: {target_path.resolve()}")
        return target_path

    # Configure KaggleHub cache (important in Colab)
    cache_root = Path.cwd()
    os.environ["KAGGLEHUB_CACHE"] = str(cache_root)
    os.environ["DISABLE_COLAB_CACHE"] = "true"

    # Download latest version into KaggleHub cache
    cache_path = Path(kagglehub.dataset_download(dataset_name))
    print("[INFO] Downloaded dataset cache path:", cache_path)

    # Copy from KaggleHub cache into target_dir (clean target_dir if needed)
    if target_path.exists():
        shutil.rmtree(target_path)

    shutil.copytree(cache_path, target_path)
    print("[INFO] Final dataset folder:", target_path.resolve())

    # Optional: clear KaggleHub cache directory to save space
    datasets_cache_dir = cache_root / "datasets"
    if datasets_cache_dir.exists():
        shutil.rmtree(datasets_cache_dir)
        print("[INFO] Removed KaggleHub datasets cache at:", datasets_cache_dir.resolve())

    return target_path

# =========================================================================
# --------------------------------------------------------------------
# 2. Walk a directory tree and print structure
# --------------------------------------------------------------------
def walk_through_dir(dir_path):
    """
    Walks through dir_path and prints number of subdirectories and files in each.

    Args:
        dir_path (str | Path): root directory

    Returns:
        None
    """
    dir_path = Path(dir_path)
    for root, dirs, files in os.walk(dir_path):
        root_path = Path(root)
        num_dirs = len(dirs)
        num_files = len(files)
        print(f"There are {num_dirs} directories and {num_files} images/files in '{root_path.relative_to(dir_path)}'.")

# =========================================================================
# --------------------------------------------------------------------
# 3. Plot one random image per class (folder-based labels)
# --------------------------------------------------------------------
def plot_one_random_image_per_class(
    train_dir,
    figsize_per_image=(4, 4),
    seed=None,
):
    """
    Plots one random image from each class subfolder in train_dir.

    Expected structure:
        train_dir/
            01-minor/
            02-moderate/
            03-severe/

    Args:
        train_dir (str | Path): path to training directory
        figsize_per_image (tuple): (width, height) per subplot in inches
        seed (int | None): random seed for reproducibility

    Returns:
        None
    """
    train_dir = Path(train_dir)
    if not train_dir.exists():
        print(f"[ERROR] Training directory not found: {train_dir.resolve()}")
        return

    class_dirs = [d for d in sorted(train_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        print(f"[ERROR] No class subfolders found in: {train_dir.resolve()}")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if seed is not None:
        random.seed(seed)

    samples = []
    for class_dir in class_dirs:
        image_paths = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]
        if not image_paths:
            print(f"[WARNING] No images found in class: {class_dir.name}")
            continue

        img_path = random.choice(image_paths)
        clean_label = class_dir.name.split("-", 1)[-1]
        samples.append((img_path, clean_label))

    if not samples:
        print("[ERROR] No images found to display.")
        return

    n_cols = len(samples)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(figsize_per_image[0] * n_cols, figsize_per_image[1])
    )

    if n_cols == 1:
        axes = [axes]

    for ax, (img_path, label) in zip(axes, samples):
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(label, fontsize=10, pad=6)
        except Exception as e:
            ax.set_visible(False)
            print(f"[WARNING] Could not open image: {img_path.name} -> {e}")

    plt.tight_layout()
    plt.show()

# =========================================================================
# --------------------------------------------------------------------
# 4. Plot train/val label distributions (bar + pies)
# --------------------------------------------------------------------
def plot_split_label_distributions(
    train_dir,
    val_dir,
    figsize=(16, 5),
):
    """
    Plots label frequencies for train and validation splits using
    one grouped bar chart and two pie charts.

    Args:
        train_dir (str | Path): path to training directory
        val_dir   (str | Path): path to validation directory
        figsize   (tuple): matplotlib figure size

    Returns:
        None
    """
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def get_clean_label_counts(data_dir):
        if not data_dir.exists():
            print(f"[ERROR] Directory not found: {data_dir.resolve()}")
            return {}

        counts = {}
        for class_dir in sorted(data_dir.iterdir()):
            if class_dir.is_dir():
                clean_label = class_dir.name.split("-", 1)[-1]
                count = sum(
                    1 for p in class_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in image_extensions
                )
                counts[clean_label] = count
        return counts

    train_counts = get_clean_label_counts(train_dir)
    val_counts = get_clean_label_counts(val_dir)

    if not train_counts or not val_counts:
        print("[ERROR] Could not compute label counts.")
        return

    labels = sorted(set(train_counts.keys()) | set(val_counts.keys()))
    train_values = [train_counts.get(label, 0) for label in labels]
    val_values = [val_counts.get(label, 0) for label in labels]

    colors = ["#4C78A8", "#F58518", "#54A24B"][:len(labels)]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Grouped bar chart
    x = range(len(labels))
    width = 0.35

    bars_train = axes[0].bar(
        [i - width / 2 for i in x],
        train_values,
        width=width,
        label="Train",
        color="#4C78A8"
    )
    bars_val = axes[0].bar(
        [i + width / 2 for i in x],
        val_values,
        width=width,
        label="Validation",
        color="#F58518"
    )

    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Label Frequency by Split")
    axes[0].set_ylabel("Number of Images")
    axes[0].legend()

    axes[0].bar_label(bars_train, fmt="%d", padding=3, fontsize=9)
    axes[0].bar_label(bars_val, fmt="%d", padding=3, fontsize=9)

    # Train pie chart
    axes[1].pie(
        train_values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors
    )
    axes[1].set_title("Train Split")

    # Validation pie chart
    axes[2].pie(
        val_values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors
    )
    axes[2].set_title("Validation Split")

    plt.tight_layout()
    plt.show()

# =========================================================================
# --------------------------------------------------------------------
# 5. Brief dataset inspection: sizes, formats, pixel stats
# --------------------------------------------------------------------
def inspect_image_dataset_brief(root_dir):
    """
    Quickly inspects an image dataset: size, modes, formats, and basic pixel stats.

    Args:
        root_dir (str | Path): root directory containing images (possibly in subfolders)

    Returns:
        None — prints a short summary
    """
    root_dir = Path(root_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    widths, heights = [], []
    modes, formats = [], []
    dtypes, mins, maxs = [], [], []
    corrupted = []

    image_paths = [
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    for p in image_paths:
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                w, h = img.size
                widths.append(w)
                heights.append(h)
                modes.append(img.mode)
                formats.append(img.format)

                arr = np.array(img)
                dtypes.append(arr.dtype)
                mins.append(arr.min())
                maxs.append(arr.max())
        except Exception:
            corrupted.append(str(p))

    if not widths:
        print("[ERROR] No readable images found.")
        return

    print(f"Total images checked: {len(widths)}")
    print(f"Corrupted/unreadable files: {len(corrupted)}")
    print(f"Width  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}")
    print(f"Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}")
    print(f"Color modes: {Counter(modes)}")
    print(f"Image formats: {Counter(formats)}")
    print(f"dtype:  {Counter(dtypes)}")
    print(f"min:    {min(mins)}")
    print(f"max:    {max(maxs)}")
    print("Note: shape varies per image; see width/height stats above.")



# =========================================================================
# --------------------------------------------------------------------
# 6. plot confusion matrix
# --------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix_for_model(
    model_or_path,
    dataset,
    class_names,
    normalize=True,
    title_prefix="Confusion matrix",
    cmap="Blues",
    figsize=(6, 5),
    save_path=None,
):
    """
    Compute and plot confusion matrix for a Keras model on a tf.data.Dataset.

    Args:
        model_or_path: tf.keras.Model or path to .keras model file.
        dataset: tf.data.Dataset yielding (images, labels).
        class_names: list of class names in label-index order.
        normalize: if True, normalize by true labels (rows).
        title_prefix: title prefix for the plot.
        cmap: matplotlib colormap.
        figsize: figure size.
        save_path: optional path to save the plot.

    Returns:
        cm_raw: raw confusion matrix counts
        y_true: true label indices
        y_pred: predicted label indices
    """
    # Load model if path was passed
    if isinstance(model_or_path, (str, Path)):
        model = tf.keras.models.load_model(model_or_path)
        model_name = Path(model_or_path).stem
    else:
        model = model_or_path
        model_name = getattr(model, "name", "model")

    y_true = []
    y_pred = []

    for batch_images, batch_labels in dataset:
        preds = model.predict(batch_images, verbose=0)
        batch_pred = np.argmax(preds, axis=1)

        labels_np = batch_labels.numpy()
        if labels_np.ndim > 1 and labels_np.shape[-1] > 1:
            labels_np = np.argmax(labels_np, axis=1)  # handle one-hot labels

        y_true.extend(labels_np.reshape(-1))
        y_pred.extend(batch_pred.reshape(-1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm_raw = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    normalize_mode = "true" if normalize else None
    cm_display = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        normalize=normalize_mode
    )

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_display,
        display_labels=class_names
    )
    disp.plot(
        cmap=cmap,
        ax=ax,
        values_format=".2f" if normalize else "d",
        colorbar=True
    )

    title_suffix = "normalized" if normalize else "counts"
    ax.set_title(f"{title_prefix} - {model_name} ({title_suffix})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

    return cm_raw, y_true, y_pred

# =========================================================================
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

def compare_models_on_image_path(
    image_path,
    class_names,
    keras_model_paths=None,
    yolo_model_path=None,
    image_size=(224, 224),
    figsize=(8, 8),
):
    """
    Compare predictions of multiple models on a single image.

    Args:
        image_path: Path to the image file.
        class_names: List of class names (index -> name) used by Keras models.
        keras_model_paths: List of .keras model paths.
        yolo_model_path: Optional YOLO .pt model path.
        image_size: Size to resize the image for Keras models.
        figsize: Matplotlib figure size.
    """
    if keras_model_paths is None:
        keras_model_paths = [
            "model_1.keras",
            "model_2.keras",
            "model_3.keras",
            "model_4.keras",
            "model_5.keras",
            "model_6.keras"]

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Try to infer true label from parent directory name, otherwise "unknown"
    parent_name = img_path.parent.name
    true_label = parent_name if parent_name in class_names else "unknown"

    # Load and show image
    image_original = np.array(
        Image.open(img_path).convert("RGB").resize(image_size)
    )
    plt.figure(figsize=figsize)
    plt.imshow(image_original)
    plt.title(f"True label: {true_label}", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"Image: {img_path.name}")
    print(f"Inferred label (from folder): {true_label}")
    print("=" * 60)

    # Prepare batch for Keras models (0–255, Rescaling inside models)
    image_for_keras = image_original.astype(np.float32)
    image_batch = np.expand_dims(image_for_keras, axis=0)

    # --- Keras models ---
    for model_path in keras_model_paths:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"⚠️ {model_path} not found, skipping...")
            continue

        model = tf.keras.models.load_model(model_path)
        preds = model.predict(image_batch, verbose=0)
        pred_idx = int(np.argmax(preds[0]))
        pred_conf = float(np.max(preds[0]))
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"idx_{pred_idx}"
        is_correct = "✓" if pred_name == true_label else "✗"

        print(f"\n{model_path.stem}:")
        print(f"  Predicted : {pred_name} {is_correct}")
        print(f"  Confidence: {pred_conf:.4f} ({pred_conf*100:.2f}%)")

    print("\n" + "=" * 60)
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix_for_model(
    model_name,
    dataset,
    class_names,
    models_dir="saved_models",
    title_prefix="Validation confusion matrix",
    cmap="Blues",
    figsize=(6, 5),
):
    """
    Load saved_models/{model_name}.keras, compute confusion matrix on a dataset,
    and plot it with raw counts (no normalization).

    Args:
        model_name: e.g. "model_3"
        dataset: tf.data.Dataset yielding (images, labels)
        class_names: list of class names in label-index order
        models_dir: folder containing saved .keras models
        title_prefix: plot title prefix
        cmap: matplotlib colormap
        figsize: figure size

    Returns:
        cm_raw, y_true, y_pred
    """
    model_path = Path(models_dir) / f"{model_name}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    y_true, y_pred = [], []

    for batch_images, batch_labels in dataset:
        preds = model.predict(batch_images, verbose=0)
        pred_idx = np.argmax(preds, axis=1)

        labels_np = batch_labels.numpy()
        if labels_np.ndim > 1 and labels_np.shape[-1] > 1:
            labels_np = np.argmax(labels_np, axis=1)

        y_true.extend(labels_np.reshape(-1))
        y_pred.extend(pred_idx.reshape(-1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm_raw = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_raw,
        display_labels=class_names,
    )
    disp.plot(
        cmap=cmap,
        ax=ax,
        values_format="d",
        colorbar=True,
    )

    ax.set_title(f"{title_prefix} - {model_name} (counts)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return cm_raw, y_true, y_pred

# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay)

def get_true_and_pred_labels(model, dataset):
    y_true = []
    y_prob = []

    for batch_x, batch_y in dataset:
        batch_pred = model.predict(batch_x, verbose=0)
        y_prob.append(batch_pred)

        if len(batch_y.shape) > 1 and batch_y.shape[-1] > 1:
            y_true.append(np.argmax(batch_y.numpy(), axis=1))
        else:
            y_true.append(batch_y.numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    return y_true, y_pred, y_prob


def evaluate_model_on_dataset(model, dataset, class_names, model_name="model"):
    y_true, y_pred, y_prob = get_true_and_pred_labels(model, dataset)

    acc = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    summary_df = pd.DataFrame([{
        "model": model_name,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    }])

    per_class_df = pd.DataFrame({
        "class_name": class_names,
        "precision": precision_per_class,
        "recall": recall_per_class,
        "f1": f1_per_class,
        "support": support_per_class
    })

    cm = confusion_matrix(y_true, y_pred)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    return {
        "summary_df": summary_df,
        "per_class_df": per_class_df,
        "confusion_matrix": cm,
        "classification_report": report_text,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_confusion_matrix_from_preds(y_true, y_pred, class_names, title="Confusion Matrix", figsize=(6, 5), cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    return cm