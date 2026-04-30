import matplotlib.pyplot as plt
import pandas as pd


def plot_split_label_distribution(y_train, y_val, y_test, title="BookingCode distribution in train/val/test splits"):
    """
    Plot stacked horizontal bars of label counts in train/val/test splits.

    Parameters
    ----------
    y_train, y_val, y_test : array-like or pandas Series
        Label vectors for each split.
    title : str
        Plot title.
    """
    # Counts per split
    train_counts = pd.Series(y_train).value_counts()
    val_counts   = pd.Series(y_val).value_counts()
    test_counts  = pd.Series(y_test).value_counts()

    # Combine into a single DataFrame, align on all labels
    label_dist = pd.DataFrame({
        "Train": train_counts,
        "Val":   val_counts,
        "Test":  test_counts,
    }).fillna(0).astype(int)

    # Sort labels by total frequency (more readable)
    label_dist = (
        label_dist.assign(Total=label_dist.sum(axis=1))
                  .sort_values("Total", ascending=True)
                  .drop(columns="Total")
    )

    plt.figure(figsize=(10, 14))  # tall figure for many labels
    y_positions = range(len(label_dist))

    plt.barh(y_positions, label_dist["Train"], label="Train", alpha=0.7)
    plt.barh(
        y_positions,
        label_dist["Val"],
        left=label_dist["Train"],
        label="Val",
        alpha=0.7,
    )
    plt.barh(
        y_positions,
        label_dist["Test"],
        left=label_dist["Train"] + label_dist["Val"],
        label="Test",
        alpha=0.7,
    )

    plt.yticks(y_positions, label_dist.index)
    plt.xlabel("Sample count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

#===================================================================
# Later for loading:
import tensorflow as tf
import json
import os

# --- load all models ---
model_paths = {
    "M1": "saved_models/model_1.keras",
    "M2": "saved_models/model_2.keras",
    "M3": "saved_models/model_3.keras",
    "M4": "saved_models/model_4.keras",
    "M5": "saved_models/model_5.keras",
    "M6": "saved_models/model_6.keras",
}

models = {}
for key, path in model_paths.items():
    if os.path.exists(path):
        models[key] = tf.keras.models.load_model(path)  # [web:569][web:570][web:573]
    else:
        print(f"Warning: model file not found: {path}")

# --- load all histories (as dicts) ---
history_paths = {
    "M1": "saved_histories/history_1.json",
    "M2": "saved_histories/history_2.json",
    "M3": "saved_histories/history_3.json",
    "M4": "saved_histories/history_4.json",
    "M5": "saved_histories/history_5.json",
    "M6": "saved_histories/history_6.json",
}

histories = {}
for key, path in history_paths.items():
    if os.path.exists(path):
        with open(path) as f:
            histories[key] = json.load(f)  # dict: metric -> list of values[web:568][web:579]
    else:
        print(f"Warning: history file not found: {path}")

# Example: access best val accuracy for M3 from loaded history
if "M3" in histories:
    best_val_acc_m3 = max(histories["M3"]["val_accuracy"])
    print("Best val_accuracy for M3 (EffNetB0):", best_val_acc_m3)