from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import History


def plot_training_history(history: History) -> None:
    """
    Plot training and validation loss/accuracy curves from a Keras History object.

    Parameters
    ----------
    history : History
        History object returned by `model.fit()`.

    Raises
    ------
    ValueError
        If required metrics are not found in `history.history`.
    """
    history_dict = history.history

    loss = history_dict.get("loss")
    val_loss = history_dict.get("val_loss")
    accuracy = history_dict.get("accuracy")
    val_accuracy = history_dict.get("val_accuracy")

    if loss is None or val_loss is None:
        raise ValueError("Expected 'loss' and 'val_loss' in history.history.")

    if accuracy is None or val_accuracy is None:
        raise ValueError("Expected 'accuracy' and 'val_accuracy' in history.history.")

    epochs = range(1, len(loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, accuracy, label="train_accuracy")
    axes[1].plot(epochs, val_accuracy, label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def load_and_prep_image(filename: str, img_size: int = 224) -> tf.Tensor:
    """
    Load an image file, decode it as RGB, resize it, and scale pixel values to [0, 1].

    Parameters
    ----------
    filename : str
        Path to the image file.
    img_size : int, optional
        Target height and width for resizing, by default 224.

    Returns
    -------
    tf.Tensor
        Preprocessed image tensor of shape `(img_size, img_size, 3)`.
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def create_tensorboard_callback(
    log_dir_root: str,
    experiment_name: str,
) -> TensorBoard:
    """
    Create a TensorBoard callback with a timestamped log directory.

    Parameters
    ----------
    log_dir_root : str
        Root directory for TensorBoard logs.
    experiment_name : str
        Name of the experiment.

    Returns
    -------
    TensorBoard
        Configured TensorBoard callback.
    """
    log_dir = Path(log_dir_root) / experiment_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    callback = TensorBoard(log_dir=str(log_dir))

    print(f"Saving TensorBoard log files to: {log_dir}")
    return callback