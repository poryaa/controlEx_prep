import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(
    history,
    figsize=(8, 5),
    xlim=(0, 30),
    ylim=(0, 1),
    grid=True,
    xlabel="Epoch",
    style=None,
    title=None,
    ax=None,
    show=True,
):
    """
    Plot Keras/TensorFlow training history.

    Parameters
    ----------
    history : keras.callbacks.History or dict
        Training history object returned by model.fit(), or a history dict.
    figsize : tuple, default=(8, 5)
        Figure size.
    xlim : tuple or None, default=(0, 30)
        X-axis limits.
    ylim : tuple or None, default=(0, 1)
        Y-axis limits.
    grid : bool, default=True
        Whether to show grid.
    xlabel : str, default="Epoch"
        Label for x-axis.
    style : list or None
        Line styles for plotted columns.
    title : str or None
        Optional chart title.
    ax : matplotlib.axes.Axes or None
        Existing axes to plot on.
    show : bool, default=True
        Whether to call plt.show().

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes.
    """
    history_dict = history.history if hasattr(history, "history") else history
    history_df = pd.DataFrame(history_dict)

    ax = history_df.plot(
        figsize=figsize if ax is None else None,
        xlim=xlim,
        ylim=ylim,
        grid=grid,
        xlabel=xlabel,
        style=style,
        ax=ax,
    )

    if title:
        ax.set_title(title)

    if show:
        plt.show()

    return ax