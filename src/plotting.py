import numpy as np
import matplotlib.pyplot as plt


def plot_waveforms(
    data,
    channel_keys,
    title="Check waveforms",
    num_samples=5,
    figsize=(8, 4),
    alpha=0.25,
):
    """
    Plot the waveforms for a given data.
    If channel_keys is empty, plot the waveforms for all channels.

    Args:
        data: The data to plot.
        channel_keys: The channels to plot.
        title: The title of the plot.
        num_samples: The number of samples to plot.
        figsize: The size of the figure.
        alpha: The alpha of the plot.
    Returns:
        The figure and axes.
    """

    if len(channel_keys) == 0:
        num_row = data.shape[1]
        fig, axes = plt.subplots(num_row, 1, figsize=figsize, sharex=True, sharey=True)

        for row_idx in range(num_row):
            ax = axes[row_idx]
            current_data = data[:, row_idx]
            ax.plot(current_data, alpha=alpha)

    else:
        fig, axes = plt.subplots(
            len(channel_keys), 1, figsize=figsize, sharex=True, sharey=True
        )

        if len(channel_keys) == 1:
            axes = [axes]

        for row_idx, channel in enumerate(channel_keys):
            ax = axes[row_idx]
            current_data = data[channel]

            random_indices = np.random.choice(
                current_data.shape[0],
                size=min(num_samples, current_data.shape[0]),
                replace=False,
            )

            for i in random_indices:
                ax.plot(current_data[i, :], alpha=alpha)

            ax.set_ylabel(channel)

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes
