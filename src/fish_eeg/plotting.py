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

        for row_idx, channel in enumerate(sorted(channel_keys)):
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


def plot_fft(
    magnitudes,
    frequencies,
    myfreq,
    myamp,
    subjid,
    dataset_index,
    channel_keys,
    period_keys=None,
    title="Check waveforms",
    num_samples=5,
    figsize=(8, 4),
    alpha=0.25,
    xlim=[],
    ylim=[],
):
    if len(channel_keys) == 0:
        fig, axes = plt.subplots(
            len(magnitudes), 1, figsize=figsize, sharex=True, sharey=True
        )
        for row_idx in range(len(magnitudes)):
            ax = axes[row_idx]
            cur_mag = magnitudes[row_idx]
            cur_freq = frequencies[row_idx]
            ax.plot(cur_freq, cur_mag, alpha=alpha)

            ax.set_ylabel(str(row_idx))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.axvline(
                x=myfreq * 2,
                color="red",
                linestyle="--",
                linewidth=0.5,
                label="threshold",
            )

    elif len(channel_keys) > 1 and len(period_keys) == 0:
        fig, axes = plt.subplots(
            len(channel_keys), 1, figsize=figsize, sharex=True, sharey=True
        )
        for row_idx, channel in enumerate(channel_keys):
            ax = axes[row_idx]
            cur_mag = magnitudes[channel]
            cur_freq = frequencies[channel]

            random_indices = np.random.choice(
                cur_mag.shape[0], size=min(num_samples, cur_mag.shape[0]), replace=False
            )

            for i in random_indices:
                ax.plot(cur_freq[i, :], cur_mag[i, :], alpha=alpha)

            ax.set_ylabel(channel)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.axvline(
                x=myfreq * 2,
                color="red",
                linestyle="--",
                linewidth=0.5,
                label="threshold",
            )

    elif len(channel_keys) > 0 and len(period_keys) > 0:
        fig, axes = plt.subplots(
            len(channel_keys),
            len(period_keys),
            figsize=figsize,
            sharex=True,
            sharey=True,
        )

        for col_idx, period in enumerate(period_keys):
            for row_idx, channel in enumerate(channel_keys):
                ax = axes[row_idx, col_idx]
                cur_mag = magnitudes[period][channel]
                cur_freq = frequencies[period][channel]

                random_indices = np.random.choice(
                    cur_mag.shape[0],
                    size=min(num_samples, cur_mag.shape[0]),
                    replace=False,
                )

                for i in random_indices:
                    ax.plot(cur_freq[i, :], cur_mag[i, :], alpha=alpha)

                ax.set_ylabel(channel)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.axvline(
                    x=myfreq * 2,
                    color="red",
                    linestyle="--",
                    linewidth=0.5,
                    label="threshold",
                )

    fig.suptitle(f"{subjid} {myfreq} Hz {myamp} dB: {title}")
    fig.tight_layout()
    plt.savefig(
        f"plots/{subjid}/{subjid}_{myfreq}Hz_{myamp}dB_{title}_{dataset_index}.png",
        format="png",
        bbox_inches="tight",
        dpi=600,
    )
    return fig, axes
