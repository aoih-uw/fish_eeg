from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from fish_eeg.utils import select_doub_freq_bin


def plot_waveforms(
    data,
    channel_keys=None,
    num_samples=5,
    alpha=0.25,
    title=None,
    figsize=(8, 4),
    parent_grid=None,
    save_path=None,
):
    """
    Flexible waveform plotter.

    Modes:
    - If channel_keys is None or empty:
         Treat `data` as a 2D array (n_samples x n_channels) and plot each column.
    - If channel_keys is provided:
         Treat `data` as a dict mapping channel -> array.
    - If parent_grid is provided:
         Plot inside an external GridSpec (no new figure).
    """

    # ---------------------------------------------
    # CASE 0: Determine operating mode
    # ---------------------------------------------
    array_mode = channel_keys is None or len(channel_keys) == 0

    if array_mode:
        # Ensure array is 2D
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[:, None]  # shape -> (N,1)
        n_rows = data.shape[1]
    else:
        n_rows = len(channel_keys)

    # ---------------------------------------------
    # CASE 1: Standalone figure
    # ---------------------------------------------
    if parent_grid is None:
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True, sharey=True)
        if n_rows == 1:
            axes = [axes]

        if array_mode:
            # -------------------------------------
            # ARRAY MODE: plot each column of data
            # -------------------------------------
            for i in range(n_rows):
                ax = axes[i]
                ax.plot(data[:, i], alpha=alpha)
                ax.set_ylabel(f"col{i}")

        else:
            # -------------------------------------
            # DICT MODE: plot from channel_keys
            # -------------------------------------
            for idx, channel in enumerate(channel_keys):
                ax = axes[idx]
                current_data = data[channel]

                random_indices = np.random.choice(
                    current_data.shape[0],
                    size=min(num_samples, current_data.shape[0]),
                    replace=False,
                )

                for r in random_indices:
                    ax.plot(current_data[r, :], alpha=alpha)

                ax.set_ylabel(channel)

        if title:
            fig.suptitle(title)
        fig.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, axes

    # ---------------------------------------------
    # CASE 2: Nested mode inside parent_grid
    # ---------------------------------------------
    else:
        axes = []

        if array_mode:
            # Each column gets its own subplot
            for i in range(n_rows):
                ax = plt.subplot(parent_grid[i])
                axes.append(ax)
                ax.plot(data[:, i], alpha=alpha)
                ax.set_ylabel(f"col{i}")

        else:
            # Dict-of-arrays mode
            for idx, channel in enumerate(channel_keys):
                ax = plt.subplot(parent_grid[idx])
                axes.append(ax)

                current_data = data[channel]
                random_indices = np.random.choice(
                    current_data.shape[0],
                    size=min(num_samples, current_data.shape[0]),
                    replace=False,
                )
                for r in random_indices:
                    ax.plot(current_data[r, :], alpha=alpha)

                ax.set_ylabel(channel)

        if title:
            axes[0].set_title(title)

        return axes


def plot_fft(
    magnitudes,
    frequencies,
    myfreq,
    myamp,
    subjid,
    dataset_index,
    channel_keys,
    period_keys=None,
    title="Check FFT",
    num_samples=5,
    figsize=(8, 4),
    alpha=0.25,
    xlim=None,
    ylim=None,
    parent_grid=None,
):
    """
    Clean FFT plotting function with correct standalone vs embedded behavior.
    """

    # ---------------------------
    # 0. Embedded mode detection
    # ---------------------------
    standalone = parent_grid is None

    if period_keys is None:
        period_keys = []

    # ---------------------------
    # 1. Embedded mode = draw on provided axis and RETURN EARLY
    # ---------------------------
    if not standalone:
        axes = []
        if len(channel_keys) == 0:
            for i, (mag_arr, freq_arr) in enumerate(zip(magnitudes, frequencies)):
                ax = plt.subplot(parent_grid[i])
                axes.append(ax)
                ax.plot(freq_arr, mag_arr, alpha=alpha)
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)
                ax.set_ylabel(f"Row {i}")

                ax.axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.6)

        # Case 2: Only channels, no periods
        elif len(channel_keys) > 0 and len(period_keys) == 0:
            for idx, channel in enumerate(channel_keys):
                ax = plt.subplot(parent_grid[idx])
                axes.append(ax)
                cur_mag = magnitudes[channel]
                cur_freq = frequencies[channel]

                random_idxs = np.random.choice(
                    cur_mag.shape[0],
                    size=min(num_samples, cur_mag.shape[0]),
                    replace=False,
                )
                for idx in random_idxs:
                    ax.plot(cur_freq[idx, :], cur_mag[idx, :], alpha=alpha)

                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)
                ax.set_ylabel(channel)

                ax.axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.6)

        # Case 3: Channels × Periods
        else:
            for c_idx, period in enumerate(period_keys):
                for r_idx, channel in enumerate(channel_keys):
                    ax = plt.subplot(parent_grid[r_idx, c_idx])
                    axes.append(ax)
                    cur_mag = magnitudes[period][channel]
                    cur_freq = frequencies[period][channel]

                    random_idxs = np.random.choice(
                        cur_mag.shape[0],
                        size=min(num_samples, cur_mag.shape[0]),
                        replace=False,
                    )
                    for idx in random_idxs:
                        ax.plot(cur_freq[idx, :], cur_mag[idx, :], alpha=alpha)

                    if xlim:
                        ax.set_xlim(xlim)
                    if ylim:
                        ax.set_ylim(ylim)
                    ax.set_ylabel(channel)

                    ax.axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.6)
        if title:
            axes[0].set_title(title)
        return axes

    # ======================================================
    # ============= 2. STANDALONE MODE =====================
    # ======================================================

    # Case 1: No channels (list-of-arrays)
    if len(channel_keys) == 0:
        fig, axes = plt.subplots(
            len(magnitudes), 1, figsize=figsize, sharex=True, sharey=True
        )

        if len(magnitudes) == 1:
            axes = [axes]

        for i, (mag_arr, freq_arr) in enumerate(zip(magnitudes, frequencies)):
            ax = axes[i]
            ax.plot(freq_arr, mag_arr, alpha=alpha)
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            ax.set_ylabel(f"Row {i}")

            ax.axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.6)

    # Case 2: Only channels, no periods
    elif len(channel_keys) > 0 and len(period_keys) == 0:
        fig, axes = plt.subplots(
            len(channel_keys), 1, figsize=figsize, sharex=True, sharey=True
        )

        if len(channel_keys) == 1:
            axes = [axes]

        for i, channel in enumerate(channel_keys):
            ax = axes[i]
            cur_mag = magnitudes[channel]
            cur_freq = frequencies[channel]

            random_idxs = np.random.choice(
                cur_mag.shape[0],
                size=min(num_samples, cur_mag.shape[0]),
                replace=False,
            )
            for idx in random_idxs:
                ax.plot(cur_freq[idx, :], cur_mag[idx, :], alpha=alpha)

            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            ax.set_ylabel(channel)

            ax.axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.6)

    # Case 3: Channels × Periods
    else:
        fig, axes = plt.subplots(
            len(channel_keys),
            len(period_keys),
            figsize=figsize,
            sharex=True,
            sharey=True,
        )

        for c_idx, period in enumerate(period_keys):
            for r_idx, channel in enumerate(channel_keys):
                ax = axes[r_idx, c_idx]
                cur_mag = magnitudes[period][channel]
                cur_freq = frequencies[period][channel]

                random_idxs = np.random.choice(
                    cur_mag.shape[0],
                    size=min(num_samples, cur_mag.shape[0]),
                    replace=False,
                )
                for idx in random_idxs:
                    ax.plot(cur_freq[idx, :], cur_mag[idx, :], alpha=alpha)

                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)
                ax.set_ylabel(channel)

                ax.axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.6)

    # ---------------------------
    # 3. Finalize figure
    # ---------------------------
    fig.suptitle(f"{subjid} {myfreq} Hz {myamp} dB: {title}")
    fig.tight_layout()

    # Save
    fig.savefig(
        f"plots/{subjid}/{subjid}_{myfreq}Hz_{myamp}dB_{title}_{dataset_index}.png",
        dpi=600,
        bbox_inches="tight",
    )

    return None


def plot_compare_denoised_waveform(
    compare_denoising_mean,
    compare_denoising_std,
    channel_keys,
    parent_grid=None,
    save_path=None,
):
    """
    Plot original vs denoised mean±std waveforms for each channel.

    - If parent_grid is None:
        Create a standalone figure with shape (n_channels, 2).
    - If parent_grid is a GridSpec:
        Use it as an inner grid with shape (n_channels, 2) and plot into it.
        (e.g. created via outer_cell.subgridspec(n_channels, 2, ...))
    """
    n_channels = len(channel_keys)

    # =========================
    # Standalone figure mode
    # =========================
    if parent_grid is None:
        fig, axes = plt.subplots(
            n_channels,
            2,
            figsize=(12, 3 * n_channels),
            sharex=True,
            sharey=True,
        )

        # Make sure axes is always 2D: (n_channels, 2)
        if n_channels == 1:
            axes = np.array([axes])  # shape (1, 2)

        for i, channel in enumerate(channel_keys):
            # Left column: Original
            ax_orig = axes[i, 0]
            mean_orig = compare_denoising_mean["original"][channel]
            std_orig = compare_denoising_std["original"][channel]

            ax_orig.plot(mean_orig, label="Mean", color="#5778a4")
            ax_orig.fill_between(
                range(len(mean_orig)),
                mean_orig - std_orig,
                mean_orig + std_orig,
                alpha=0.3,
                color="#5778a4",
            )
            ax_orig.set_title(f"{channel} - Original")
            ax_orig.set_ylabel("Value")
            if i == n_channels - 1:
                ax_orig.set_xlabel("Time / Sample")

            # Right column: Denoised
            ax_denoise = axes[i, 1]
            mean_denoise = compare_denoising_mean["denoised"][channel]
            std_denoise = compare_denoising_std["denoised"][channel]

            ax_denoise.plot(mean_denoise, label="Mean", color="#e49444")
            ax_denoise.fill_between(
                range(len(mean_denoise)),
                mean_denoise - std_denoise,
                mean_denoise + std_denoise,
                alpha=0.3,
                color="#e49444",
            )
            ax_denoise.set_title(f"{channel} - Denoised")
            ax_denoise.set_ylabel("Value")
            if i == n_channels - 1:
                ax_denoise.set_xlabel("Time / Sample")

        plt.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return None

    # =========================
    # Grid cell / parent_grid mode
    # =========================
    else:
        # parent_grid is expected to be something like:
        # inner_grid = outer_grid[row_idx, col_idx].subgridspec(n_channels, 2, ...)
        fig = plt.gcf()
        axes = np.empty((n_channels, 2), dtype=object)

        for i, channel in enumerate(channel_keys):
            # Create axes INSIDE the given parent_grid
            ax_orig = fig.add_subplot(parent_grid[i, 0])
            ax_denoise = fig.add_subplot(parent_grid[i, 1])

            axes[i, 0] = ax_orig
            axes[i, 1] = ax_denoise

            # Left column: Original
            mean_orig = compare_denoising_mean["original"][channel]
            std_orig = compare_denoising_std["original"][channel]

            ax_orig.plot(mean_orig, label="Mean", color="#5778a4")
            ax_orig.fill_between(
                range(len(mean_orig)),
                mean_orig - std_orig,
                mean_orig + std_orig,
                alpha=0.3,
                color="#5778a4",
            )
            ax_orig.set_title(f"{channel} - Original", fontsize=8)
            if i == n_channels - 1:
                ax_orig.set_xlabel("Time / Sample")

            # Right column: Denoised
            mean_denoise = compare_denoising_mean["denoised"][channel]
            std_denoise = compare_denoising_std["denoised"][channel]

            ax_denoise.plot(mean_denoise, label="Mean", color="#e49444")
            ax_denoise.fill_between(
                range(len(mean_denoise)),
                mean_denoise - std_denoise,
                mean_denoise + std_denoise,
                alpha=0.3,
                color="#e49444",
            )
            ax_denoise.set_title(f"{channel} - Denoised", fontsize=8)
            if i == n_channels - 1:
                ax_denoise.set_xlabel("Time / Sample")

        # No
        return axes


def plot_compare_denoised_fft(
    fft_freq_vecs,
    fft_magnitudes_mean,
    fft_magnitudes_std,
    channel_keys,
    myfreq,
    myamp,
    parent_grid=None,
    fft_ylim=None,
    freq_padding=100,
    save_path=None,
):
    """
    Plot original vs denoised FFT mean±std for each channel.

    Parameters
    ----------
    fft_freq_vecs : dict
        {
          "original": {channel: 1D array of freqs},
          "denoised": {channel: 1D array of freqs},
        }
    fft_magnitudes_mean : dict
        {
          "original": {channel: 1D array of mean magnitudes},
          "denoised": {...},
        }
    fft_magnitudes_std : dict
        {
          "original": {channel: 1D array of std magnitudes},
          "denoised": {...},
        }
    channel_keys : list[str]
        e.g. ["ch1", "ch2", "ch3", "ch4"]
    myfreq : float
        Stimulus frequency (Hz).
    myamp : float
        Stimulus amplitude (dB). (only used for naming / titles if desired)
    parent_grid : None or GridSpec / SubplotSpec
        - If None: create a standalone figure (n_channels x 2).
        - Else: draw into parent_grid[i, j] as an inner grid.
    fft_ylim : tuple or None
        y-limits shared across subplots.
    freq_padding : float
        +/- padding around 2 * myfreq for x-limits.
    save_path : str or None
        If provided in standalone mode, save figure to this path.

    Returns
    -------
    None (standalone mode)
    axes (grid mode): np.ndarray of shape (n_channels, 2) of Axes
    """

    data_types = ["original", "denoised"]
    n_channels = len(channel_keys)
    n_data_types = len(data_types)

    # =========================
    # Standalone figure mode
    # =========================
    if parent_grid is None:
        fig, axes = plt.subplots(
            n_channels,
            n_data_types,
            figsize=(12, 3 * n_channels),
            sharex=True,
            sharey=True,
        )

        # Ensure axes is 2D (n_channels, 2)
        if n_channels == 1:
            axes = np.array([axes])  # shape -> (1, 2)

        for i, channel in enumerate(channel_keys):
            for j, dtype in enumerate(data_types):
                ax = axes[i, j]

                freqs = fft_freq_vecs[dtype][channel]
                mean_mag = fft_magnitudes_mean[dtype][channel]
                std_mag = fft_magnitudes_std[dtype][channel]

                # Color by type
                color = "#5778a4" if dtype == "original" else "#e49444"

                ax.plot(freqs, mean_mag, linewidth=1.5, color=color, label="Mean")
                ax.fill_between(
                    freqs,
                    mean_mag - std_mag,
                    mean_mag + std_mag,
                    alpha=0.3,
                    color=color,
                    label="±1 SD",
                )

                # Titles and labels
                ax.set_title(f"{channel} - {dtype}", fontsize=10)
                if i == n_channels - 1:
                    ax.set_xlabel("Frequency (Hz)")
                if j == 0:
                    ax.set_ylabel("Magnitude")

                # Limits
                if fft_ylim is not None:
                    ax.set_ylim(fft_ylim)
                else:
                    max_original = max(
                        [max(v) for i, v in fft_magnitudes_mean["original"].items()]
                    )
                    max_denoised = max(
                        [max(v) for i, v in fft_magnitudes_mean["denoised"].items()]
                    )
                    max_total = max(max_original, max_denoised)
                    ax.set_ylim([0, max_total + 0.005])
                ax.set_xlim([myfreq * 2 - freq_padding, myfreq * 2 + freq_padding])
                ax.axvline(x=myfreq * 2, color="r", linestyle="--", linewidth=0.5)

                ax.grid(True, alpha=0.3)

                # One legend is usually enough, but this keeps them per-panel
                ax.legend(fontsize=6)

        plt.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return None

    # =========================
    # Grid cell / parent_grid mode
    # =========================
    else:
        fig = plt.gcf()
        axes = np.empty((n_channels, n_data_types), dtype=object)

        for i, channel in enumerate(channel_keys):
            for j, dtype in enumerate(data_types):
                ax = fig.add_subplot(parent_grid[i, j])
                axes[i, j] = ax

                freqs = fft_freq_vecs[dtype][channel]
                mean_mag = fft_magnitudes_mean[dtype][channel]
                std_mag = fft_magnitudes_std[dtype][channel]

                color = "#5778a4" if dtype == "original" else "#e49444"

                ax.plot(freqs, mean_mag, linewidth=1.2, color=color, label="Mean")
                ax.fill_between(
                    freqs,
                    mean_mag - std_mag,
                    mean_mag + std_mag,
                    alpha=0.3,
                    color=color,
                    label="±1 SD",
                )

                # Titles compact in grid mode
                ax.set_title(f"{channel} - {dtype}", fontsize=8)

                if i == n_channels - 1:
                    ax.set_xlabel("Frequency (Hz)")
                if j == 0:
                    ax.set_ylabel("Magnitude")

                if fft_ylim is not None:
                    ax.set_ylim(fft_ylim)
                else:
                    max_original = max(
                        [max(v) for i, v in fft_magnitudes_mean["original"].items()]
                    )
                    max_denoised = max(
                        [max(v) for i, v in fft_magnitudes_mean["denoised"].items()]
                    )
                    max_total = max(max_original, max_denoised)
                    ax.set_ylim([0, max_total + 0.005])
                ax.set_xlim([myfreq * 2 - freq_padding, myfreq * 2 + freq_padding])
                ax.axvline(x=myfreq * 2, color="r", linestyle="--", linewidth=0.5)

                ax.grid(True, alpha=0.3)

                # Usually skip legends in grid mode for clutter,
                # but you could enable for one panel if you want.

        return axes


def plot_bootstrap_fft_comparison(
    bootstrap_means,
    bootstrap_stds,
    weighted_freq_vec,
    myfreq,
    myamp,
    subjid,
    dataset_index,
    idx=0,
    xlim=None,
    ylim=None,
    save_fig=False,
    save_path=None,
    ax1=None,
    ax2=None,
):
    """
    Compare bootstrap FFT (prestim vs stimresp) either as a standalone figure
    or inside provided axes (for grid use).

    Parameters
    ----------
    mean_fft_prestim, std_fft_prestim : 1D arrays
    mean_fft_stimresp, std_fft_stimresp : 1D arrays
    freq_vec : 1D array
    myfreq, myamp : float
    subjid : str
    dataset_index : int
    xlim, ylim : tuple or None
    save_fig : bool
        Only used in standalone mode.
    save_path : str or None
        If not None, prefix for saving.
    ax1, ax2 : matplotlib.axes.Axes or None
        - If both None: create a new figure with 1x2 subplots (standalone).
        - If provided: draw into these axes (grid mode).

    Returns
    -------
    fig (standalone mode) or (ax1, ax2) (grid mode)
    """

    mean_fft_prestim = bootstrap_means["prestim"][idx]
    std_fft_prestim = bootstrap_stds["prestim"][idx]

    mean_fft_stimresp = bootstrap_means["stimresp"][idx]
    std_fft_stimresp = bootstrap_stds["stimresp"][idx]

    freq_vec = weighted_freq_vec["prestim"]["ch1"][idx]

    standalone = ax1 is None or ax2 is None

    # ------------------------
    # Set up figure / axes
    # ------------------------
    if standalone:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    else:
        fig = ax1.figure

    # ------------------------
    # Prestim FFT
    # ------------------------
    ax1.plot(freq_vec, mean_fft_prestim, "-", color="#5778a4", label="Mean FFT")
    ax1.fill_between(
        freq_vec,
        mean_fft_prestim - std_fft_prestim,
        mean_fft_prestim + std_fft_prestim,
        alpha=0.3,
        color="#5778a4",
        label="±1 STD",
    )

    ax1.axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.5, alpha=0.7)

    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude")
    ax1.set_title("Stim OFF")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ------------------------
    # Stimresp FFT
    # ------------------------
    ax2.plot(freq_vec, mean_fft_stimresp, "-", color="#e49444", label="Mean FFT")
    ax2.fill_between(
        freq_vec,
        mean_fft_stimresp - std_fft_stimresp,
        mean_fft_stimresp + std_fft_stimresp,
        alpha=0.3,
        color="#e49444",
        label="±1 STD",
    )

    ax2.axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.5, alpha=0.7)

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_title("Stim ON")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # ------------------------
    # Limits
    # ------------------------
    if xlim is not None:
        ax1.set_xlim(xlim)
    else:
        # your original default
        xlim = [myfreq * 2 - myamp, myfreq * 2 + myamp]
        ax1.set_xlim(xlim)
    ax2.set_xlim(ax1.get_xlim())

    if ylim is not None:
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)

    # ------------------------
    # Save / return
    # ------------------------
    if standalone:
        plt.tight_layout()

        if save_fig:
            filename = f"{subjid}_{myfreq}Hz_{myamp}dB_grand_avg_comparison_{dataset_index}.png"
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                filename = os.path.join(save_path, filename)
            else:
                os.makedirs(f"plots/{subjid}", exist_ok=True)
                filename = os.path.join(f"plots/{subjid}", filename)
            fig.savefig(filename, dpi=600, bbox_inches="tight")

        return None
    else:
        # In grid mode we don't touch layout or save; caller owns the figure.
        return ax1, ax2


def plot_doub_freq_hist(
    doub_freq_dict,
    subjid,
    submetric,
    myfreq,
    myamp,
    bin_num,
    dataset_index,
    ax=None,
    log_y=True,
    save_path=None,
):
    """
    Plot histograms of 2x-response magnitudes for prestim vs stimresp.

    Parameters
    ----------
    doub_freq_dict : dict
        {
          "prestim": {submetric: 1D array},
          "stimresp": {submetric: 1D array},
        }
        for a single (freq, amp) combination.
    subjid : str
    submetric : str
        Key under 'prestim' and 'stimresp' to plot.
    myfreq : float
        Frequency (Hz).
    myamp : float
        Amplitude (dB).
    bin_num : int
        Number of bins.
    dataset_index : int
        For naming/saving.
    ax : matplotlib.axes.Axes or None
        - If None: create a new figure and axis (standalone mode).
        - If provided: plot into this axis (grid mode).
    log_y : bool
        Whether to use log-scale on y-axis.
    save_path : str or None
        Only used in standalone mode. Full file path (including filename).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis the histogram was drawn on.
    """

    standalone = ax is None

    if standalone:
        fig, ax = plt.subplots(figsize=(15, 10))
    else:
        fig = ax.figure

    # Plot histograms for both periods
    ax.hist(
        doub_freq_dict["prestim"][submetric],
        bins=bin_num,
        alpha=0.6,
        color="#5778a4",
        label="Prestim",
    )
    ax.hist(
        doub_freq_dict["stimresp"][submetric],
        bins=bin_num,
        alpha=0.6,
        color="#e49444",
        label="Stimresp",
    )

    # Titles / labels
    ax.set_title(f"{subjid} {myfreq}Hz {myamp}dB", fontsize=14)
    ax.set_xlabel("AEP Response Relative Magnitude", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    if log_y:
        ax.set_yscale("log")

    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(fontsize=10)

    if standalone:
        fig.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, format="png", bbox_inches="tight", dpi=600)
        plt.show()

    return ax


class Plotter:
    def __init__(self, eegdataset):
        self.eegdataset = eegdataset
        self.save_path = (
            "/net/noble/vol3/user/ssontha2/classes/CSE583/fish_eeg/analysis/yash_plots"
        )

    def plot_waveforms_by_frequency_rows(
        self,
        attr="rms_subsampled_data",
        num_samples=10,
        alpha=0.3,
        cell_width=8,
        cell_height=4,
        channel_keys=None,
        save_path=None,
    ):
        data = getattr(self.eegdataset, attr)
        if isinstance(data, dict):
            pass
        else:
            data = data.item()

        if channel_keys is None or len(channel_keys) == 0:
            channel_keys = []

        if num_samples is None or num_samples == []:
            num_samples = []

        # Group by freq
        freq_groups = defaultdict(list)
        for (freq, db), arr in data.items():
            freq_groups[freq].append((db, arr))

        freqs_sorted = sorted(freq_groups.keys())
        for freq in freqs_sorted:
            freq_groups[freq].sort(key=lambda x: x[0])

        n_rows = len(freqs_sorted)
        max_cols = max(len(dbs) for dbs in freq_groups.values())

        fig = plt.figure(figsize=(cell_width * max_cols, cell_height * n_rows))
        outer_grid = fig.add_gridspec(
            n_rows,
            max_cols,
            wspace=0.5,  # horizontal space between cells
            hspace=0.5,
        )  # vertical space between cells)

        for row_idx, freq in enumerate(freqs_sorted):
            row_dbs = freq_groups[freq]

            for col_idx in range(max_cols):
                outer_ax = fig.add_subplot(outer_grid[row_idx, col_idx])

                if col_idx < len(row_dbs):
                    db, arrayordict = row_dbs[col_idx]

                    # --- nested subgrid for individual channel plots ---
                    if len(channel_keys) > 0:
                        inner_grid = outer_grid[row_idx, col_idx].subgridspec(
                            len(channel_keys), 1, hspace=1.0, wspace=1.0
                        )
                    else:
                        arrayordict = arrayordict["S"]
                        inner_grid = outer_grid[row_idx, col_idx].subgridspec(
                            arrayordict.shape[1], 1, hspace=0.3
                        )

                    plot_waveforms(
                        data=arrayordict,
                        channel_keys=channel_keys,
                        num_samples=num_samples,
                        alpha=alpha,
                        title=f"{freq} Hz, {db} dB",
                        parent_grid=inner_grid,
                    )

                    outer_ax.axis("off")

                else:
                    outer_ax.axis("off")

        plt.tight_layout()

        if save_path is not None:
            save_path = save_path + "/waveforms"
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, f"{attr}_{num_samples}.png"),
                dpi=300,
                bbox_inches="tight",
            )

        return None

    def plot_fft_all_ica(
        self,
        attr,
        channel_keys,
        period_keys,
        subjid,
        dataset_index,
        num_samples=5,
        alpha=0.25,
        cell_width=4,
        cell_height=3,
        save_path=None,
    ):
        """
        Creates a grid:
            rows = frequencies
            cols = amplitudes (dB)
        And calls your original plot_fft() in each cell.
        """

        data = getattr(self.eegdataset, attr)

        if channel_keys is None or len(channel_keys) == 0:
            channel_keys = []

        if period_keys is None or len(period_keys) == 0:
            period_keys = []

        # Group by freq
        freq_groups = defaultdict(list)
        for (freq, db), arr in data.items():
            freq_groups[freq].append((db, arr))

        freqs_sorted = sorted(freq_groups.keys())
        for freq in freqs_sorted:
            freq_groups[freq].sort(key=lambda x: x[0])

        n_rows = len(freqs_sorted)
        max_cols = max(len(dbs) for dbs in freq_groups.values())

        fig = plt.figure(figsize=(8 * max_cols, 4 * n_rows))
        outer_grid = fig.add_gridspec(
            n_rows,
            max_cols,
            wspace=0.5,  # horizontal space between cells
            hspace=0.5,
        )  # vertical space between cells)

        for row_idx, freq in enumerate(freqs_sorted):
            row_dbs = freq_groups[freq]

            for col_idx in range(max_cols):
                outer_ax = fig.add_subplot(outer_grid[row_idx, col_idx])

                if col_idx < len(row_dbs):
                    db, arrayordict = row_dbs[col_idx]

                    # --- nested subgrid for individual channel plots ---
                    if len(channel_keys) > 0 and len(period_keys) > 0:
                        inner_grid = outer_grid[row_idx, col_idx].subgridspec(
                            len(channel_keys), len(period_keys), hspace=1.0, wspace=1.0
                        )
                    else:
                        inner_grid = outer_grid[row_idx, col_idx].subgridspec(
                            arrayordict[0].shape[0], 1, hspace=1.0, wspace=1.0
                        )

                    plot_fft(
                        arrayordict[0],
                        arrayordict[1],
                        freq,
                        db,
                        subjid,
                        0,
                        channel_keys=channel_keys,
                        period_keys=period_keys,
                        title=f"{freq} Hz, {db} dB",
                        num_samples=num_samples,
                        figsize=(8, 4),
                        alpha=alpha,
                        xlim=[freq * 2 - db, freq * 2 + db],
                        parent_grid=inner_grid,
                    )

                    outer_ax.axis("off")
                else:
                    outer_ax.axis("off")
        plt.tight_layout()

        if save_path is not None:
            save_path = save_path + "/ffts"
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, f"{attr}_{num_samples}.png"),
                dpi=300,
                bbox_inches="tight",
            )

        return None

    def plot_compare_denoised_by_frequency_rows(
        self,
        attr="reconstructed_ica_data",
        cell_width=15,
        cell_height=10,
        wspace=0.2,
        hspace=0.2,
        save_path=None,
    ):
        """
        Big grid of denoising comparison plots.

        - Each row = frequency
        - Each column = amplitude (dB)
        - Inside each cell: original vs denoised waveforms for all channels
        plotted by `plot_compare_denoised_waveform` using a sub-gridspec.

        Parameters
        ----------
        compare_denoising_mean : dict
            Dict keyed by (freq, db) -> {
                "original": {channel: 1D array},
                "denoised": {channel: 1D array},
            }
        compare_denoising_std : dict
            Same keys as compare_denoising_mean, with std arrays.
        channel_keys : list[str]
            List of channel names like ["ch1", "ch2", "ch3", "ch4"].
        cell_width, cell_height : float
            Size of each (freq, dB) cell in inches.
        wspace, hspace : float
            Spacing between outer grid cells.
        save_path : str or None
            If provided, save the big figure here.

        Returns
        -------
        fig : matplotlib.figure.Figure
        outer_grid : matplotlib.gridspec.GridSpec
        """

        # --- group (freq, db) pairs by freq ---

        data = getattr(self.eegdataset, attr)

        freq_groups = defaultdict(list)
        for freq, db in data.keys():
            freq_groups[freq].append(db)

        # sort freqs and amps
        freqs_sorted = sorted(freq_groups.keys())
        for freq in freqs_sorted:
            freq_groups[freq] = sorted(freq_groups[freq])

        n_rows = len(freqs_sorted)
        max_cols = max(len(dbs) for dbs in freq_groups.values())

        # --- make outer figure and grid ---
        fig = plt.figure(figsize=(cell_width * max_cols, cell_height * n_rows))
        outer_grid = fig.add_gridspec(
            n_rows,
            max_cols,
            wspace=wspace,
            hspace=hspace,
        )

        # --- fill each cell with an inner grid + plots ---
        for row_idx, freq in enumerate(freqs_sorted):
            db_list = freq_groups[freq]

            for col_idx in range(max_cols):
                if col_idx < len(db_list):
                    db = db_list[col_idx]

                    # inner: n_channels rows, 2 cols (original / denoised)
                    inner_grid = outer_grid[row_idx, col_idx].subgridspec(
                        len(self.eegdataset.channel_keys),
                        2,
                        hspace=0.3,
                        wspace=0.1,
                    )

                    # call your inner plotting function
                    plot_compare_denoised_waveform(
                        data[(freq, db)]["compare_denoising_mean"],
                        data[(freq, db)]["compare_denoising_std"],
                        channel_keys=self.eegdataset.channel_keys,
                        parent_grid=inner_grid,
                        save_path=None,  # only outer fig handles saving
                    )

                    # optional cell-level title on the top row of inner grid
                    # (attach to first channel/original axis)

                    ss = outer_grid[row_idx, col_idx]  # SubplotSpec
                    pos = ss.get_position(fig)  # Bbox in figure coords

                    fig.text(
                        x=pos.x0 + pos.width / 2.0,
                        y=pos.y1 + 0.005,  # top of the cell
                        s=f"{freq} Hz / {db} dB",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                else:
                    # empty cell: just turn off axis
                    empty_ax = fig.add_subplot(outer_grid[row_idx, col_idx])
                    empty_ax.axis("off")

        if save_path is not None:
            save_path = save_path + "/compare_denoised"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(
                os.path.join(save_path, f"{freq}Hz_{db}dB_compare_denoised.png"),
                dpi=300,
                bbox_inches="tight",
            )

        return None

    def plot_compare_denoised_fft_by_frequency_rows(
        self,
        attr="reconstructed_ica_data",
        cell_width=15,
        cell_height=10,
        wspace=0.2,
        hspace=0.2,
        fft_ylim=None,
        freq_padding=100,
        save_path=None,
    ):
        """
        Big grid of denoising comparison FFT plots.

        - Each row = frequency
        - Each column = amplitude (dB)
        - Inside each cell: original vs denoised FFTs for all channels
        plotted by `plot_compare_denoised_fft_cell` using a sub-gridspec.

        Parameters
        ----------
        attr : str
            Name of the attribute on `self.eegdataset` that holds a dict:
                {
                (freq, db): {
                    "fft_freq_vecs": {...},
                    "fft_magnitudes_mean": {...},
                    "fft_magnitudes_std": {...},
                },
                ...
                }
        cell_width, cell_height : float
            Size of each (freq, dB) cell in inches.
        wspace, hspace : float
            Spacing between outer grid cells.
        fft_ylim : tuple or None
            y-limits for magnitude (shared across all subplots).
        freq_padding : float
            +/- padding around 2 * freq for x-limits.
        save_path : str or None
            If provided, treated as a directory; the big figure is saved there.

        Returns
        -------
        fig : matplotlib.figure.Figure
        outer_grid : matplotlib.gridspec.GridSpec
        """

        data = getattr(self.eegdataset, attr)
        channel_keys = self.eegdataset.channel_keys

        # --- group (freq, db) pairs by freq ---
        freq_groups = defaultdict(list)
        for freq, db in data.keys():
            freq_groups[freq].append(db)

        # Sort freqs and amps
        freqs_sorted = sorted(freq_groups.keys())
        for freq in freqs_sorted:
            freq_groups[freq] = sorted(freq_groups[freq])

        n_rows = len(freqs_sorted)
        max_cols = max(len(dbs) for dbs in freq_groups.values())

        # --- make outer figure and grid ---
        fig = plt.figure(figsize=(cell_width * max_cols, cell_height * n_rows))
        outer_grid = fig.add_gridspec(
            n_rows,
            max_cols,
            wspace=wspace,
            hspace=hspace,
        )

        # --- fill each cell with an inner grid + FFT plots ---
        for row_idx, freq in enumerate(freqs_sorted):
            db_list = freq_groups[freq]

            for col_idx in range(max_cols):
                if col_idx < len(db_list):
                    db = db_list[col_idx]

                    inner_grid = outer_grid[row_idx, col_idx].subgridspec(
                        len(channel_keys),
                        2,  # original / denoised
                        hspace=0.3,
                        wspace=0.1,
                    )

                    fft_entry = data[(freq, db)]
                    fft_freq_vecs = fft_entry["fft_freq_vecs"]
                    fft_magnitudes_mean = fft_entry["fft_magnitudes_mean"]
                    fft_magnitudes_std = fft_entry["fft_magnitudes_std"]

                    plot_compare_denoised_fft(
                        fft_freq_vecs=fft_freq_vecs,
                        fft_magnitudes_mean=fft_magnitudes_mean,
                        fft_magnitudes_std=fft_magnitudes_std,
                        channel_keys=channel_keys,
                        myfreq=freq,
                        myamp=db,
                        parent_grid=inner_grid,
                        fft_ylim=fft_ylim,
                        freq_padding=freq_padding,
                    )

                    # Cell-level title (like in waveform version)
                    ss = outer_grid[row_idx, col_idx]
                    pos = ss.get_position(fig)

                    fig.text(
                        x=pos.x0 + pos.width / 2.0,
                        y=pos.y1 + 0.005,
                        s=f"{freq} Hz / {db} dB",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                else:
                    # empty cell: just turn off axis
                    empty_ax = fig.add_subplot(outer_grid[row_idx, col_idx])
                    empty_ax.axis("off")

        # Save once for the whole grid
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            out_path = os.path.join(
                save_path, "compare_denoised_fft_by_frequency_rows.png"
            )
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

        return fig, outer_grid

    def plot_doub_freq_hist_by_frequency_rows(
        self,
        attr="bootstrap_data",
        subjid="subj",
        submetric="SNR",
        bin_num=50,
        dataset_index=0,
        cell_width=5,
        cell_height=4,
        wspace=0.3,
        hspace=0.3,
        log_y=True,
        save_dir=None,
    ):
        """
        Big grid of 2x-response histograms.

        - Each row = frequency
        - Each column = amplitude (dB)
        - Inside each cell: prestim vs stimresp histogram for a given (freq, db).

        Parameters
        ----------
        attr : str
            Name of attribute on self.eegdataset containing:
                {
                (freq, db): doub_freq_dict,
                ...
                }
            where each doub_freq_dict looks like:
                {
                "prestim": {submetric: 1D array},
                "stimresp": {submetric: 1D array},
                }
        subjid : str
            Subject id for titles.
        submetric : str
            Key inside doub_freq_dict["prestim"] and ["stimresp"] to plot.
        bin_num : int
            Number of bins for histograms.
        dataset_index : int
            For naming/saving.
        cell_width, cell_height : float
            Size of each (freq, dB) cell in inches.
        wspace, hspace : float
            Spacing between outer grid cells.
        log_y : bool
            Whether to use log y-axis for all cells.
        save_dir : str or None
            Directory to save the big figure. If None, no saving.

        Returns
        -------
        fig : matplotlib.figure.Figure
        outer_grid : matplotlib.gridspec.GridSpec
        """

        data = getattr(self.eegdataset, attr)

        # --- group (freq, db) pairs by freq ---
        freq_groups = defaultdict(list)
        for freq, db in data.keys():
            freq_groups[freq].append(db)

        # Sort freqs and amps
        freqs_sorted = sorted(freq_groups.keys())
        for freq in freqs_sorted:
            freq_groups[freq] = sorted(freq_groups[freq])

        n_rows = len(freqs_sorted)
        max_cols = max(len(dbs) for dbs in freq_groups.values())

        # --- make outer figure and grid ---
        fig = plt.figure(figsize=(cell_width * max_cols, cell_height * n_rows))
        outer_grid = fig.add_gridspec(
            n_rows,
            max_cols,
            wspace=wspace,
            hspace=hspace,
        )

        # --- fill each cell ---
        for row_idx, freq in enumerate(freqs_sorted):
            db_list = freq_groups[freq]

            for col_idx in range(max_cols):
                if col_idx < len(db_list):
                    db = db_list[col_idx]

                    # One axis per (freq, db) cell
                    ax = fig.add_subplot(outer_grid[row_idx, col_idx])

                    bootstrap_means = data[(freq, db)]["bootstrap_means"]
                    freq_vec = self.eegdataset.reconstructed_ica_fft_output[(freq, db)][
                        1
                    ]

                    doub_freq_dict = select_doub_freq_bin(
                        bootstrap_means,
                        freq_vec,
                        self.eegdataset.period_keys,
                        freq,
                        window_size=100,
                    )

                    # Reuse inner function, but in grid mode
                    plot_doub_freq_hist(
                        doub_freq_dict=doub_freq_dict,
                        subjid=subjid,
                        submetric=submetric,
                        myfreq=freq,
                        myamp=db,
                        bin_num=bin_num,
                        dataset_index=dataset_index,
                        ax=ax,  # <- important: grid mode
                        log_y=log_y,
                        save_path=None,  # outer figure handles saving
                    )

                    # Optional: cleaner titles in the grid (override if you want)
                    ax.set_title(f"{freq} Hz / {db} dB", fontsize=10)

                    # Only label axes on outer edges to avoid clutter
                    if row_idx < n_rows - 1:
                        ax.set_xlabel("")
                    if col_idx > 0:
                        ax.set_ylabel("")
                else:
                    # Empty cell: turn off axis
                    empty_ax = fig.add_subplot(outer_grid[row_idx, col_idx])
                    empty_ax.axis("off")

        # Save once for the whole grid
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(
                save_dir,
                f"{subjid}_doub_freq_hist_grid_dataset{dataset_index}.png",
            )
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

        return fig, outer_grid

    def plot_bootstrap_fft_comparison_by_frequency_rows(
        self,
        attr="bootstrap_data",
        subjid="subj",
        dataset_index=0,
        cell_width=6,
        cell_height=4,
        wspace=0.3,
        hspace=0.3,
        xlim=None,
        ylim=None,
        save_dir=None,
    ):
        """
        Big grid of bootstrap FFT comparisons.

        - Each row = frequency
        - Each column = amplitude (dB)
        - Inside each cell: 1x2 subplot (Stim OFF vs Stim ON)
        drawn by `plot_bootstrap_fft_comparison`.

        Parameters
        ----------
        attr : str
            Name of attribute on self.eegdataset containing:
                {
                (freq, db): {
                    "mean_prestim": 1D array,
                    "std_prestim": 1D array,
                    "mean_stimresp": 1D array,
                    "std_stimresp": 1D array,
                    "freq_vec": 1D array,
                },
                ...
                }
        subjid : str
        dataset_index : int
        cell_width, cell_height : float
            Size of each (freq, dB) cell in inches.
        wspace, hspace : float
            Spacing between outer grid cells.
        xlim, ylim : tuple or None
            Passed through to inner plotting function.
        save_dir : str or None
            Directory to save the big figure.

        Returns
        -------
        fig, outer_grid
        """

        data = getattr(self.eegdataset, attr)

        # --- group (freq, db) pairs by freq ---
        freq_groups = defaultdict(list)
        for freq, db in data.keys():
            freq_groups[freq].append(db)

        freqs_sorted = sorted(freq_groups.keys())
        for freq in freqs_sorted:
            freq_groups[freq] = sorted(freq_groups[freq])

        n_rows = len(freqs_sorted)
        max_cols = max(len(dbs) for dbs in freq_groups.values())

        # --- make outer figure and grid ---
        fig = plt.figure(figsize=(cell_width * max_cols, cell_height * n_rows))
        outer_grid = fig.add_gridspec(
            n_rows,
            max_cols,
            wspace=wspace,
            hspace=hspace,
        )

        # --- fill each cell ---
        for row_idx, freq in enumerate(freqs_sorted):
            db_list = freq_groups[freq]

            for col_idx in range(max_cols):
                if col_idx < len(db_list):
                    db = db_list[col_idx]

                    # Inner 1x2 grid for Stim OFF / ON
                    inner_grid = outer_grid[row_idx, col_idx].subgridspec(
                        1, 2, wspace=0.3
                    )
                    ax1 = fig.add_subplot(inner_grid[0, 0])
                    ax2 = fig.add_subplot(inner_grid[0, 1])

                    entry = data[(freq, db)]

                    plot_bootstrap_fft_comparison(
                        bootstrap_means=entry["bootstrap_means"],
                        bootstrap_stds=entry["bootstrap_std"],
                        weighted_freq_vec=self.eegdataset.reconstructed_ica_fft_output[
                            freq, db
                        ][1],
                        myfreq=freq,
                        myamp=db,
                        subjid=subjid,
                        dataset_index=dataset_index,
                        xlim=xlim,
                        ylim=ylim,
                        save_fig=False,  # grid handles saving
                        save_path=None,
                        ax1=ax1,
                        ax2=ax2,
                    )

                    # Cell-level label
                    ss = outer_grid[row_idx, col_idx]
                    pos = ss.get_position(fig)
                    fig.text(
                        x=pos.x0 + pos.width / 2.0,
                        y=pos.y1 + 0.005,
                        s=f"{freq} Hz / {db} dB",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                    # Optional: simplify inner axis labels to reduce clutter
                    if row_idx < n_rows - 1:
                        ax1.set_xlabel("")
                        ax2.set_xlabel("")
                else:
                    empty_ax = fig.add_subplot(outer_grid[row_idx, col_idx])
                    empty_ax.axis("off")

        # Save once
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(
                save_dir,
                f"{subjid}_bootstrap_fft_grid_dataset{dataset_index}.png",
            )
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

        return fig, outer_grid
