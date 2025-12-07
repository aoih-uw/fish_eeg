from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


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

        if save_path is not None:
            plt.gcf().savefig(save_path, dpi=300, bbox_inches="tight")

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
    ax=None,
):
    """
    Clean FFT plotting function with correct standalone vs embedded behavior.
    """

    # ---------------------------
    # 0. Embedded mode detection
    # ---------------------------
    standalone = ax is None

    if period_keys is None:
        period_keys = []

    # ---------------------------
    # 1. Embedded mode = draw on provided axis and RETURN EARLY
    # ---------------------------
    if not standalone:
        channel = channel_keys[0]  # wrapper ensures a single channel is passed

        cur_mag = magnitudes[channel]
        cur_freq = frequencies[channel]

        random_indices = np.random.choice(
            cur_mag.shape[0],
            size=min(num_samples, cur_mag.shape[0]),
            replace=False,
        )

        for i in random_indices:
            ax.plot(cur_freq[i, :], cur_mag[i, :], alpha=alpha)

        # Formatting
        ax.axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.6)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(channel)

        return ax  # ←–––––––––––––––––––––––––––––– EARLY RETURN prevents extra empty figure

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

        axes[-1].axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.6)

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

        axes[-1].axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.6)

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

        axes[-1, -1].axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.6)

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


def save_bootstrap_plots_to_pdf(
    bootstrap_means,
    bootstrap_stds,
    weighted_freq_vec,
    myfreq,
    myamp,
    subjid,
    dataset_index,
    pdf_path="bootstrap_plots.pdf",
    indices=None,
    xlim=None,
    ylim=None,
):
    """
    Generates a multipage PDF with each page showing the 2-panel FFT plot
    (Stim OFF + Stim ON) for each bootstrap sample.

    Requires that plot_bootstrap_fft_comparison(...) returns a figure object.
    """

    def plot_bootstrap_fft_comparison(
        mean_fft_prestim,
        std_fft_prestim,
        mean_fft_stimresp,
        std_fft_stimresp,
        freq_vec,
        myfreq,
        myamp,
        subjid,
        dataset_index,
        xlim=None,
        ylim=None,
        save_fig=False,
        save_path=None,
    ):
        # Create figure with shared axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

        # Plot prestim FFT
        ax1.plot(freq_vec, mean_fft_prestim, "-", color="#5778a4", label="Mean FFT")
        ax1.fill_between(
            freq_vec,
            mean_fft_prestim - std_fft_prestim,
            mean_fft_prestim + std_fft_prestim,
            alpha=0.3,
            color="#5778a4",
            label="±1 STD",
        )

        # Add vertical line at 2nd harmonic
        ax1.axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.5, alpha=0.7)

        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude")
        ax1.set_title("Stim OFF")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot stimresp FFT
        ax2.plot(freq_vec, mean_fft_stimresp, "-", color="#e49444", label="Mean FFT")
        ax2.fill_between(
            freq_vec,
            mean_fft_stimresp - std_fft_stimresp,
            mean_fft_stimresp + std_fft_stimresp,
            alpha=0.3,
            color="#e49444",
            label="±1 STD",
        )

        # Add vertical line at 2nd harmonic
        ax2.axvline(myfreq * 2, color="red", linestyle="--", linewidth=0.5, alpha=0.7)

        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_title("Stim ON")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Set x limits if provided
        if xlim is not None:
            ax1.set_xlim(xlim)

        if ylim is not None:
            ax1.set_ylim(ylim)

        plt.tight_layout()

        # Save figure if requested
        if save_fig:
            filename = f"plots/{subjid}/{subjid}_{myfreq}Hz_{myamp}dB_grand_avg_comparison_{dataset_index}.png"
            if save_path:
                import os

                filename = os.path.join(save_path, filename)
            plt.savefig(filename, dpi=600, bbox_inches="tight")

        # plt.show()

        return fig

    # check 11/20/25

    # If no subset of indices provided, use all bootstrap samples
    if indices is None:
        indices = range(weighted_freq_vec["prestim"]["ch1"].shape[0])
    else:
        indices = range(indices)

    with PdfPages(pdf_path) as pdf:
        for idx in indices:
            # Extract bootstrap data for this index
            mean_fft_prestim = bootstrap_means["prestim"][idx]
            std_fft_prestim = bootstrap_stds["prestim"][idx]

            mean_fft_stimresp = bootstrap_means["stimresp"][idx]
            std_fft_stimresp = bootstrap_stds["stimresp"][idx]

            freq_vec = weighted_freq_vec["prestim"]["ch1"][idx]

            # Call YOUR existing plotting function, but do NOT save PNGs
            fig = plot_bootstrap_fft_comparison(
                mean_fft_prestim,
                std_fft_prestim,
                mean_fft_stimresp,
                std_fft_stimresp,
                freq_vec,
                myfreq,
                myamp,
                subjid,
                dataset_index,
                xlim=xlim,
                ylim=ylim,
                save_fig=False,
                save_path=None,
            )

            # Add a figure title at the top of the page
            fig.suptitle(f"Bootstrap Sample {idx}", fontsize=14)

            # Add this figure as a new page in the PDF
            pdf.savefig(fig)

            # Close the figure (prevents memory bloat)
            plt.close(fig)

    print(f"\nSaved multipage PDF with {len(indices)} pages to:\n  {pdf_path}\n")


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
        cell_width=4,
        cell_height=3,
        channel_keys=None,
        period_keys=None,
        save_path=None,
    ):
        data = getattr(self.eegdataset, attr)
        if isinstance(data, dict):
            pass
        else:
            data = data.item()

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

        fig = plt.figure(figsize=(cell_width * max_cols, cell_height * n_rows))
        outer_grid = fig.add_gridspec(n_rows, max_cols)

        for row_idx, freq in enumerate(freqs_sorted):
            row_dbs = freq_groups[freq]

            for col_idx in range(max_cols):
                outer_ax = fig.add_subplot(outer_grid[row_idx, col_idx])

                if col_idx < len(row_dbs):
                    db, array = row_dbs[col_idx]

                    # --- nested subgrid for individual channel plots ---
                    inner_grid = outer_grid[row_idx, col_idx].subgridspec(
                        len(channel_keys), 1
                    )

                    plot_waveforms(
                        data=array,
                        channel_keys=channel_keys,
                        num_samples=num_samples,
                        alpha=alpha,
                        title=f"{freq} Hz, {db} dB",
                        parent_grid=inner_grid,
                    )

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

        # ----------------------------------------------------
        # Get data from attribute
        # Expected structure: {(freq, amp): (magnitudes, frequencies)}
        # ----------------------------------------------------
        data = getattr(self.eegdataset, attr)
        items = list(data.items())

        # ----------------------------------------------------
        # Group by frequency, just like plot_waveforms_by_frequency_rows
        # freq_groups[freq] = [(amp, (magnitudes, frequencies)), ...]
        # ----------------------------------------------------
        freq_groups = defaultdict(list)
        for (freq, amp), fftout in items:
            freq_groups[freq].append((amp, fftout))

        freqs_sorted = sorted(freq_groups.keys())

        for freq in freqs_sorted:
            # sort by amp/dB
            freq_groups[freq].sort(key=lambda x: x[0])

        n_rows = len(freqs_sorted)
        max_cols = max(len(v) for v in freq_groups.values())

        # ----------------------------------------------------
        # Create grid figure
        # ----------------------------------------------------
        fig = plt.figure(figsize=(cell_width * max_cols, cell_height * n_rows))
        outer_grid = fig.add_gridspec(n_rows, max_cols)

        # ----------------------------------------------------
        # Fill grid
        # ----------------------------------------------------
        for row_idx, freq in enumerate(freqs_sorted):
            row_dbs = freq_groups[freq]  # list of (amp, fftout)

            for col_idx in range(max_cols):
                ax = fig.add_subplot(outer_grid[row_idx, col_idx])

                if col_idx < len(row_dbs):
                    amp, fftout = row_dbs[col_idx]
                    magnitudes = fftout[0]
                    frequencies = fftout[1]

                    # -----------------------------------------
                    # Call your original plot_fft completely unchanged
                    # -----------------------------------------
                    plot_fft(
                        magnitudes=magnitudes,
                        frequencies=frequencies,
                        myfreq=freq,
                        myamp=amp,
                        subjid=subjid,
                        dataset_index=dataset_index,
                        channel_keys=channel_keys,
                        period_keys=period_keys,  # keeps your CASE 2 behavior
                        title=f"FFT {freq} Hz, {amp} dB",
                        num_samples=num_samples,
                        alpha=alpha,
                        xlim=[freq * 2 - 100, freq * 2 + 100],
                        ylim=[0, 0.002],
                        ax=ax,  # ← render inside grid cell
                    )

                    ax.set_title(f"{freq} Hz | {amp} dB")

                else:
                    ax.axis("off")

        fig.tight_layout()

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                save_path + "/" + subjid + "_" + attr + "_.png",
                format="png",
                dpi=600,
                bbox_inches="tight",
            )

        return None

    def save_bootstrap_plots_to_pdf(
        self,
        bootstrap_means,
        bootstrap_stds,
        weighted_freq_vec,
        myfreq,
        myamp,
        subjid,
        dataset_index,
        pdf_path="bootstrap_plots.pdf",
        indices=None,
        xlim=None,
        ylim=None,
    ):
        return save_bootstrap_plots_to_pdf(
            bootstrap_means,
            bootstrap_stds,
            weighted_freq_vec,
            myfreq,
            myamp,
            subjid,
            dataset_index,
            pdf_path,
            indices,
            xlim,
            ylim,
        )
