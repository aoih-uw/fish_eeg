import numpy as np
from fish_eeg.utils import get_channels
from fish_eeg.data import EEGDataset, ConfigAccessor
from fish_eeg.constants import sampling_frequency
from fish_eeg.constants import baseline_artifact_freqs


class Reconstructor:
    def __init__(self, eegdataset: EEGDataset, cfg: ConfigAccessor | None = None):
        """
        Initialize the Reconstructor with EEG dataset and configuration.

        Parameters
        ----------
        eegdataset : EEGDataset
            The EEG dataset containing ICA and FFT outputs to be reconstructed.
        cfg : ConfigAccessor | None, optional
            Configuration accessor for reconstruction parameters. If None, uses default
            configuration with method='ICA'.

        Attributes
        ----------
        fft_data : dict
            FFT output data from the EEG dataset.
        ica_data : dict
            ICA output data from the EEG dataset.
        channel_keys : list
            List of channel identifiers.
        method : str
            Reconstruction method to use (default: 'ICA').
        """
        self.eegdataset = eegdataset
        self.fft_data = eegdataset.ica_fft_output
        self.ica_data = eegdataset.ica_output
        self.channel_keys = get_channels(self.eegdataset)
        cfg = cfg or ConfigAccessor(None)
        self.method = cfg.get("reconstruct", "method", default="ICA")
        self.cfg = cfg.get("reconstruct", "params", default=ConfigAccessor(None))

    def select_doub_freq_bin(
        self,
        magnitudes,
        frequencies,
        period_keys=[],
        myfreq=55,
        window_size=100,
        double_freq_mask_tolerance=3,
        artifact_freqs=baseline_artifact_freqs,
        art_mask_tolerance=3,
        snr_scaler=10,
    ):
        """
        Calculate double frequency magnitude and SNR for ICA components.

        Identifies the signal magnitude at double the stimulation frequency and computes
        the signal-to-noise ratio by comparing it to surrounding frequency bins.

        Parameters
        ----------
        magnitudes : np.ndarray or dict
            Magnitude data from FFT analysis. Can be array or dict with period keys.
        frequencies : np.ndarray or dict
            Frequency vectors corresponding to magnitudes.
        period_keys : list, optional
            List of period identifiers (e.g., ['prestim', 'stimresp']). Empty list
            processes single period (default: []).
        myfreq : float, optional
            Stimulation frequency in Hz (default: 55).
        window_size : float, optional
            Frequency window size around target frequency in Hz (default: 100).
        double_freq_mask_tolerance : float, optional
            Tolerance in Hz for double frequency mask (default: 3).
        artifact_freqs : list, optional
            List of artifact frequencies to exclude (default: baseline_artifact_freqs).
        art_mask_tolerance : float, optional
            Tolerance in Hz for artifact frequency masking (default: 3).
        snr_scaler : float, optional
            Scaling factor for SNR calculation in dB (default: 10).

        Returns
        -------
        dict
            If period_keys is empty: Dictionary with 'doub_freq_mag' and 'SNR' arrays.
            If period_keys provided: Nested dictionary with period keys containing
            'doub_freq_mag' and 'SNR' for each period.
        """
        if len(period_keys) == 0:
            doub_freq_dict = {}

            target_freq = 2 * myfreq
            part_window = window_size / 2
            artifact_freqs = [
                myfreq,
                target_freq,
                *artifact_freqs,
            ]
            freq_vec = frequencies[0]

            # Double frequency mask (3 Hz tolerance)
            doub_mask = np.abs(freq_vec - target_freq) <= self.cfg.get(
                "double_freq_mask_tolerance", double_freq_mask_tolerance
            )
            window_mask = (freq_vec >= target_freq - part_window) & (
                freq_vec <= target_freq + part_window
            )

            for freq in artifact_freqs:
                art_mask = np.abs(freq_vec - freq) <= self.cfg.get(
                    "art_mask_tolerance", art_mask_tolerance
                )
                window_mask[art_mask] = False

            num_it = magnitudes.shape[0]

            doub_freq_tmp = []
            snr_tmp = []

            for cur_it in range(num_it):
                cur_data = magnitudes[cur_it]
                doub_mag = cur_data[doub_mask]
                remain_mag = cur_data[window_mask]

                doub_mag = np.mean(doub_mag)
                remain_mag = np.mean(remain_mag)

                doub_freq_tmp.append(doub_mag)
                snr_tmp.append(
                    self.cfg.get("snr_scaler", snr_scaler)
                    * np.log10((doub_mag) / (remain_mag))
                )

            # Move this inside the period loop
            doub_freq_dict = {
                "doub_freq_mag": np.hstack(doub_freq_tmp),
                "SNR": np.hstack(snr_tmp),
            }

        elif len(period_keys) > 0:
            doub_freq_dict = {"prestim": {}, "stimresp": {}}

            target_freq = 2 * myfreq
            part_window = window_size / 2
            artifact_freqs = [
                myfreq,
                target_freq,
                *artifact_freqs,
            ]
            freq_vec = frequencies["prestim"]["ch1"]
            # Double frequency mask (3 Hz tolerance)
            doub_mask = np.abs(freq_vec - target_freq) <= self.cfg.get(
                "double_freq_mask_tolerance", double_freq_mask_tolerance
            )
            window_mask = (freq_vec >= target_freq - part_window) & (
                freq_vec <= target_freq + part_window
            )

            for freq in artifact_freqs:
                art_mask = np.abs(freq_vec - freq) <= self.cfg.get(
                    "art_mask_tolerance", art_mask_tolerance
                )
                window_mask[art_mask] = False

            num_it = len(magnitudes["prestim"])
            doub_mask = doub_mask[
                0
            ]  # I am doing this since freq_vec is a matrix? check this...
            window_mask = window_mask[0]

            for period in period_keys:
                doub_freq_tmp = []
                snr_tmp = []
                for cur_it in range(num_it):
                    cur_data = magnitudes[period][cur_it]
                    doub_mag = cur_data[doub_mask]
                    remain_mag = cur_data[window_mask]

                    doub_mag = np.mean(doub_mag)
                    remain_mag = np.mean(remain_mag)

                    doub_freq_tmp.append(doub_mag)
                    snr_tmp.append(
                        self.cfg.get("snr_scaler", snr_scaler)
                        * np.log10((doub_mag) / (remain_mag))
                    )

                # Move this inside the period loop
                doub_freq_dict[period] = {
                    "doub_freq_mag": np.hstack(doub_freq_tmp),
                    "SNR": np.hstack(snr_tmp),
                }

        return doub_freq_dict

    def create_ICA_weights(self, doub_freq_dict):
        """
        Convert SNR values to normalized weights for ICA component weighting.

        Transforms SNR from dB scale to linear scale and normalizes to sum to 1.

        Parameters
        ----------
        doub_freq_dict : dict
            Dictionary containing 'SNR' key with SNR values in dB.

        Returns
        -------
        np.ndarray
            Normalized weights that sum to 1, with higher weights for components
            with higher SNR.
        """
        weights = []
        snr_db = doub_freq_dict["SNR"]
        linear_snr = 10 ** (snr_db / 10)  # take the inverse log to calculate linear_snr
        weights = linear_snr / linear_snr.sum()  # normalize by the sum of the snrs
        return weights

    def reconstruct_ICA(
        self,
        ica_results,
        current_cond,
        channel_keys,
        components_to_keep,
        component_weights=None,
    ):
        """
        Reconstruct EEG signals from selected ICA components with optional weighting.

        Performs weighted reconstruction by filtering ICA components and combining
        them back into channel space.

        Parameters
        ----------
        ica_results : dict
            Dictionary containing ICA decomposition with keys:
            - 'S': Independent component time courses (samples x components)
            - 'A': Mixing matrix (channels x components)
        current_cond : dict
            Current condition data with channel keys, used to determine trial structure.
        channel_keys : list
            List of channel identifiers for output organization.
        components_to_keep : list
            Indices of ICA components to include in reconstruction.
        component_weights : np.ndarray | None, optional
            Weights for each component. If None, uses equal weighting (default: None).

        Returns
        -------
        dict
            Dictionary mapping channel keys to reconstructed data arrays with shape
            (n_trials, n_samples_per_trial).

        Raises
        ------
        AssertionError
            If number of weights doesn't match number of components to keep.
        """
        recon_restruct_data = {}

        # Extract ICA results
        S = ica_results["S"]
        A = ica_results["A"]

        # Create a weighted reconstruction
        if component_weights is None:
            component_weights = np.ones(len(components_to_keep))

        assert len(component_weights) == len(components_to_keep), (
            "Number of weights must match number of components"
        )

        # Filter independent components
        S_filtered = S[:, components_to_keep]
        A_filtered = A[:, components_to_keep]

        S_weighted = S_filtered * component_weights

        reconstructed_data = np.dot(
            S_weighted, A_filtered.T
        )  # Matrix multiplication of the two matrices to get signals back

        n_trials = current_cond["ch1"].shape[0]
        n_samples_per_trial = current_cond["ch1"].shape[1]

        for idx, channel in enumerate(channel_keys):
            recon_restruct_data[channel] = reconstructed_data[:, idx].reshape(
                n_trials, n_samples_per_trial
            )

        return recon_restruct_data

    def compare_denoised_waveform(
        self, filt_data, recon_restruct_data, channel_keys, myfreq, myamp, fs
    ):
        """
        Compare original and denoised signals in time and frequency domains.

        Computes means, standard deviations, and FFT analysis for both original
        filtered data and reconstructed denoised data.

        Parameters
        ----------
        filt_data : dict
            Original filtered data with channel keys mapping to trial arrays.
        recon_restruct_data : dict
            Reconstructed denoised data with channel keys mapping to trial arrays.
        channel_keys : list
            List of channel identifiers.
        myfreq : float
            Stimulation frequency in Hz.
        myamp : float
            Stimulation amplitude (not used in current implementation).
        fs : int
            Sampling frequency in Hz.

        Returns
        -------
        tuple
            Five-element tuple containing:
            - compare_denoising_mean : dict
                Mean waveforms for 'original' and 'denoised' by channel.
            - compare_denoising_std : dict
                Standard deviation waveforms for 'original' and 'denoised' by channel.
            - fft_magnitudes_mean : dict
                FFT magnitudes of mean waveforms by data type and channel.
            - fft_magnitudes_std : dict
                FFT magnitudes of std waveforms by data type and channel.
            - fft_freq_vecs : dict
                Frequency vectors for FFT analysis by data type and channel.
        """
        data_type = ["original", "denoised"]
        # Calculate grand mean
        compare_denoising_mean = {
            "original": {
                channel: np.mean(filt_data[channel], axis=0) for channel in channel_keys
            },
            "denoised": {
                channel: np.mean(recon_restruct_data[channel], axis=0)
                for channel in channel_keys
            },
        }

        compare_denoising_std = {
            "original": {
                channel: np.std(filt_data[channel], axis=0) for channel in channel_keys
            },
            "denoised": {
                channel: np.std(recon_restruct_data[channel], axis=0)
                for channel in channel_keys
            },
        }

        # Calculate ffts
        fft_magnitudes_mean = {"original": {}, "denoised": {}}

        fft_magnitudes_std = {"original": {}, "denoised": {}}

        fft_freq_vecs = {"original": {}, "denoised": {}}

        for dtype in data_type:
            for channel in channel_keys:
                cur_mag_mean = compare_denoising_mean[dtype][channel]
                cur_mag_std = compare_denoising_std[dtype][channel]
                n_samples = len(cur_mag_mean)

                # Frequency vector
                freqs = np.fft.rfftfreq(fs, d=1 / fs)
                fft_freq_vecs[dtype][channel] = freqs

                # Mean
                fft_vals = np.fft.rfft(cur_mag_mean, n=fs)
                magnitude = np.abs(fft_vals) / n_samples
                fft_magnitudes_mean[dtype][channel] = magnitude

                # STD
                fft_vals = np.fft.rfft(cur_mag_std, n=fs)
                magnitude = np.abs(fft_vals) / n_samples
                fft_magnitudes_std[dtype][channel] = magnitude

        return (
            compare_denoising_mean,
            compare_denoising_std,
            fft_magnitudes_mean,
            fft_magnitudes_std,
            fft_freq_vecs,
        )

    def pipeline(self, weighted: bool = False):
        """
        Execute the full reconstruction pipeline for all conditions in the dataset.

        Processes each coordinate (frequency, amplitude pair) by computing SNR-based
        weights, reconstructing signals from ICA components, and comparing original
        vs denoised waveforms.

        Parameters
        ----------
        weighted : bool, optional
            If True, uses SNR-based weighting for component reconstruction.
            If False, uses equal weighting (default: False).

        Returns
        -------
        EEGDataset
            The input dataset with added 'reconstructed_ica_data' attribute containing
            reconstruction results for all coordinates.

        Raises
        ------
        ValueError
            If reconstruction method is not 'ICA'.
        """
        reconstructed_ica_data = {}
        method = self.method
        for coord, ffts in self.fft_data.items():
            if method == "ICA":
                ica_doub_freq = self.select_doub_freq_bin(
                    magnitudes=ffts[0],
                    frequencies=ffts[1],
                    period_keys=[],
                    myfreq=coord[0],
                )
                if weighted:
                    weights = self.create_ICA_weights(ica_doub_freq)
                else:
                    weights = None

                recon_restruct_data = self.reconstruct_ICA(
                    self.eegdataset.ica_output[coord],
                    self.eegdataset.rms_subsampled_data.item()[coord],
                    self.channel_keys,
                    [0, 1, 2, 3],
                    component_weights=weights,
                )
                reconstructed_ica_data[coord] = recon_restruct_data

                comparison_results = self.compare_denoised_waveform(
                    self.eegdataset.bandpass_data.item()[coord],
                    recon_restruct_data,
                    self.channel_keys,
                    coord[0],
                    coord[1],
                    sampling_frequency,
                )
                reconstructed_ica_data[coord]["compare_denoising_mean"] = (
                    comparison_results[0]
                )
                reconstructed_ica_data[coord]["compare_denoising_std"] = (
                    comparison_results[1]
                )
                reconstructed_ica_data[coord]["fft_magnitudes_mean"] = (
                    comparison_results[2]
                )
                reconstructed_ica_data[coord]["fft_magnitudes_std"] = (
                    comparison_results[3]
                )
                reconstructed_ica_data[coord]["fft_freq_vecs"] = comparison_results[4]

            self.eegdataset.reconstructed_ica_data = reconstructed_ica_data
            return self.eegdataset
        else:
            raise ValueError(f"Unknown reconstruct method: {method!r}. Must be 'ICA'.")
