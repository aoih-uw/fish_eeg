import numpy as np
from utils import get_channels
from data import EEGDataset
from constants import sampling_frequency


class Reconstructor:
    def __init__(self, eegdataset: EEGDataset):
        self.eegdataset = eegdataset
        self.fft_data = eegdataset.ica_fft_output
        self.ica_data = eegdataset.ica_output
        self.channel_keys = get_channels(self.eegdataset)

    def select_doub_freq_bin(
        self, magnitudes, frequencies, period_keys=[], myfreq=55, window_size=100
    ):
        if len(period_keys) == 0:
            doub_freq_dict = {}

            target_freq = 2 * myfreq
            part_window = window_size / 2
            artifact_freqs = [
                myfreq,
                target_freq,
                60,
                120,
                180,
                240,
                300,
                360,
                420,
                480,
                540,
                600,
                660,
                720,
                780,
                840,
                900,
            ]
            freq_vec = frequencies[0]

            # Double frequency mask (3 Hz tolerance)
            doub_mask = np.abs(freq_vec - target_freq) <= 3
            window_mask = (freq_vec >= target_freq - part_window) & (
                freq_vec <= target_freq + part_window
            )

            for freq in artifact_freqs:
                art_mask = np.abs(freq_vec - freq) <= 3
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
                snr_tmp.append(10 * np.log10((doub_mag) / (remain_mag)))

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
                60,
                120,
                180,
                240,
                300,
                360,
                420,
                480,
                540,
                600,
                660,
                720,
                780,
                840,
                900,
            ]
            freq_vec = frequencies["prestim"]["ch1"]
            # Double frequency mask (3 Hz tolerance)
            doub_mask = np.abs(freq_vec - target_freq) <= 3
            window_mask = (freq_vec >= target_freq - part_window) & (
                freq_vec <= target_freq + part_window
            )

            for freq in artifact_freqs:
                art_mask = np.abs(freq_vec - freq) <= 3
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
                    snr_tmp.append(10 * np.log10((doub_mag) / (remain_mag)))

                # Move this inside the period loop
                doub_freq_dict[period] = {
                    "doub_freq_mag": np.hstack(doub_freq_tmp),
                    "SNR": np.hstack(snr_tmp),
                }

        return doub_freq_dict

    def create_ICA_weights(self, doub_freq_dict):
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
        reconstructed_ica_data = {}
        for coord, ffts in self.fft_data.items():
            ica_doub_freq = self.select_doub_freq_bin(
                magnitudes=ffts[0], frequencies=ffts[1], period_keys=[], myfreq=coord[0]
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
            (
                compare_denoising_mean,
                compare_denoising_std,
                fft_magnitudes_mean,
                fft_magnitudes_std,
                fft_freq_vecs,
            ) = self.compare_denoised_waveform(
                self.eegdataset.bandpass_data.item()[coord],
                recon_restruct_data,
                self.channel_keys,
                coord[0],
                coord[1],
                sampling_frequency,
            )
            reconstructed_ica_data[coord]["compare_denoising_mean"] = (
                compare_denoising_mean
            )
            reconstructed_ica_data[coord]["compare_denoising_std"] = (
                compare_denoising_std
            )
            reconstructed_ica_data[coord]["fft_magnitudes_mean"] = fft_magnitudes_mean
            reconstructed_ica_data[coord]["fft_magnitudes_std"] = fft_magnitudes_std
            reconstructed_ica_data[coord]["fft_freq_vecs"] = fft_freq_vecs

        self.eegdataset.reconstructed_ica_data = reconstructed_ica_data
        return self.eegdataset
