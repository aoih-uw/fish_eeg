from fish_eeg.data import EEGDataset
from fish_eeg.utils import get_channels
import numpy as np


class FFT:
    def __init__(self, eegdataset: EEGDataset):
        self.eegdataset = eegdataset
        self.data = self.eegdataset.ica_output
        self.channels = get_channels(eegdataset)
        try:
            self.data = self.eegdataset.reconstructed_ica_data["separated_by_period"]
        except:
            pass

    def compute_fft(
        self, data, sampling_frequency, period_keys=[], channel_keys=[], smallest_dim=1
    ):
        if len(period_keys) == 0 and len(channel_keys) == 0:
            fft_magnitudes = {}
            fft_freq_vecs = {}

            trial_magnitudes = []
            freq_vecs = []

            if smallest_dim == 1:
                num_cols = data.shape[smallest_dim]
                for trial in range(num_cols):
                    cur_trial = data[:, trial]
                    n_samples = len(cur_trial)
                    # Calculate frequencies for positive frequencies only
                    freqs = np.fft.rfftfreq(
                        sampling_frequency, d=1 / sampling_frequency
                    )  # fs= window length, d = sampling period

                    # Calculate magnitude spectrum
                    fft_vals = np.fft.rfft(
                        cur_trial, n=sampling_frequency
                    )  # cur_trial= data to do, fs = zero pad up to the fs
                    magnitude = (
                        np.abs(fft_vals) / n_samples
                    )  # divide by the number of samples to recover magntiude values
                    trial_magnitudes.append(magnitude)
                    freq_vecs.append(freqs)

                    fft_magnitudes = np.vstack(trial_magnitudes)
                    fft_freq_vecs = np.vstack(freq_vecs)

        # Only have channels
        elif len(period_keys) == 0 and len(channel_keys) > 0:
            # Initialize dictionaries with nested structure
            fft_magnitudes = {}
            fft_freq_vecs = {}

            for channel in channel_keys:
                trial_magnitudes = []
                freq_vecs = []
                num_trial = min(data[channel].shape)
                for trial in range(num_trial):
                    cur_trial = data[channel][trial]
                    n_samples = len(cur_trial)
                    # Calculate frequencies for positive frequencies only
                    freqs = np.fft.rfftfreq(
                        sampling_frequency, d=1 / sampling_frequency
                    )

                    # Calculate magnitude spectrum
                    fft_vals = np.fft.rfft(cur_trial, n=sampling_frequency)
                    magnitude = np.abs(fft_vals) / n_samples

                    trial_magnitudes.append(magnitude)
                    freq_vecs.append(freqs)

                fft_magnitudes[channel] = np.vstack(trial_magnitudes)
                fft_freq_vecs[channel] = np.vstack(freq_vecs)

        # Have both period and channel delination
        elif len(period_keys) > 0 and len(channel_keys) > 0:
            # Initialize dictionaries with nested structure
            fft_magnitudes = {period: {} for period in period_keys}
            fft_freq_vecs = {period: {} for period in period_keys}

            for period in period_keys:
                for channel in channel_keys:
                    trial_magnitudes = []
                    freq_vecs = []
                    num_trial = data[period][channel].shape[0]

                    for trial in range(num_trial):
                        cur_trial = data[period][channel][trial]
                        n_samples = len(cur_trial)
                        # Calculate frequencies for positive frequencies only
                        freqs = np.fft.rfftfreq(
                            sampling_frequency, d=1 / sampling_frequency
                        )

                        # Calculate magnitude spectrum
                        fft_vals = np.fft.rfft(cur_trial, n=sampling_frequency)
                        magnitude = np.abs(fft_vals) / n_samples

                        trial_magnitudes.append(magnitude)
                        freq_vecs.append(freqs)

                    fft_magnitudes[period][channel] = np.vstack(trial_magnitudes)
                    fft_freq_vecs[period][channel] = np.vstack(freq_vecs)

        # Return both dictionaries
        return fft_magnitudes, fft_freq_vecs

    def pipeline(self, sampling_frequency):
        fft_out = {}
        for coord, array in self.data.items():
            if self.data != self.eegdataset.ica_output:
                out = self.compute_fft(
                    array,
                    sampling_frequency,
                    period_keys=self.eegdataset.period_keys,
                    channel_keys=self.eegdataset.channel_keys,
                    smallest_dim=[],
                )
                fft_out[coord] = out
            else:
                out = self.compute_fft(
                    array["S"],
                    sampling_frequency,
                    period_keys=[],
                    channel_keys=[],
                    smallest_dim=1,
                )
                fft_out[coord] = out

        if self.data != self.eegdataset.ica_output:
            self.eegdataset.reconstructed_ica_fft_output = fft_out
        else:
            self.eegdataset.ica_fft_output = fft_out
        return self.eegdataset
