import numpy as np
from fish_eeg.utils import get_channels
from fish_eeg.data import EEGDataset


class Preprocessor:
    def __init__(self, eegdataset: EEGDataset, seed: int = 42):
        self.eegdataset = eegdataset
        self.data = self.eegdataset.data
        self.seed = seed
        self.channels = get_channels(eegdataset)

    def FilterHighRMSTrials(self, dictionary: dict) -> np.ndarray:
        """

        Filter high RMS trials from the data.
        Calculates RMS for each trial for each channel in a freq/amp pair stimulus.
        Filters out trials with RMS greater than 3 standard deviations above the mean RMS for that channel.

        Args:
            self.data: The data to filter high RMS trials from.
        Returns:
            self.eegdataset.filtered_data: The filtered data.
        Example:
            FilterHighRMSTrials() -> self.eegdataset.filtered_data
        """

        filtered_dict = {}
        for channel in self.channels:
            rms_per_row = np.sqrt(
                np.mean(dictionary[channel] ** 2, axis=1)
            )  # Get RMS of each row, axis = 1 means collapse rows
            rms_mean = np.mean(rms_per_row)
            rms_std = np.std(rms_per_row)
            threshold = rms_mean + (rms_std * 3)
            keep_rows = rms_per_row <= threshold
            filtered_dict[channel] = dictionary[channel][keep_rows]
            filtered_dict[f"{channel}_total_trials"] = sum(keep_rows)
        return filtered_dict

    def SubsampleTrialsPerChannel(self, dictionary: dict, seed) -> np.ndarray:
        """
        Subsample trials per channel to the minimum number of trials of all channels.
        Ensures consistent sample size across channels for each freq/amp pair stimulus.

        Args:
            seed: The seed to use for the random number generator.
        Returns:
            The subsampled data.
        Example:
            SubsampleTrialsPerChannel(seed) -> subsampled_data
        """

        ### set seed outside loop to ensure different random samples for each channel!!!
        np.random.seed(seed)

        def get_min_trials(dictionary: dict) -> int:
            ### small helper function to get the minimum number of trials of all 4 channels
            return min(
                [dictionary[f"{channel}_total_trials"] for channel in self.channels]
            )

        min_trials = get_min_trials(dictionary)
        subsampled_dict = {}
        for channel in self.channels:
            #### subsample each channel to the minimum number of trials of all 4 channels
            selected_indices = np.random.choice(
                np.arange(dictionary[channel].shape[0]),
                size=min_trials,
                replace=False,
            )
            subsampled_dict[channel] = dictionary[channel][selected_indices]
            subsampled_dict[f"{channel}_total_trials"] = min_trials
        return subsampled_dict

    def pipeline(self) -> np.ndarray:
        """
        Pipeline for removing artefacts.
        Currently filtering high RMS trials and then randomly subsamples trials
        for each channel to the minimum number of trials of all channels.
        """
        filtered_data = {}
        subsampled_data = {}
        for coord, dictionary in self.data.item().items():
            filtered_data[coord] = self.FilterHighRMSTrials(dictionary)
            subsampled_data[coord] = self.SubsampleTrialsPerChannel(
                filtered_data[coord], self.seed
            )

        filtered_data = np.array(filtered_data, dtype=object)
        subsampled_data = np.array(subsampled_data, dtype=object)
        self.eegdataset.rms_filtered_data = filtered_data
        self.eegdataset.rms_subsampled_data = subsampled_data
        return self.eegdataset
