import numpy as np
from utils import get_channels


class Preprocessor:
    def __init__(self, data: np.ndarray, seed: int = 42):
        self.data = data
        self.seed = seed
        self.channels = get_channels(data)

    def FilterHighRMSTrials(self) -> np.ndarray:
        """

        Filter high RMS trials from the data.
        Calculates RMS for each trial for each channel in a freq/amp pair stimulus.
        Filters out trials with RMS greater than 3 standard deviations above the mean RMS for that channel.

        Args:
            data: The data to filter high RMS trials from.
        Returns:
            The filtered data.
        Example:
            FilterHighRMSTrials(data) -> filtered_data
        """

        filtered_data = {}
        for coord, dictionary in self.data.item().items():
            filtered_dict = {}
            for channel in self.channels:
                rms_per_row = np.sqrt(
                    np.mean(dictionary[channel] ** 2, axis=1)
                )  # Get RMS of each row, axis = 1 means collapse rows
                rms_mean = np.mean(rms_per_row)
                rms_std = np.std(rms_per_row)
                ### Question from Yash: Is this threshold standard? Or customized to this project?
                threshold = rms_mean + (rms_std * 3)
                keep_rows = rms_per_row <= threshold
                filtered_dict[channel] = dictionary[channel][keep_rows]
                filtered_dict[f"{channel}_total_trials"] = sum(keep_rows)
            filtered_data[coord] = filtered_dict
        ### later on can use self.filtered_data for different types of artefact removal for example, other than RMS perhaps?
        self.filtered_data = filtered_data
        return filtered_data

    def SubsampleTrialsPerChannel(self, seed: int) -> np.ndarray:
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
        subsampled_data = {}

        def get_min_trials(dictionary: dict) -> int:
            ### small helper function to get the minimum number of trials of all 4 channels
            return min(
                [dictionary[f"{channel}_total_trials"] for channel in self.channels]
            )

        for coord, dictionary in self.filtered_data.item().items():
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
            subsampled_data[coord] = subsampled_dict
        self.subsampled_data = subsampled_data

        return subsampled_data

    def RemoveArtefacts(self) -> np.ndarray:
        """
        Pipeline for filtering + subsampling
        """

        _ = self.FilterHighRMSTrials()
        subsampled_data = self.SubsampleTrialsPerChannel(seed=self.seed)
        return subsampled_data
