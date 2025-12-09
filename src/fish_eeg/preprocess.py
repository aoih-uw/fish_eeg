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
        Filters out trials from each channel using median absolute deviation (MAD) since EEG is not normally distributed.

        Args:
            self.data: The data to filter high RMS trials from.
        Returns:
            self.eegdataset.filtered_data: The filtered data.
        Example:
            FilterHighRMSTrials() -> self.eegdataset.filtered_data
        """

        filtered_dict = {}
        keep_rows_list = []
        for channel in self.channels:
            rms_per_row = np.sqrt(
                np.mean(dictionary[channel] ** 2, axis=1)
            ) 
             #Jeffrey: Since EEG data is not normally distributed, using Z-score is not a good approach to find outliers
            #Instead, I suggest using MAD/modified z-scores. my implementation is as follows:

            median = np.median(rms_per_row)
            mad = np.median(np.abs(rms_per_row - median))

            # MAD checks for outliers, of course since we divide by mad we need to account for if MAD = 0
            if mad < 1e-6: #Protect for numerical stability, we conside values below 1e-6 to be small enough to have no outliers
                median_rms = np.median(rms_per_row)
                max_rms = np.max(rms_per_row)

                # Remove only if the max is clearly abnormal, (5 times the median)
                if max_rms > median_rms * 5:   # 3 is a common threshold, We use 5 to be less strict
                    keep_rows = np.ones_like(rms_per_row, dtype=bool)
                    idx = np.argmax(rms_per_row)   # index of max RMS
                    keep_rows[idx] = False
                else:
                    keep_rows = np.ones_like(rms_per_row, dtype=bool)
            else:
                modified_z = 0.6745 * (rms_per_row - median) / mad #Formula for MAD, choose 0.6745 to skew distribution away from normal to match theoretical EEG data
                keep_rows = np.abs(modified_z) <= 3.5  #Jeffrey: 3.5 is the standard threshold value

            keep_rows_list.append(keep_rows) #Keep a record of what trials to keep and ditch

        keep_rows = keep_rows_list[0]

        for array in keep_rows_list[1:]:
            keep_rows = keep_rows & array 
            #keep_rows and array consist of True and False values
            #This allows us to run & to get an array for all indexes that never were flagged
            
        for channel in self.channels: #keep only the trials that meet the threshold
            filtered_dict[channel] = dictionary[channel][keep_rows]
            filtered_dict[f"{channel}_total_trials"] = int(keep_rows.sum())


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
