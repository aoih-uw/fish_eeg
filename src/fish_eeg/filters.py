from fish_eeg.data import EEGDataset
from fish_eeg.utils import get_channels
from scipy.signal import butter, filtfilt
import numpy as np


class Filter:
    def __init__(self, eegdataset: EEGDataset):
        self.eegdataset = eegdataset
        self.data = self.eegdataset.rms_subsampled_data
        self.channels = get_channels(eegdataset)

    def bandpass(self, dictionary: dict, low, high, fs, order=4):
        ny = 0.5 * fs
        b, a = butter(order, [low / ny, high / ny], btype="band")
        bandpass_dict = {}
        for key in dictionary.keys():
            if key in self.channels:
                out = filtfilt(b, a, dictionary[key], axis=1)
                bandpass_dict[key] = out
            else:
                bandpass_dict[key] = dictionary[key]
        return bandpass_dict

    def pipeline(self, low, high, fs, order=4):
        bandpass_data = {}
        for coord, dictionary in self.data.item().items():
            bandpass_data[coord] = self.bandpass(dictionary, low, high, fs, order)
        self.eegdataset.bandpass_data = np.array(bandpass_data, dtype=object)
        return self.eegdataset
