from fish_eeg.data import EEGDataset
from fish_eeg.utils import get_channels
from scipy.signal import butter, filtfilt
from fish_eeg.utils import dotdict


class Filter:
    def __init__(self, eegdataset: EEGDataset, cfg: dict | None = None):
        self.eegdataset = eegdataset
        self.data = self.eegdataset.rms_subsampled_data
        self.channels = get_channels(eegdataset)
        cfg = cfg or dotdict({})  # if None, use empty
        if not isinstance(cfg, dotdict):
            cfg = dotdict(cfg)
        filter_cfg = cfg.get("filters", dotdict({}))
        self.method = filter_cfg.get("method", "bandpass")
        self.cfg = filter_cfg.get("params", dotdict({}))

    def bandpass(
        self, dictionary: dict, low=70, high=1400, fs=22050, order=4, ny_fs_ratio=0.5
    ):
        low = self.cfg.get("low", low)
        high = self.cfg.get("high", high)
        fs = self.cfg.get("sampling_frequency", fs)
        order = self.cfg.get("order", order)
        ny_fs_ratio = self.cfg.get("ny_fs_ratio", ny_fs_ratio)

        ny = ny_fs_ratio * fs
        b, a = butter(order, [low / ny, high / ny], btype="band")
        bandpass_dict = {}
        for key in dictionary.keys():
            if key in self.channels:
                out = filtfilt(b, a, dictionary[key], axis=1)
                bandpass_dict[key] = out
            else:
                bandpass_dict[key] = dictionary[key]
        return bandpass_dict

    def pipeline(self):
        method = self.method
        if method == "bandpass":
            bandpass_data = {}
            for coord, dictionary in self.data.items():
                bandpass_data[coord] = self.bandpass(dictionary)
            self.eegdataset.bandpass_data = bandpass_data
            return self.eegdataset
        else:
            raise ValueError(f"Unknown filter method: {method!r}. Must be 'bandpass'.")
