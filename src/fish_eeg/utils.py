import numpy as np
from fish_eeg.data import EEGDataset


def get_channels(eegdataset: EEGDataset) -> list[str]:
    """
    Get the list of channels for a given data.

    Args:
        eegdataset: The eegdataset to get the channels from.
    Returns:
        A list of channels.

    Example:
        get_channels(eegdataset) -> ["ch1", "ch2", "ch3", "ch4"]
    """
    unique_channels = set()
    for coord, dictionary in eegdataset.data.item().items():
        for key in list(dictionary.keys()):
            if key.startswith("ch") and key[-1].isdigit():
                unique_channels.add(key)
    return list(unique_channels)


def separate_periods(eegdataset, data_attr: str = "reconstructed_ica_data"):
    try:
        data = getattr(eegdataset, data_attr)  # <–– dynamically fetch attribute
    except AttributeError:
        raise ValueError(f"Invalid data attribute: {data_attr}")

    data = eegdataset.reconstructed_ica_data
    separated_data = {}

    period_len = eegdataset.period_len
    latency = eegdataset.latency

    for coord in eegdataset.reconstructed_ica_data.keys():
        separated_dict = {"prestim": {}, "stimresp": {}}
        for period in eegdataset.period_keys:
            for channel in eegdataset.channel_keys:
                if period == "prestim":
                    separated_dict[period][channel] = data[coord][channel][
                        :, latency : latency + period_len
                    ]
                elif period == "stimresp":
                    separated_dict[period][channel] = data[coord][channel][
                        :, latency + period_len : latency + period_len * 2
                    ]
        separated_data[coord] = separated_dict

    getattr(eegdataset, data_attr)["separated_by_period"] = separated_data

    return eegdataset


def collapse_channels(eegdataset, attr: str = "reconstructed_ica_fft_output"):
    data = getattr(eegdataset, attr)

    for coord in data.keys():
        collapsed_dict = {"prestim": None, "stimresp": None}
        for period in eegdataset.period_keys:
            tmp = []
            for channel in eegdataset.channel_keys:
                tmp.append(data[coord][0][period][channel])
            collapsed_dict[period] = np.vstack(tmp)
        getattr(eegdataset, attr)[coord][0]["collapsed_channels"] = collapsed_dict

    return eegdataset
