import numpy as np
from data import EEGDataset


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
