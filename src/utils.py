import numpy as np


def get_channels(data: np.ndarray) -> list[str]:
    """
    Get the list of channels for a given data.

    Args:
        data: The data to get the channels from.
    Returns:
        A list of channels.

    Example:
        get_channels(data) -> ["ch1", "ch2", "ch3", "ch4"]
    """
    unique_channels = set()
    for coord, dictionary in data.item().items():
        for key in list(dictionary.keys()):
            if key.startswith("ch"):
                unique_channels.add(key)
    return list(unique_channels)
