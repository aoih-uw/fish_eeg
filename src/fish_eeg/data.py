import numpy as np
from dataclasses import dataclass
from fish_eeg.constants import PERIOD_KEYS, METRIC_KEYS, SUBMETRIC_KEYS
from typing import Optional, Any, Mapping
import yaml


#### Dataclass for EEGDataset ####
@dataclass
class EEGDataset:
    """
    Central dataset class for the fish-eeg project. Stores all initial data
    in this object and analysis of data appends intermediate data to this object.
    """

    data: np.ndarray
    freq_amp_table: np.ndarray
    latency: int
    channel_keys: list[str]
    period_keys: list[str]
    metric_keys: list[str]
    submetric_keys: list[str]
    period_len: int


#### I/O for data ####
def load_data(path: str, subjid: str) -> EEGDataset:
    """
    Loads data from the given path and returns an EEGDataset object.

    Args:
        path: The path to the data.
        subjid: The subject ID.
    Returns:
        An EEGDataset object.

    Example:
        load_data(path, subjid) -> eegdataset
    """

    loaded = np.load(f"{path}/{subjid}_data.npz", allow_pickle=True)
    data = loaded["data"]
    freq_amp_table = loaded["freq_amp_table"]
    latency = loaded["latency"].item()
    channel_keys = loaded["channel_keys"].tolist()
    period_keys = PERIOD_KEYS
    metric_keys = METRIC_KEYS
    submetric_keys = SUBMETRIC_KEYS

    return EEGDataset(
        data=data.item(),
        freq_amp_table=freq_amp_table,
        latency=latency,
        channel_keys=channel_keys,
        period_keys=period_keys,
        metric_keys=metric_keys,
        submetric_keys=submetric_keys,
        period_len=3528,  ### this should be part of npz/mtx file instead of hard coded
    )


#### Random helpers ####
def subset_stimulus(data, myfreq, myamp):
    """
    Subset the data for a given frequency and amplitude.

    Args:
        data: The data to subset.
        myfreq: The frequency to subset.
        myamp: The amplitude to subset.
    Returns:
        The subsetted data.
    Example:
        subset_stimulus(data, myfreq, myamp) -> subsetted_data
    """

    specific_key = (np.float64(myfreq), np.float64(myamp))
    current_cond = data.item()[specific_key]

    return current_cond


def separate_periods(data, period_len, period_keys, channel_keys, latency):
    separated_data = {"prestim": {}, "stimresp": {}}

    for period in period_keys:
        for channel in channel_keys:
            if period == "prestim":
                separated_data[period][channel] = data[channel][
                    :, latency : latency + period_len
                ]
            elif period == "stimresp":
                separated_data[period][channel] = data[channel][
                    :, latency + period_len : latency + period_len * 2
                ]

    return separated_data


def read_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
