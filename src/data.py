import numpy as np
from dataclasses import dataclass
from constants import PERIOD_KEYS, METRIC_KEYS, SUBMETRIC_KEYS


#### Dataclass for EEGDataset ####
@dataclass
class EEGDataset:
    data: np.ndarray
    freq_amp_table: np.ndarray
    latency: int
    channel_keys: list[str]
    period_keys: list[str]
    metric_keys: list[str]
    submetric_keys: list[str]


#### I/O for data ####
def load_data(path: str, subjid: str) -> EEGDataset:
    ### Question from yash: Is this type of data always stored in this same filename format?
    loaded = np.load(f"{path}/{subjid}_data.npz", allow_pickle=True)
    data = loaded["data"]
    freq_amp_table = loaded["freq_amp_table"]
    latency = loaded["latency"].item()
    channel_keys = loaded["channel_keys"].tolist()
    period_keys = PERIOD_KEYS
    metric_keys = METRIC_KEYS
    submetric_keys = SUBMETRIC_KEYS

    return EEGDataset(
        data=data,
        freq_amp_table=freq_amp_table,
        latency=latency,
        channel_keys=channel_keys,
        period_keys=period_keys,
        metric_keys=metric_keys,
        submetric_keys=submetric_keys,
    )


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
