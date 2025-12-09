import numpy as np
from dataclasses import dataclass
from fish_eeg.constants import PERIOD_KEYS, METRIC_KEYS, SUBMETRIC_KEYS
from typing import Optional, Any, Mapping


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


@dataclass
class PipelineConfig:
    def __init__(self, yaml_loaded: dict):
        self.preprocess: dict = yaml_loaded["preprocess"]
        self.filters: dict = yaml_loaded["filters"]
        self.denoiser: dict = yaml_loaded["denoiser"]
        self.reconstruct: dict = yaml_loaded["reconstruct"]
        self.statistics: dict = yaml_loaded["statistics"]


@dataclass
class SectionAccessor:
    """
    Safe wrapper around a single config section (a dict-like).

    Lets you do both:
        cfg.preprocess.get("method", "rms_subsampled")
        cfg.preprocess.method
    """

    _data: Optional[Mapping[str, Any]] = None

    @property
    def is_present(self) -> bool:
        return bool(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        if self._data is None:
            return default
        return self._data.get(key, default)

    def as_dict(self) -> dict:
        return dict(self._data or {})

    def __getattr__(self, item: str) -> Any:
        """
        Attribute-style access: section.method, section.seed, etc.
        Falls back to normal AttributeError if key not present.
        """
        if self._data is not None and item in self._data:
            return self._data[item]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{item}'")


class ConfigAccessor:
    """
    Thin wrapper around PipelineConfig that is safe to use even if config is None.

    Usage:
        cfg = ConfigAccessor(pipeline_config_or_none)

        # dict-like section/key access
        method = cfg.get("preprocess", "method", default="rms_subsampled")

        # section accessor
        preprocess = cfg.section("preprocess")
        method = preprocess.get("method", "rms_subsampled")
        seed = preprocess.seed  # attribute form

        # attribute shortcut for sections:
        method = cfg.preprocess.method
    """

    def __init__(self, config: Optional["PipelineConfig"] = None):
        self._config = config

    @property
    def is_present(self) -> bool:
        return self._config is not None

    # ------------ high-level helpers ------------

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Safe access:
            cfg.get('preprocess', 'method', default='rms_subsampled')
        """
        return self.section(section).get(key, default)

    def section(self, section: str) -> SectionAccessor:
        """
        Return a SectionAccessor for a given section name.

        Safe even if:
        - overall config is None
        - that section doesn't exist
        """
        if self._config is None:
            return SectionAccessor(None)

        section_dict = getattr(self._config, section, None)
        return SectionAccessor(section_dict)

    # ------------ attribute shortcut for sections ------------

    def __getattr__(self, name: str) -> SectionAccessor:
        """
        Fallback: treat unknown attributes as section names.

        This lets you do:
            cfg.preprocess.method
            cfg.filters.low
        even though `preprocess` / `filters` aren't real attributes
        on ConfigAccessor itself.
        """
        # don't swallow private / dunder attributes
        if name.startswith("_"):
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
        return self.section(name)


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
        data=data,
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


def collapse_channels(data, period_keys, channel_keys):
    collapsed_dict = {"prestim": None, "stimresp": None}
    for period in period_keys:
        tmp = []
        for channel in channel_keys:
            tmp.append(data[period][channel])
        collapsed_dict[period] = np.vstack(tmp)

    return collapsed_dict
