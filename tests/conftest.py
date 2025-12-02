import pytest
import numpy as np


@pytest.fixture
def fake_channels(monkeypatch):
    def _fake_get_channels(_):
        return ["ch1", "ch2", "ch3", "ch4"]

    monkeypatch.setattr("fish_eeg.preprocess.get_channels", _fake_get_channels)
    return ["ch1", "ch2", "ch3", "ch4"]


import pytest
import numpy as np
from fish_eeg.data import EEGDataset

@pytest.fixture
def fakedataset(fake_channels):
    """
    Returns a minimal EEGDataset-like object for testing.
    """
    
    # Create dummy data
    data_dict = {}
    for ch in fake_channels:
        # 2 trials, 50 samples per channel
        data_dict[ch] = np.random.randn(2, 50)
    
    # Wrap in 0-d object array for using Filter
    fakedata = np.array({"coordA": data_dict}, dtype=object)

    # ADD OTHER DATA HERE

    # Minimal freq_amp_table (please change if testing this aspect)
    fakefreq_amp_table = np.random.randn(2, 5)

    # Create EEGDataset instance
    ds = EEGDataset(
        data=fakedata,
        freq_amp_table=fakefreq_amp_table,
        latency=0,
        channel_keys=fake_channels,
        period_keys=["prestim", "stimresp"],
        metric_keys=["rms", "fft"],
        submetric_keys=["mean", "std"]
    )

    # Say the rms subsample is just data (no significant change to testing methods)
    ds.rms_subsampled_data = fakedata

    return ds



@pytest.fixture
def small_clean_dict(fake_channels):
    """2 rows per channel, clean distribution."""
    return {ch: np.random.randn(2, 5) for ch in fake_channels}
