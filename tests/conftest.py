import pytest
import numpy as np


@pytest.fixture
def fake_channels(monkeypatch):
    def _fake_get_channels(_):
        return ["ch1", "ch2", "ch3", "ch4"]

    monkeypatch.setattr("fish_eeg.preprocess.get_channels", _fake_get_channels)
    return ["ch1", "ch2", "ch3", "ch4"]


@pytest.fixture
def fakedataset():
    """Minimal EEGDataset-like stub."""

    class _FakeDS:
        def __init__(self, data):
            self.data = data
            self.rms_filtered_data = None
            self.rms_subsampled_data = None

    return _FakeDS


@pytest.fixture
def small_clean_dict(fake_channels):
    """2 rows per channel, clean distribution."""
    return {ch: np.random.randn(2, 5) for ch in fake_channels}
