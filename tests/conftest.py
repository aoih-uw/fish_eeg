import pytest
import numpy as np
from fish_eeg.data import EEGDataset
from fish_eeg.preprocess import Preprocessor


@pytest.fixture
def fake_channels(monkeypatch):
    def _fake_get_channels(_):
        return ["ch1", "ch2", "ch3", "ch4"]

    monkeypatch.setattr("fish_eeg.preprocess.get_channels", _fake_get_channels)
    return ["ch1", "ch2", "ch3", "ch4"]

@pytest.fixture
def fake_dataset(fake_channels):
    """
    Factory fixture: returns an EEGDataset-like object.
    Can accept either a dict (ch -> 2D array) or a numpy array.
    Ensures that ds.data and ds.rms_subsampled_data are always dicts.
    """
    def _make(wrapped_data=None):
        # If no wrapped_data provided, create minimal structure
        if wrapped_data is None:
            wrapped_data = {
                (0, 0): {ch: np.random.randn(5, 10) for ch in fake_channels}
            }

        ds = EEGDataset(
            data=wrapped_data,
            freq_amp_table=np.random.randn(2, 3),
            latency=0,
            channel_keys=fake_channels,
            period_keys=["prestim", "stimresp"],
            metric_keys=["rms", "fft"],
            submetric_keys=["mean", "std"],
            period_len=10
        )
        return ds
    return _make


@pytest.fixture
def small_clean_dict(fake_channels):
    """2 rows per channel, clean distribution."""
    return {ch: np.random.randn(2, 5) for ch in fake_channels}


@pytest.fixture
def sinusoid_dataset(fake_dataset):
    """
    Returns an EEGDataset where each channel contains sinusoidal data.
    If different_trials=True, each trial gets a different frequency.
    """

    def _make(
        fs=1000,
        n_samples=1000,
        n_trials=5,
        base_freq=10,
        different_trials=False
    ):

        t = np.arange(n_samples) / fs
        channel_keys = ["ch1", "ch2", "ch3", "ch4"]

        data_dict = {}

        for ch in channel_keys:
            ch_idx = int(ch.replace("ch", ""))

            trials = []

            for trial in range(n_trials):

                if different_trials:
                    # EXAMPLE variation: frequency changes per trial
                    freq = base_freq * (ch_idx + trial)
                    phase = np.random.uniform(0, 2*np.pi)
                    amplitude = 1.0 + 0.1 * trial
                else:
                    # SAME signal every trial
                    freq = base_freq * (ch_idx + 1)
                    phase = 0
                    amplitude = 1.0

                signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
                trials.append(signal)

            data_dict[ch] = np.array(trials)

        # Pack into the wrapped format that fake_dataset expects
        wrapped = np.array({"data": data_dict}, dtype=object)

        ds = fake_dataset(wrapped_data=wrapped)

        return ds

    return _make



@pytest.fixture
def temp_eeg_data(tmp_path, fake_channels):
    """
    Creates a temporary .npz file with fake EEG data for testing load_data.
    Returns (path, subjid) tuple.
    """
    # Create fake data
    data_dict = {}
    for ch in fake_channels:
        data_dict[ch] = np.random.randn(2, 50)

    fakefreq_amp_table = np.random.randn(2, 5)

    # Subject ID
    subjid = "test_subject"
    file_path = tmp_path / f"{subjid}_data.npz"
    # Save to temporary file
    np.savez(
        file_path,
        data=data_dict,
        freq_amp_table=fakefreq_amp_table,
        latency=np.array([0]),
        channel_keys=np.array(fake_channels)
    )

    return str(tmp_path), subjid