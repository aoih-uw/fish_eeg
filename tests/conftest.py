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


# @pytest.fixture
# def fake_dataset(fake_channels):
#     """
#     Returns a minimal EEGDataset-like object for testing.
#     """
    
#     # Create dummy data
#     data_dict = {}
#     for ch in fake_channels:
#         # 2 trials, 50 samples per channel
#         data_dict[ch] = np.random.randn(2, 50)
    
#     # Wrap in 0-d object array for using Filter
#     fakedata = np.array({"data": data_dict}, dtype=object)

#     # Minimal freq_amp_table (please change if testing this aspect)
#     fakefreq_amp_table = np.random.randn(2, 5)

#     # Create EEGDataset instance
#     ds = EEGDataset(
#         data=fakedata,
#         freq_amp_table=fakefreq_amp_table,
#         latency=0,
#         channel_keys=fake_channels,
#         period_keys=["prestim", "stimresp"],
#         metric_keys=["rms", "fft"],
#         submetric_keys=["mean", "std"]
#     )

#     # Say the rms subsample is just data (no significant change to testing methods)
#     ds.rms_subsampled_data = fakedata

#     # ADD ANY OTHER STRUCTURAL COMPONENTS HERE

#     return ds

@pytest.fixture
def fake_dataset(fake_channels):
    """
    Factory fixture: returns an EEGDataset-like object.
    Can be called with or without custom wrapped data.
    """

    def _make(wrapped_data=None):
        #No data provided
        if wrapped_data is None:
            data_dict = {ch: np.random.randn(2, 50) for ch in fake_channels}
            wrapped_data = np.array({"data": data_dict}, dtype=object)

        # Minimal freq_amp_table
        fakefreq_amp_table = np.random.randn(2, 5)

                # ---- Normalize wrapped_data into raw dict ----
        if isinstance(wrapped_data, dict):
            raw = wrapped_data

        elif isinstance(wrapped_data, np.ndarray):
            # CASE 1: 0-dim object array → contains a dict
            if wrapped_data.ndim == 0:
                raw = wrapped_data.item()

            # CASE 2: Multi-dimensional ndarray → this *is* the data
            else:
                # Wrap into dict for consistency
                raw = {ch: wrapped_data for ch in fake_channels}

        else:
            raise TypeError(f"wrapped_data must be dict or ndarray, got {type(wrapped_data)}")

        # ---- Extract the actual data dict ----
        if "data" in raw:
            data_dict = raw["data"]
        else:
            # Either { (freq,amp): {...} } OR {ch: array}
            first_key = next(iter(raw.keys()))
            val = raw[first_key]

            # If mapping channels → array
            if isinstance(val, np.ndarray):
                # raw is {ch: array}
                data_dict = raw
            else:
                # raw is { (freq,amp): {ch: array} }
                data_dict = val
        # Period length (# time points)
        period_len = data_dict[fake_channels[0]].shape[1]

        # Build dataset
        ds = EEGDataset(
            data=wrapped_data,
            freq_amp_table=fakefreq_amp_table,
            latency=0,
            channel_keys=fake_channels,
            period_keys=["prestim", "stimresp"],
            metric_keys=["rms", "fft"],
            submetric_keys=["mean", "std"],
            period_len= period_len
        )

        # Keep this for compatibility
        ds.rms_subsampled_data = wrapped_data

        return ds

    return _make


@pytest.fixture
def small_clean_dict(fake_channels):
    """2 rows per channel, clean distribution."""
    return {ch: np.random.randn(2, 5) for ch in fake_channels}


@pytest.fixture
def fake_ica_output(fake_dataset):
    """Create a minimal ICA-output-like object derived from the dataset."""

    ds = fake_dataset()   # contains channels, trials, etc.

    class FakeICA:
        pass

    obj = FakeICA()
    obj.channel_keys = ds.channel_keys
    obj.period_keys = ds.period_keys

    # ICA components: same shape as EEG data
    obj.reconstructed_ica_data = {
        period: {
            ch: np.copy(ds.data.item()["data"][ch])
            for ch in ds.channel_keys
        }
        for period in ds.period_keys
    }

    obj.ica_output = None  # if your code expects this attribute

    return obj

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
