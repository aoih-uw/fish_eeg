# test_filter.py
import numpy as np
from fish_eeg.filters import Filter


# ----------------------------------------------------
# 1. SMOKE TEST
# ----------------------------------------------------
# tests/test_filter.py
import numpy as np
from fish_eeg.filters import Filter

def test_smoke_bandpass_runs(fakedataset, fake_channels):
    """
    author: Michael James
    reviewer: 
    category: smoke 
    Ensure bandpass() runs with no errors.
    """

    # 1. Initialize Filter with the EEGDataset fixture
    f = Filter(fakedataset)

    # 2. Extract the channel dictionary for testing bandpass
    channel_dict = fakedataset.rms_subsampled_data.item()["data"]

    # 3. Call bandpass
    out = f.bandpass(channel_dict, low=1, high=30, fs=100)

    # 4. Assertions
    assert isinstance(out, dict)
    assert set(out.keys()) == set(fake_channels)




# ----------------------------------------------------
# 2. ONE-SHOT TEST
# ----------------------------------------------------
def test_one_shot_constant_signal(fakedataset, fake_channels):
    """
    author: Michael James
    reviewer: 
    category: one shot test
    A constant signal should produce ~zero through a bandpass filter.
    """

    # Create constant signal (long enough for filtfilt)
    n_samples = 100
    const_signal = np.ones((1, n_samples))
    channel_dict = {ch: const_signal for ch in fake_channels}

    # Use fake dataset
    ds = fakedataset
    f = Filter(ds)

    # Apply bandpass (5-15 Hz)
    out = f.bandpass(channel_dict, low=5, high=15, fs=100, order=4)
    filtered = out[fake_channels[0]]

    # For a constant input, bandpass ~ zero
    assert np.allclose(filtered, 0, atol=1e-2)


# ----------------------------------------------------
# 3. EDGE TEST
# ----------------------------------------------------
def test_edge_missing_channel_pass_through(fakedataset):
    """
    author: Michael James
    reviewer: 
    category: edge test
    If a channel is NOT listed in get_channels(), bandpass must return it unchanged.
    """

    unknown_key = "not_a_channel"
    # small array with enough samples
    d = {unknown_key: np.random.randn(1, 50)}

    f = Filter(fakedataset)
    out = f.bandpass(d, low=1, high=30, fs=100)

    # Channel not in get_channels() should remain unchanged
    assert np.array_equal(out[unknown_key], d[unknown_key])


# ----------------------------------------------------
# 4. PATTERN TEST
# ----------------------------------------------------
def test_pattern_pipeline_structure(fakedataset):
    """
    author: Michael James
    reviewer: 
    category: pattern test
    pipeline() should preserve coordinates and produce a dict-of-dicts.
    """

    f = Filter(fakedataset)
    out_ds = f.pipeline(low=1, high=30, fs=100, order=4)

    # pipeline output should be stored in bandpass_data
    assert isinstance(out_ds.bandpass_data, np.ndarray)

    # Extract dict-of-dicts
    bp_dict = out_ds.bandpass_data.item()
    assert list(bp_dict.keys()) == ["data"]
    assert isinstance(bp_dict["data"], dict)
