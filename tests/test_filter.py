# test_filter.py
import numpy as np
from fish_eeg.filters import Filter


# ----------------------------------------------------
# 1. SMOKE TEST
# ----------------------------------------------------
def test_smoke_bandpass_runs(fakedataset, fake_channels):
    """
    author: Michael James
    reviewer: 
    category: smoke 
    Ensure bandpass() runs with no errors.
    """
    dictionary = {
            "ch1": np.random.randn(6, 3),
            "ch1_total_trials": 6,
            "ch2": np.random.randn(4, 3),
            "ch2_total_trials": 4,
            "ch3": np.random.randn(10, 3),
            "ch3_total_trials": 10,
            "ch4": np.random.randn(4, 3),
            "ch4_total_trials": 4,
        }

    ds = fakedataset(None)
    rms_subsampled_data = np.array(dictionary, dtype=object)
    f = Filter(ds)

    out = f.bandpass(rms_subsampled_data, low=1, high=30, fs=100)

    assert isinstance(out, dict)
    assert set(out.keys()) == set(fake_channels)
    return


# ----------------------------------------------------
# 2. ONE-SHOT TEST
# ----------------------------------------------------
def test_one_shot_constant_signal(FakeDataset, fake_channels):
    """
    author: Michael James
    reviewer: 
    category: one shot test
    A constant signal should produce ~zero through a bandpass filter.
    """
    const_signal = np.ones((1, 200))
    small_dict = {ch: const_signal for ch in fake_channels}

    ds = FakeDataset()
    ds.rms_subsampled_data = np.array(ds.data, dtype=object)
    f = Filter(ds)

    out = f.bandpass(small_dict, low=5, high=15, fs=100)
    filtered = out[fake_channels[0]]

    assert np.allclose(filtered, 0, atol=1e-2)
    return


# ----------------------------------------------------
# 3. EDGE TEST
# ----------------------------------------------------
def test_edge_missing_channel_pass_through(FakeDataset):
    """
    author: Michael James
    reviewer: 
    category: edge test
    If a channel is NOT listed in get_channels(), bandpass must return it unchanged.
    """
    unknown_key = "not_a_channel"
    d = {unknown_key: np.array([[1, 2, 3]])}

    ds = FakeDataset(data=small_clean_dict(fake_channels))
    ds.rms_subsampled_data = np.array(ds.data, dtype=object)
    f = Filter(ds)

    out = f.bandpass(d, low=1, high=30, fs=100)

    assert np.array_equal(out[unknown_key], d[unknown_key])
    return


# ----------------------------------------------------
# 4. PATTERN TEST
# ----------------------------------------------------
def test_pattern_pipeline_structure(FakeDataset, small_clean_dict):
    """
    author: Michael James
    reviewer: 
    category: pattern test
    pipeline() should preserve coordinates and produce a dict-of-dicts.
    """
    ds = FakeDataset()
    ds.rms_subsampled_data = np.array(ds.data, dtype=object)

    f = Filter(ds)
    out_ds = f.pipeline(low=1, high=30, fs=100)

    assert isinstance(out_ds.bandpass_data, np.ndarray)

    bp_dict = out_ds.bandpass_data.item()
    assert list(bp_dict.keys()) == ["coordA"]
    assert isinstance(bp_dict["coordA"], dict)
    return
