"""
test_filter.py
Makes 4 unit tests for testing the filter function in the fish_eeg module
"""
import numpy as np
import sys
import os
from fish_eeg.filters import Filter
from fish_eeg.preprocess import Preprocessor

# Add src/ to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fish_eeg.filters import Filter


def test_smoke_bandpass_runs(fake_dataset, fake_channels):
    """
    author: Michael James
    reviewer: Jeff
    category: smoke 
    Ensure bandpass() runs with no errors.
    """
    n_trials = 5
    n_samples = 200
    wrapped_data = {
        (0,0): {ch: np.random.randn(n_trials, n_samples) for ch in fake_channels}
    }

    ds = fake_dataset(wrapped_data)

    # Run pipeline until filter
    preprocessor = Preprocessor(ds)
    ds = preprocessor.pipeline()
    try:
        filter = Filter(ds)
        ds = filter.pipeline()
        print("Pipeline successful up to Filter")
    except Exception:
        print("Pipeline failed at Filter")

    assert isinstance(ds.bandpass_data, dict) #Must ensure we keep bandpass,
    #must have further into pipeline.


def test_one_shot_constant_signal(fake_dataset, fake_channels):
    """
    author: Michael James
    reviewer: Jeff
    category: one shot test
    A constant signal should produce ~zero through a bandpass filter.
    """

    # Create constant signal (long enough for filtfilt)
    n_samples = 100
    const_signal = np.ones((1, n_samples))
    channel_dict = {ch: const_signal for ch in fake_channels}

    # Use fake dataset
    ds = fake_dataset()
    preprocessor = Preprocessor(ds)
    ds = preprocessor.pipeline()
    f = Filter(ds)

    # Apply bandpass (5-15 Hz)
    out = f.bandpass(channel_dict, low=5, high=15, fs=100, order=4)
    filtered = out[fake_channels[0]]

    # For a constant input, bandpass ~ zero
    assert np.allclose(filtered, 0, atol=1e-2)


def test_edge_missing_channel_pass_through(fake_dataset,fake_channels):
    """
    author: Michael James
    reviewer: Jeff
    category: edge test
    If a channel is NOT listed in get_channels(), bandpass must return it unchanged.
    """

    unknown_key = "not_a_channel"
    # small array with enough samples
    d = {unknown_key: np.random.randn(1, 50)}

    #Set up data
    n_trials = 100
    n_samples = 20
    wrapped_data = {
        (0,0): {ch: np.random.randn(n_trials, n_samples) for ch in fake_channels}
    }

    # Run pipeline
    ds = fake_dataset(wrapped_data)
    preprocessor = Preprocessor(ds)
    ds = preprocessor.pipeline()
    f = Filter(ds)
    out = f.bandpass(d, low=1, high=30, fs=100)

    # Channel not in get_channels() should remain unchanged
    assert np.array_equal(out[unknown_key], d[unknown_key])


def test_pattern_pipeline_structure(fake_dataset,fake_channels):
    """
    author: Michael James
    reviewer: Jeff
    category: pattern test
    pipeline() should preserve coordinates and produce a dict-of-dicts.
    """
    #Set up data
    n_trials = 100
    n_samples = 100
    wrapped_data = {
        (0,0): {ch: np.random.randn(n_trials, n_samples) for ch in fake_channels}
    }

    # Run pipeline
    ds = fake_dataset(wrapped_data)
    preprocessor = Preprocessor(ds)
    ds = preprocessor.pipeline()
    f = Filter(ds)
    out_ds = f.pipeline()

    
    # Check top-level type
    assert isinstance(out_ds.bandpass_data, dict), "bandpass_data should be a dict"

    # Check structure of coordinates
    for coord, channel_data in out_ds.bandpass_data.items():
        assert isinstance(channel_data, dict), f"Coordinate {coord} should map to a dict"
        # Ensure all expected channels are present
        channels_in_data = [k for k in channel_data.keys() if not k.endswith("_total_trials")]
        assert set(channels_in_data) == set(fake_channels), (
            f"Coordinate {coord} missing channels or has extra channels: "
            f"{set(channels_in_data) ^ set(fake_channels)}"
        )
        # Ensure all arrays are numpy arrays
        assert all(isinstance(arr, np.ndarray) for ch, arr in channel_data.items() if not ch.endswith("_total_trials")), (
            f"Coordinate {coord} has non-numpy array channel(s)"
        )
