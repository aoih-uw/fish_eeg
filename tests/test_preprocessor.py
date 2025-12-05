"""
Tests for preprocess.py functions
Author: Yash Sonthalia
Reviewer:
"""

import numpy as np
from fish_eeg.preprocess import Preprocessor

# ============================================================
#  SUBSAMPLETRIALSPERCHANNEL — 4 TESTS
# ============================================================


def test_subsample_smoke(fake_channels, FakeDataset):
    """
    Smoke: method should execute.
    Args:
        fake_channels: The fake channels to use for the test.
        FakeDataset: The fake dataset to use for the test.
    Returns:
        out: The subsampled data.
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
    ds = FakeDataset(None)
    p = Preprocessor(ds)
    out = p.SubsampleTrialsPerChannel(dictionary, 0)
    assert isinstance(out, dict)


def test_subsample_one_shot(fake_channels, FakeDataset):
    """
    One-shot: ensure min-trials logic works.
    Args:
        fake_channels: The fake channels to use for the test.
        FakeDataset: The fake dataset to use for the test.
    Returns:
        out: The subsampled data.
    """
    dictionary = {
        "ch1": np.random.randn(5, 4),
        "ch1_total_trials": 5,
        "ch2": np.random.randn(3, 4),
        "ch2_total_trials": 3,
        "ch3": np.random.randn(9, 4),
        "ch3_total_trials": 9,
        "ch4": np.random.randn(3, 4),
        "ch4_total_trials": 3,
    }
    ds = FakeDataset(None)
    p = Preprocessor(ds)
    out = p.SubsampleTrialsPerChannel(dictionary, seed=0)
    for ch in fake_channels:
        assert out[ch].shape[0] == 3


def test_subsample_edge_single_trial(fake_channels, FakeDataset):
    """
    Edge: if one channel has only 1 trial → all must subsample to 1.
    Args:
        fake_channels: The fake channels to use for the test.
        FakeDataset: The fake dataset to use for the test.
    Returns:
        out: The subsampled data.
    """
    dictionary = {
        "ch1": np.random.randn(1, 5),
        "ch1_total_trials": 1,
        "ch2": np.random.randn(10, 5),
        "ch2_total_trials": 10,
        "ch3": np.random.randn(7, 5),
        "ch3_total_trials": 7,
        "ch4": np.random.randn(3, 5),
        "ch4_total_trials": 3,
    }
    ds = FakeDataset(None)
    p = Preprocessor(ds)
    out = p.SubsampleTrialsPerChannel(dictionary, seed=123)
    for ch in fake_channels:
        assert out[ch].shape[0] == 1


def test_subsample_pattern_reproducible(fake_channels, FakeDataset):
    """
    Pattern: with same seed, subsampling is deterministic.
    Args:
        fake_channels: The fake channels to use for the test.
        FakeDataset: The fake dataset to use for the test.
    Returns:
        out: The subsampled data.
    """
    dictionary = {ch: np.arange(20).reshape(10, 2) for ch in fake_channels}
    # add *_total_trials
    for ch in fake_channels:
        dictionary[f"{ch}_total_trials"] = 10

    ds = FakeDataset(None)
    p = Preprocessor(ds)

    out1 = p.SubsampleTrialsPerChannel(dictionary, seed=42)
    out2 = p.SubsampleTrialsPerChannel(dictionary, seed=42)

    for ch in fake_channels:
        assert np.array_equal(out1[ch], out2[ch])


# ============================================================
#  PIPELINE — 4 TESTS
# ============================================================


def test_pipeline_smoke(fake_channels, FakeDataset):
    """
    Smoke: pipeline should run end-to-end.
    Args:
        fake_channels: The fake channels to use for the test.
        FakeDataset: The fake dataset to use for the test.
    Returns:
        out.rms_filtered_data: The filtered data.
        out.rms_subsampled_data: The subsampled data.
    """
    data = {
        (0, 1): {ch: np.random.randn(8, 5) for ch in fake_channels},
        (1, 2): {ch: np.random.randn(8, 5) for ch in fake_channels},
    }
    ds = FakeDataset(np.array(data, dtype=object))
    p = Preprocessor(ds)
    out = p.pipeline()
    assert out.rms_filtered_data is not None
    assert out.rms_subsampled_data is not None


def test_pipeline_one_shot(fake_channels, FakeDataset):
    """
    One-shot: channels with a single outlier get filtered then subsampled.
    Args:
        fake_channels: The fake channels to use for the test.
        FakeDataset: The fake dataset to use for the test.
    Returns:
        out: The filtered and subsampled data.
    """
    dictionary = {
        ch: np.vstack([np.ones((4, 5)), np.full((1, 5), 999)]) for ch in fake_channels
    }
    ds = FakeDataset(np.array({(10, 20): dictionary}, dtype=object))
    p = Preprocessor(ds)
    out = p.pipeline()

    # after filtering: 4 rows
    # after subsampling: all channels will have min=4 rows
    coord = (10, 20)
    filtered = out.rms_filtered_data.item()[coord]

    for ch in fake_channels:
        assert filtered[ch].shape[0] == 4


def test_pipeline_edge_empty_after_filter(fake_channels, FakeDataset):
    """
    Edge: if ALL rows are extreme outliers → no filtering removed (std=0 case).
    Args:
        fake_channels: The fake channels to use for the test.
        FakeDataset: The fake dataset to use for the test.
    Returns:
        out: The filtered and subsampled data.
    """
    dictionary = {ch: np.full((5, 5), 1000) for ch in fake_channels}
    ds = FakeDataset(np.array({(5, 5): dictionary}, dtype=object))
    p = Preprocessor(ds)
    out = p.pipeline()

    filtered = out.rms_filtered_data.item()[(5, 5)]
    for ch in fake_channels:
        assert filtered[ch].shape == (5, 5)


def test_pipeline_pattern_coords_kept_separate(fake_channels, FakeDataset):
    """
    Pattern: pipeline preserves dictionary keys (coords).
    Args:
        fake_channels: The fake channels to use for the test.
        FakeDataset: The fake dataset to use for the test.
    Returns:
        out: The filtered and subsampled data.
    """
    data = {
        (1, 1): {ch: np.random.randn(6, 4) for ch in fake_channels},
        (2, 2): {ch: np.random.randn(6, 4) for ch in fake_channels},
        (3, 3): {ch: np.random.randn(6, 4) for ch in fake_channels},
    }

    ds = FakeDataset(np.array(data, dtype=object))
    p = Preprocessor(ds)
    out = p.pipeline()

    coords = list(out.rms_filtered_data.item().keys())
    assert set(coords) == {(1, 1), (2, 2), (3, 3)}
