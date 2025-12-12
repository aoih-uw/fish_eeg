"""
Tests for preprocess.py functions
Authors: Yash Sonthalia & Jeffrey Jackson
Reviewer:
"""

import sys
import os
import numpy as np
import pytest

# Add src/ to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fish_eeg.preprocess import Preprocessor


# Smoke Test
# Created: Jeffrey Jackson
# Checked: Michael James

def test_filter_high_rms_smoke(fake_dataset, fake_channels):
    """Smoke test: runs without raising an exception."""
    ds = fake_dataset()  # uses default minimal data
    pre = Preprocessor(ds)

    # build small dict matching channels
    small_data = {ch: np.random.randn(5, 10) for ch in fake_channels}

    try:
        pre.FilterHighRMSTrials(small_data)
    except Exception as e:
        pytest.fail(f"FilterHighRMSTrials crashed: {e}")


# One Shot Test
# Created: Jeffrey Jackson
# Checked: Michael James

def test_filter_high_rms_one_trial_removed(fake_dataset, fake_channels):
    """
    Create a dataset where ONE channel has one extremely high RMS row.
    Because trials across channels must remain aligned, the outlier
    row should be removed across ALL channels.
    """
    # 20 normal rows + 1 gigantic outlier at the same index
    clean = {ch: np.ones((20, 5)) for ch in fake_channels}
    for ch in fake_channels:
        clean[ch] = np.vstack([clean[ch], np.ones((1, 5)) * 1e6])

    for ch in fake_channels:
        print(f"{ch} shape before filtering: {clean[ch].shape}")

    ds = fake_dataset(clean)
    pre = Preprocessor(ds)
    result = pre.FilterHighRMSTrials(clean)

    # Because the outlier is at the same index across channels,
    # each channel should end up with exactly 20 rows.
    for ch in fake_channels:
        assert result[ch].shape[0] == 20



# Edge Test
# Created: Jeffrey Jackson
# Checked: Michael James
def test_filter_high_rms_edge_all_identical_kept(fake_dataset, fake_channels):
    """
    If all trials are identical (zero variance), std = 0 and threshold = mean,
    so RMS == threshold and all rows pass. Test the actual behavior.
    """
    high_noise = {ch: np.ones((20, 5)) * 1e6 for ch in fake_channels}

    ds = fake_dataset(high_noise)
    pre = Preprocessor(ds)

    result = pre.FilterHighRMSTrials(high_noise)

    # Because std==0 and threshold==mean, all rows are kept
    for ch in fake_channels:
        assert result[ch].shape[0] == 20



# Pattern Test
# Created: Jeffrey Jackson
# Checked: Michael James

def test_filter_high_rms_pattern_known(fake_dataset):
    """
    Feed a precise known-pattern dataset where RMS is predictable.
    Only known rows should be removed.
    """
    # Build 20 normal RMS rows
    normal = np.ones((20, 4))

    # Insert predictable rows at top
    ch1 = np.array([
        [1,1,1,1],       # RMS = 1
        [50,50,50,50],   # RMS = 50 → should be removed
        [2,2,2,2],       # RMS = 2
    ])
    ch1 = np.vstack([ch1, normal])  # append normal rows

    data = {
        "ch1": ch1,
        "ch2": np.ones((23,4)),
        "ch3": np.ones((23,4)),
        "ch4": np.ones((23,4)),
    }

    ds = fake_dataset(data)
    pre = Preprocessor(ds)

    result = pre.FilterHighRMSTrials(data)

    filtered = result["ch1"]

    # Only the big RMS row (index=1) should be removed
    assert filtered.shape[0] == 22
    assert np.all(filtered[0] == np.array([1,1,1,1]))
    assert np.all(filtered[1] == np.array([2,2,2,2]))


# ============================================================
#  SUBSAMPLETRIALSPERCHANNEL — 4 TESTS
# ============================================================


def test_subsample_smoke(fake_channels, fake_dataset):
    """
    Smoke: method should execute.
    Args:
        fake_channels: The fake channels to use for the test.
        fake_dataset: The fake dataset to use for the test.
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
    ds = fake_dataset(None)
    p = Preprocessor(ds)
    out = p.SubsampleTrialsPerChannel(dictionary, 0)
    assert isinstance(out, dict)


def test_subsample_one_shot(fake_channels, fake_dataset):
    """
    One-shot: ensure min-trials logic works.
    Args:
        fake_channels: The fake channels to use for the test.
        fake_dataset: The fake dataset to use for the test.
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
    ds = fake_dataset(None)
    p = Preprocessor(ds)
    out = p.SubsampleTrialsPerChannel(dictionary, seed=0)
    for ch in fake_channels:
        assert out[ch].shape[0] == 3


def test_subsample_edge_single_trial(fake_channels, fake_dataset):
    """
    Edge: if one channel has only 1 trial → all must subsample to 1.
    Args:
        fake_channels: The fake channels to use for the test.
        fake_dataset: The fake dataset to use for the test.
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
    ds = fake_dataset(None)
    p = Preprocessor(ds)
    out = p.SubsampleTrialsPerChannel(dictionary, seed=123)
    for ch in fake_channels:
        assert out[ch].shape[0] == 1


def test_subsample_pattern_reproducible(fake_channels, fake_dataset):
    """
    Pattern: with same seed, subsampling is deterministic.
    Args:
        fake_channels: The fake channels to use for the test.
        fake_dataset: The fake dataset to use for the test.
    Returns:
        out: The subsampled data.
    """
    dictionary = {ch: np.arange(20).reshape(10, 2) for ch in fake_channels}
    # add *_total_trials
    for ch in fake_channels:
        dictionary[f"{ch}_total_trials"] = 10

    ds = fake_dataset(None)
    p = Preprocessor(ds)

    out1 = p.SubsampleTrialsPerChannel(dictionary, seed=42)
    out2 = p.SubsampleTrialsPerChannel(dictionary, seed=42)

    for ch in fake_channels:
        assert np.array_equal(out1[ch], out2[ch])


# ============================================================
#  PIPELINE — 4 TESTS
# ============================================================

def test_pipeline_smoke(fake_channels, fake_dataset):
    data = {
        (0, 1): {ch: np.random.randn(8, 5) for ch in fake_channels},
        (1, 2): {ch: np.random.randn(8, 5) for ch in fake_channels},
    }
    ds = fake_dataset(data)
    pre = Preprocessor(ds)
    out = pre.pipeline()
    assert out.rms_filtered_data is not None
    assert out.rms_subsampled_data is not None


def test_pipeline_one_shot(fake_channels, fake_dataset):
    dictionary = {ch: np.vstack([np.ones((4, 5)), np.full((1, 5), 999)]) for ch in fake_channels}
    ds = fake_dataset({(10, 20): dictionary})
    pre = Preprocessor(ds)
    out = pre.pipeline()

    filtered = out.rms_filtered_data[(10, 20)]
    for ch in fake_channels:
        assert filtered[ch].shape[0] == 4


def test_pipeline_edge_empty_after_filter(fake_channels, fake_dataset):
    dictionary = {ch: np.full((5, 5), 1000) for ch in fake_channels}
    ds = fake_dataset({(5, 5): dictionary})
    pre = Preprocessor(ds)
    out = pre.pipeline()

    filtered = out.rms_filtered_data[(5, 5)]
    for ch in fake_channels:
        assert filtered[ch].shape == (5, 5)


def test_pipeline_pattern_coords_kept_separate(fake_channels, fake_dataset):
    data = {
        (1, 1): {ch: np.random.randn(6, 4) for ch in fake_channels},
        (2, 2): {ch: np.random.randn(6, 4) for ch in fake_channels},
        (3, 3): {ch: np.random.randn(6, 4) for ch in fake_channels},
    }
    ds = fake_dataset(data)
    pre = Preprocessor(ds)
    out = pre.pipeline()

    coords = list(out.rms_filtered_data.keys())
    assert set(coords) == {(1, 1), (2, 2), (3, 3)}