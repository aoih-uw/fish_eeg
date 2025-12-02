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

def test_filter_high_rms_smoke(FakeDataset, small_clean_dict):
    """Smoke test: runs without raising an exception."""
    # Preprocessor expects eegdataset.data to be a numpy object array containing a dict of coords
    wrapped_data = np.array({"coord1": small_clean_dict}, dtype=object)
    ds = FakeDataset(wrapped_data)

    pre = Preprocessor(ds)

    # Run the method
    output = pre.FilterHighRMSTrials(small_clean_dict)

    # Output should be a dict and non-empty
    assert isinstance(output, dict)
    assert len(output) > 0


# One Shot Test
# Created: Jeffrey Jackson
# Checked: Michael James

def test_filter_high_rms_one_trial_removed(FakeDataset, fake_channels):
    """
    Create a dataset where ONE channel has one extremely high RMS row.
    Check that exactly one row is removed.
    """
    clean = {ch: np.random.randn(3, 5) for ch in fake_channels}
    clean["ch1"][1] *= 10000  # Giant outlier

    wrapped_data = np.array({"coord1": clean}, dtype=object)
    ds = FakeDataset(wrapped_data)

    pre = Preprocessor(ds)

    result = pre.FilterHighRMSTrials(clean)

    # ch1 should now have only 2 rows
    assert result["ch1"].shape[0] == 2


# Edge Test
# Created: Jeffrey Jackson
# Checked: Michael James

def test_filter_high_rms_edge_all_removed(FakeDataset, fake_channels):
    """
    All rows exceed the RMS threshold → everything should be removed.
    """
    high_noise = {ch: np.ones((4, 5)) * 9999 for ch in fake_channels}

    wrapped_data = np.array({"coord1": high_noise}, dtype=object)
    ds = FakeDataset(wrapped_data)

    pre = Preprocessor(ds)

    result = pre.FilterHighRMSTrials(high_noise)

    # All channels should have 0 rows left
    for ch in fake_channels:
        assert result[ch].shape[0] == 0



# Pattern Test
# Created: Jeffrey Jackson
# Checked: Michael James

def test_filter_high_rms_pattern_known(FakeDataset):
    """
    Feed a precise known-pattern dataset where RMS is predictable.
    Only known rows should be removed.
    """
    data = {
        "ch1": np.array([
            [1, 1, 1, 1],      # RMS = 1
            [10, 10, 10, 10],  # RMS = 10 (should be removed)
            [2, 2, 2, 2],      # RMS = 2
        ])
    }

    wrapped_data = np.array({"coord1": data}, dtype=object)
    ds = FakeDataset(wrapped_data)

    pre = Preprocessor(ds)

    result = pre.FilterHighRMSTrials(data)

    filtered = result["ch1"]

    # Only the high-RMS row should be removed
    assert filtered.shape[0] == 2
    assert np.array_equal(filtered[0], [1, 1, 1, 1])
    assert np.array_equal(filtered[1], [2, 2, 2, 2])
