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

def test_filter_high_rms_smoke(fake_dataset, small_clean_dict):
    """Smoke test: runs without raising an exception."""
    # Preprocessor expects eegdataset.data to be a numpy object array containing a dict of coords
    wrapped_data = np.array({"coord1": small_clean_dict}, dtype=object)
    ds = fake_dataset(wrapped_data)

    pre = Preprocessor(ds)

    # Run the method
    output = pre.FilterHighRMSTrials(small_clean_dict)

    # Output should be a dict and non-empty
    assert isinstance(output, dict)
    assert len(output) > 0


# One Shot Test
# Created: Jeffrey Jackson
# Checked: Michael James

def test_filter_high_rms_one_trial_removed(fake_dataset, fake_channels):
    """
    Create a dataset where ONE channel has one extremely high RMS row.
    Check that exactly one row is removed.
    """
     # 20 normal rows + 1 gigantic outlier
    clean = {ch: np.random.randn(20, 5) for ch in fake_channels}
    for ch in fake_channels:
        clean[ch] = np.vstack([
            clean[ch],
            np.ones((1, 5)) * 1e6   # huge RMS row
        ])

    ds = fake_dataset(clean)
    pre = Preprocessor(ds)

    result = pre.FilterHighRMSTrials(clean)

    # Each channel should now have ONLY 20 rows (last one removed)
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

