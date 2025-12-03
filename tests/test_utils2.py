"""
Tests for utils.py functions
Author: Christopher Tritt
Reviewer: yashsonthalia
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "fish_eeg"))

from utils import get_channels
from data import EEGDataset


def test_get_channels_smoke():
    """
    author: Christopher Tritt
    reviewer: [To be assigned]
    category: smoke test
    
    Basic smoke test to ensure get_channels runs without crashing
    with a minimal valid input.
    """
    # Create a minimal valid EEGDataset
    test_data = {
        (100.0, 110.0): {
            "ch1": np.array([[1, 2, 3]]),
            "ch2": np.array([[4, 5, 6]]),
            "other_key": "value"
        }
    }
    
    eegdataset = EEGDataset(
        data=np.array(test_data, dtype=object),
        freq_amp_table=np.array([[100, 110]]),
        latency=2118,
        channel_keys=["ch1", "ch2"],
        period_keys=["prestim", "stimresp"],
        metric_keys=["mean", "std"],
        submetric_keys=["doub_freq_mag", "SNR"]
    )
    
    # Should run without error
    result = get_channels(eegdataset)
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_channels_one_shot():
    """
    author: Christopher Tritt
    reviewer: [To be assigned]
    category: one-shot test
    
    One-shot test with known input and expected output.
    Tests that the function correctly extracts channel names from a
    typical EEGDataset structure.
    """
    # Create test data with known channels
    test_data = {
        (220.0, 130.0): {
            "ch1": np.array([[1, 2, 3, 4, 5]]),
            "ch2": np.array([[6, 7, 8, 9, 10]]),
            "ch3": np.array([[11, 12, 13, 14, 15]]),
            "ch4": np.array([[16, 17, 18, 19, 20]]),
            "timestamp": 12345,
            "other_metadata": "value"
        }
    }
    
    eegdataset = EEGDataset(
        data=np.array(test_data, dtype=object),
        freq_amp_table=np.array([[220, 130]]),
        latency=2118,
        channel_keys=["ch1", "ch2", "ch3", "ch4"],
        period_keys=["prestim", "stimresp"],
        metric_keys=["mean", "std"],
        submetric_keys=["doub_freq_mag", "SNR"]
    )
    
    result = get_channels(eegdataset)
    
    # Should return exactly these 4 channels
    assert len(result) == 4
    assert set(result) == {"ch1", "ch2", "ch3", "ch4"}


def test_get_channels_edge_empty_data():
    """
    author: Christopher Tritt
    reviewer: [To be assigned]
    category: edge test
    
    Edge case: Test behavior with empty data dictionary.
    The function should handle empty data gracefully.
    """
    # Create dataset with empty data dictionary
    test_data = {}
    
    eegdataset = EEGDataset(
        data=np.array(test_data, dtype=object),
        freq_amp_table=np.array([]),
        latency=2118,
        channel_keys=[],
        period_keys=["prestim", "stimresp"],
        metric_keys=["mean", "std"],
        submetric_keys=["doub_freq_mag", "SNR"]
    )
    
    result = get_channels(eegdataset)
    
    # Should return empty list for empty data
    assert isinstance(result, list)
    assert len(result) == 0


def test_get_channels_edge_no_channels():
    """
    author: Christopher Tritt
    reviewer: [To be assigned]
    category: edge test
    
    Edge case: Test with data that contains no channel keys.
    Should return empty list when only non-channel keys are present.
    """
    # Create data with no channel keys
    test_data = {
        (100.0, 110.0): {
            "timestamp": 12345,
            "metadata": "some_value",
            "frequency": 100.0
        }
    }
    
    eegdataset = EEGDataset(
        data=np.array(test_data, dtype=object),
        freq_amp_table=np.array([[100, 110]]),
        latency=2118,
        channel_keys=[],
        period_keys=["prestim", "stimresp"],
        metric_keys=["mean", "std"],
        submetric_keys=["doub_freq_mag", "SNR"]
    )
    
    result = get_channels(eegdataset)
    
    # Should return empty list when no channels present
    assert result == []


def test_get_channels_edge_mixed_keys():
    """
    author: Christopher Tritt
    reviewer: [To be assigned]
    category: edge test
    
    Edge case: Test that function correctly filters channel keys from
    other keys that start with 'ch' but don't end with digits.
    """
    # Create data with various key formats
    test_data = {
        (100.0, 110.0): {
            "ch1": np.array([[1, 2, 3]]),
            "ch2": np.array([[4, 5, 6]]),
            "channel": "not_a_channel",  # starts with ch but not valid
            "ch_temp": "also_not_valid",  # starts with ch but not valid
            "chest": "still_not_valid",   # starts with ch but not valid
            "timestamp": 12345
        }
    }
    
    eegdataset = EEGDataset(
        data=np.array(test_data, dtype=object),
        freq_amp_table=np.array([[100, 110]]),
        latency=2118,
        channel_keys=["ch1", "ch2"],
        period_keys=["prestim", "stimresp"],
        metric_keys=["mean", "std"],
        submetric_keys=["doub_freq_mag", "SNR"]
    )
    
    result = get_channels(eegdataset)
    
    # Should only return valid channel keys (ch + digit)
    assert len(result) == 2
    assert set(result) == {"ch1", "ch2"}


def test_get_channels_pattern_multiple_stimuli():
    """
    author: Christopher Tritt
    reviewer: [To be assigned]
    category: pattern test
    
    Pattern test: Verify that get_channels correctly extracts channels
    across multiple stimulus conditions (different frequency-amplitude pairs).
    The function should find all unique channels across all conditions.
    """
    # Create data with multiple stimulus conditions
    test_data = {
        (55.0, 130.0): {
            "ch1": np.array([[1, 2, 3]]),
            "ch2": np.array([[4, 5, 6]]),
            "timestamp": 1000
        },
        (100.0, 105.0): {
            "ch1": np.array([[7, 8, 9]]),
            "ch3": np.array([[10, 11, 12]]),
            "timestamp": 2000
        },
        (220.0, 125.0): {
            "ch2": np.array([[13, 14, 15]]),
            "ch4": np.array([[16, 17, 18]]),
            "timestamp": 3000
        }
    }
    
    eegdataset = EEGDataset(
        data=np.array(test_data, dtype=object),
        freq_amp_table=np.array([[55, 130], [100, 105], [220, 125]]),
        latency=2118,
        channel_keys=["ch1", "ch2", "ch3", "ch4"],
        period_keys=["prestim", "stimresp"],
        metric_keys=["mean", "std"],
        submetric_keys=["doub_freq_mag", "SNR"]
    )
    
    result = get_channels(eegdataset)
    
    # Should find all unique channels across all stimulus conditions
    assert len(result) == 4
    assert set(result) == {"ch1", "ch2", "ch3", "ch4"}


def test_get_channels_pattern_large_channel_numbers():
    """
    author: Christopher Tritt
    reviewer: [To be assigned]
    category: pattern test
    
    Pattern test: Verify the function works with larger channel numbers
    (e.g., ch10, ch99) to ensure the digit-checking logic is robust.
    """
    # Create data with multi-digit channel numbers
    test_data = {
        (100.0, 110.0): {
            "ch1": np.array([[1, 2, 3]]),
            "ch10": np.array([[4, 5, 6]]),
            "ch99": np.array([[7, 8, 9]]),
            "ch123": np.array([[10, 11, 12]]),
            "timestamp": 12345
        }
    }
    
    eegdataset = EEGDataset(
        data=np.array(test_data, dtype=object),
        freq_amp_table=np.array([[100, 110]]),
        latency=2118,
        channel_keys=["ch1", "ch10", "ch99", "ch123"],
        period_keys=["prestim", "stimresp"],
        metric_keys=["mean", "std"],
        submetric_keys=["doub_freq_mag", "SNR"]
    )
    
    result = get_channels(eegdataset)
    
    # Should correctly identify all channels with any number of digits
    assert len(result) == 4
    assert set(result) == {"ch1", "ch10", "ch99", "ch123"}


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
