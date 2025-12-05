import numpy as np
from dataclasses import dataclass
from fish_eeg.data import load_data
import os
import pytest

# Create fixture for temporary testing files
@pytest.fixture
def temp_eeg_data(tmp_path, fake_channels):
    """
    Creates a temporary .npz file with fake EEG data for testing load_data.
    Returns (path, subjid) tuple.
    """
    # Create fake data matching the structure load_data expects
    data_dict = {}
    for ch in fake_channels:
        data_dict[ch] = np.random.randn(2, 50)
    
    fakedata = np.array({"data": data_dict}, dtype=object)
    fakefreq_amp_table = np.random.randn(2, 5)
    
    # Subject ID
    subjid = "test_subject"
    
    # Save to temporary file
    np.savez(
        tmp_path / f"{subjid}.npz",
        data=fakedata,
        freq_amp_table=fakefreq_amp_table,
        latency=np.array([0]),
        channel_keys=np.array(fake_channels)
    )
    
    return str(tmp_path), subjid
    
# 1. Smoke test
# Author: Aoi Hunsaker
# Purpose: Basic identification when load_data crashes
# Reviewer: Jeff Jackson
def test_load_data_smoke(temp_eeg_data):
    my_path, subjid = temp_eeg_data
    try:
        EEGDataset = load_data(my_path, subjid)
        print(f"smoke test success for {subjid}.")
    except Exception:
        print(f"smoke test failed for {subjid}. Did not successfully load in dataset.")
        
# 2. One-shot test 
# Author: Aoi Hunsaker
# Purpose: Test a known input and output
# Reviewer: Jeff Jackson
def test_load_data_one_shot(temp_eeg_data):
    my_path, subjid = temp_eeg_data
    EEGDataset = load_data(my_path, subjid)
    print(f"One shot test: {EEGDataset.data.shape == (4,100)}")
    
# 3. Edge test
# Purpose: Properly find empty files
# Author: Aoi Hunsaker
# Reviewer: Jeff Jackson
# Create empty dataset
def test_load_data_edge():
    os.makedirs("test_data",exist_ok=True)
    np.savez("test_data/empty_data.npz",
            data=np.array([]),
            freq_amp_table=np.array([]),
            latency=np.array([0]),
            channel_keys=np.array([]),
        )

    try:
        EEGDataset = load_data("test_data","empty_data")
        if EEGDataset.data.size == 0:
            print("Edge test passed: empty dataset was found correctly")
        else:
            print("Edge test failed: empty dataset was not found to be empty")
    except Exception as e:
            print(f"Edge test failed with exception{e}")
        
# 4. Pattern test
# Author: Aoi Hunsaker
# Purpose: Make sure that load_data returns all the components I expect in EEGDataset
# Reviewer: Jeff Jackson
def test_load_data_pattern(temp_eeg_data):
    my_path, subjid = temp_eeg_data
    EEGDataset = load_data(my_path, subjid)
    expected_comps = ["data", 
                    "freq_amp_table", 
                    "latency", 
                    "channel_keys", 
                    "period_keys", 
                    "metric_keys", 
                    "submetric_keys"]
    pattern_test = all(hasattr(EEGDataset, attr) for attr in expected_comps)
    print("Pattern test:", pattern_test)