import numpy as np
from dataclasses import dataclass
from fish_eeg.data import load_data
import os
import pytest

# Check if running in CI environment
IN_CI = os.getenv('CI') is not None

# 1. Smoke test
# Author: Aoi Hunsaker
# Purpose: Basic identification when load_data crashes
# Reviewer:
@pytest.mark.skipif(IN_CI, reason="Skipping test that requires large data files not available in CI") 
def test_load_data_smoke():
    my_path = "/home/sphsc/cse583/fish_eeg/analysis/"
    subjid = "hydrolagusColliei_5"
    try:
        EEGDataset = load_data(my_path,subjid)
        print(f"smoke test success for {subjid}.")
    except Exception:
        print(f"smoke test failed for {subjid}. Did not successfully load in dataset.")
        
# 2. One-shot test 
# Author: Aoi Hunsaker
# Purpose: Test a known input and output
# Reviewer:
@pytest.mark.skipif(IN_CI, reason="Skipping test that requires large data files not available in CI")
def test_load_data_one_shot():
    my_path = "/home/sphsc/cse583/fish_eeg/analysis/"
    subjid = "hydrolagusColliei_5"
    EEGDataset = load_data(my_path,subjid)
    print(f"One shot test: {EEGDataset.data.shape == (4,100)}") # Old format of dataset had this dimension, I want to identify these old datasets since I need to treat them differently

# 3. Edge test
# Purpose: Properly find empty files
# Author: Aoi Hunsaker
# Reviewer:
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
# Reviewer:
@pytest.mark.skipif(IN_CI, reason="Skipping test that requires large data files not available in CI")
def test_load_data_pattern():
    my_path = "/home/sphsc/cse583/fish_eeg/analysis/"
    subjid = "hydrolagusColliei_5"
    EEGDataset = load_data(my_path,subjid)
    expected_comps = ["data", 
                    "freq_amp_table", 
                    "latency", 
                    "channel_keys", 
                    "period_keys", 
                    "metric_keys", 
                    "submetric_keys"]
    pattern_test = all(hasattr(EEGDataset, attr) for attr in expected_comps)
    print("Pattern test:", pattern_test)