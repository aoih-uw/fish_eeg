import pytest
import numpy as np

from fish_eeg.preprocess import Preprocessor
from fish_eeg.denoisers import Denoiser
from fish_eeg.fft import FFT
from fish_eeg.filters import Filter

def test_fft_smoke(fake_dataset):
    ds = fake_dataset()
    # Known structure from fixture
    n_channels = 4    
    n_samples = 50
    # Create fake ICA output expected by FFT
    ds.ica_output = {
        "stim": {
            "S": np.random.randn(n_samples, n_channels)
        }
    } #Not testing ICA, define ica output so it runs

    try:
        fft = FFT(ds)
        sampling_frequency = 1000
        ds = fft.pipeline(sampling_frequency)
    except Exception as e:
        pytest.fail(f"FFT pipeline raised an exception: {e}")

def test_fft_differentHz_per_channel(sinusoid_dataset):
    """
    Check that channels ch1-ch4 have frequencies 20, 30, 40, 50 Hz
    param sinusoid_dataset: a fixture in conftest.py that creates a sine wave for each trial,
    with 20Hz, 30Hz, 40Hz, 50Hz per increasing channel. (Ch1 -> 20Hz)
    """
    ds = sinusoid_dataset(different_trials=False)
    channel_keys = ["ch1", "ch2", "ch3", "ch4"]
    expected_freqs = [20, 30, 40, 50]

    # Extract data dict
    first_coord = list(ds.data)[0]        
    data_dict = ds.data[first_coord]      

    detected_freqs = []
    sampling_frequency = 1000

    for idx, ch in enumerate(channel_keys):
        # Prepare ICA-like output for just this channel, required for pipeline method in fft.py
        ds.ica_output = {
            "stim": {
                "S": data_dict[ch].T  
            }
        }

        fft = FFT(ds)
        ds = fft.pipeline(sampling_frequency)

        # FFT output
        magnitudes, frequencies = ds.ica_fft_output["stim"]

        # We only need to compare first trial since all trials are identical for testing purposes
        peak_idx = np.argmax(magnitudes[0])
        detected_freqs.append(frequencies[0][peak_idx])

    # Assert that each channel peak matches expected
    assert np.allclose(detected_freqs, expected_freqs, atol=1.0)