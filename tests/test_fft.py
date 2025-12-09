import pytest
import numpy as np

from fish_eeg.fft import FFT

def test_fft_smoke(fake_dataset):
    ds = fake_dataset()
    # Known structure from fixture
    n_channels = 4    # e.g., 4
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