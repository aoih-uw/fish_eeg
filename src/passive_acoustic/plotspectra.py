# Import Modules

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch


# User inputs 1. List filenames for spectra 2. List freq/amp pairs
folder_path_input = "data/PAcoustic/audio/port_townsend/";

# Function to make spectra from .wav file (load and convert)
def spec_wav_psd(filename, nperseg=1024, window='hann'):
    """
    Reads a .wav file, computes the Power Spectral Density (PSD)

    Parameters
    ----------
    filename : str
        Path to the .wav file.
    nperseg : int, optional
        Length of each segment for Welch’s method (default: 1024).
    window : str, optional
        Window type for Welch’s method (default: 'hann').

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    psd_db : ndarray
        Power spectral density in decibels (dB/Hz).
    """
    # Read the WAV file
    fs, data = wavfile.read(filename)

    # If stereo, convert to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Compute Power Spectral Density using Welch’s method
    f, psd = welch(data, fs=fs, window=window, nperseg=nperseg)

    # Convert to dB scale
    psd_db = 10 * np.log10(psd)

    return f, psd_db

# Get relative path for dir
current_dir = os.path.dirname(__file__)
path_to_data = os.path.join(current_dir, '..','..', folder_path_input)
path_to_data = os.path.abspath(path_to_data)

# Loop through folder to find files
wav_files = [f for f in os.listdir(path_to_data) if f.lower().endswith('.wav')]
if not wav_files:
    print("No .wav files found in", path_to_data)

print(wav_files)

# Plotting loop
for wav_file in wav_files:
    filepath = os.path.join(path_to_data, wav_file)
    print(f"Processing: {wav_file}")

    f, psd = spec_wav_psd(filepath, nperseg=nperseg, window=window)



    # save_path = os.path.join(path_to_data, )
    # plt.savefig(save_path, dpi=300)
    # print(f"Saved combined PSD plot to {save_path}")



    


