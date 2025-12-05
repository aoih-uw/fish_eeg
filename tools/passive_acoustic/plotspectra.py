# Import Modules

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch

# User inputs 1. List filenames for spectra 
folder_path_input = "data/PAcoustic/audio/port_townsend/";

# User inputs 2. List freq/amp pairs
subjid = 'data/hydrolagusColliei_8' # which fish

loaded = np.load(f'{subjid}_data.npz', allow_pickle=True)
freq_amp_table = loaded['freq_amp_table']
print('\nFreq/Amp used in fisheeg\n')
print(freq_amp_table)

freq_amp_table = np.unique(freq_amp_table, axis=0) # not repeating identical inputs

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
    f, psd = welch(data, fs=fs, window=window, nperseg=nperseg) # Power / Hz

    # Convert PSD to Power 
    df = f[1]-f[0] 
    power = psd * df # Power

    # Reference for SPL [dB re 1 µPa]
    pref = 1e-6

    # Convert to dB scale
    power_db = 10 * np.log10(power / (pref**2)) # dB re 1 µPa

    return f, power_db

def plot_psd(path_to_data, wav_file, f, power_db, freq_amp_table):
    """
    Plots observed ambient noise and fish EEG stimuli,
    and saves both the plot (JPG) and the raw PSD data (CSV).
    """

    # Base filename (no extension)
    base_name = os.path.splitext(os.path.basename(wav_file))[0]
    spec_file = f"{base_name}.jpg"
    csv_file  = f"{base_name}.csv"

    # ==== Create Plot ====
    plt.figure(figsize=(8, 5))
    plt.plot(f, power_db, 'b-', label='Observed Acoustic Data')

    # Add fish EEG freq/amp points
    plt.plot(freq_amp_table[:, 0], freq_amp_table[:, 1], 'ko', label='fish_eeg')

    plt.xlim((0, 2000))
    plt.title(base_name)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('SPL [dB re 1 µPa]')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.tight_layout()
    plot_path = os.path.join(path_to_data, spec_file)
    plt.savefig(plot_path, dpi=300, format='jpg')
    plt.close()

    # ==== Save PSD Data to CSV ====
    csv_path = os.path.join(path_to_data, csv_file)
    df = pd.DataFrame({
        "Frequency_Hz": f,
        "Power_dB": power_db
    })
    df.to_csv(csv_path, index=False)

    print(f"Saved plot: {plot_path}")
    print(f"Saved CSV:  {csv_path}")

    return spec_file


# Get relative path for dir
current_dir = os.path.dirname(__file__)
path_to_data = os.path.join(current_dir, '..','..', folder_path_input)
path_to_data = os.path.abspath(path_to_data)

# Loop through folder to find files
wav_files = [f for f in os.listdir(path_to_data) if f.lower().endswith('.wav')]
if not wav_files:
    print("No .wav files found in", path_to_data)

print('\nWav Files in Ambient Acoustic Data\n')
print(wav_files)

# Plotting loop
for wav_file in wav_files:
    filepath = os.path.join(path_to_data, wav_file)
    print(f"Processing: {wav_file}")

    f, power_db = spec_wav_psd(filepath, nperseg=1024, window='hann')

    plot_psd(path_to_data, wav_file, f, power_db, freq_amp_table)



    


