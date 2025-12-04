import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import welch
import matplotlib.pyplot as plt

# ===== SETTINGS =====
folders = [
    "data/PAcoustic/audio/port_townsend",
    "data/PAcoustic/audio/orcasound_lab/rpi"
]
folder_labels = ["Port Townsend", "San Juan Island"]
audiogram_csv = "results/aoi/ratfish_auditory_thresholds.csv"
f_min, f_max = 0, 500
f_step = 5   # PSD resolution

# ===== Function: compute Power Spectra re SPL =====
def get_psd(filename, nperseg=1024):
    fs, data = wavfile.read(filename)
    nperseg = fs/5
    if data.ndim > 1:
        data = data.mean(axis=1)
    f, psd = welch(data, fs=fs, nperseg=nperseg)
    df = f[1] - f[0]
    power = psd * df
    pref = 1e-6
    power_db = 10 * np.log10(power / pref**2)
    return f, power_db

# ===== Load audiogram =====
audiogram = pd.read_csv(audiogram_csv)
freq_aud = audiogram['Stimulus Frequency'].values
amp_aud = audiogram['Auditory Threshold (dB)'].values

# ===== f bins =====
f_target = np.arange(f_min, f_max + f_step, f_step)
Nf = len(f_target)

# ===== Storage dictionaries =====
mean_dict = {}
std_dict = {}

# ===== Process each folder =====
for folder, label in zip(folders, folder_labels):
    wav_files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
    wav_files = sorted(wav_files)
    print(f"\nFolder '{label}': found {len(wav_files)} WAV files.")

    # Storage matrix
    P = np.zeros((len(wav_files), Nf))

    # Loop through WAV files
    for i, fname in enumerate(wav_files):
        full_path = os.path.join(folder, fname)
        f, p_db = get_psd(full_path)

        # restrict to 0–500 Hz
        mask = (f >= f_min) & (f <= f_max)
        f = f[mask]
        p_db = p_db[mask]

        # Match to 5 Hz bins
        p_interp = np.interp(f_target, f, p_db)
        P[i, :] = p_interp

    # Compute mean and std
    mean_dict[label] = P.mean(axis=0)
    std_dict[label] = P.std(axis=0)

# Plot
plt.figure(figsize=(10, 6))
colors = ["blue", "green"]

for i, label in enumerate(folder_labels):
    mean_psd = mean_dict[label]
    std_psd = std_dict[label]
    plt.plot(f_target, mean_psd, color=colors[i], linewidth=2, label=f"{label} Mean Power")
    plt.fill_between(f_target, mean_psd - std_psd, mean_psd + std_psd, color=colors[i], alpha=0.3, label=f"{label} ±1 std")

# Audiogram (native points)
plt.plot(freq_aud, amp_aud, 'r--o', linewidth=2, markersize=6, label="Proven Ratfish Hearing")

plt.xlabel("Frequency (Hz)")
plt.ylabel("SPL (dB re 1 μPa)")
plt.title("Ambient Noise in Puget Sound Locations vs Preliminary Audiogram")
plt.grid(True)
plt.legend()
plt.xlim([0, 500])
plt.tight_layout()
plt.savefig("tools/analysis/audiogram_and_Puget_Sound.png", dpi=300)
plt.show()

print("\nSaved:\n  src/analysis/audiogram_and_Puget_Sound.png")
