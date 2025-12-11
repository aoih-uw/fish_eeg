#### Constants ####
"""
Constants used across the analysis pipeline.

This module defines keys for preprocessing and metrics, the sampling
frequency of the recordings, latency correction values, and the list
of line-noise–related baseline artifact frequencies to remove.

Variables
---------
PERIOD_KEYS : list of str
    Labels for the prestimulus and stimulus-response segments.
CHANNEL_KEYS : list of str
    Names of the recorded channels.
METRIC_KEYS : list of str
    Summary metrics computed for each segment.
SUBMETRIC_KEYS : list of str
    Additional frequency-domain statistical metrics.
sampling_frequency : int
    EEG recording sample rate in Hz.
latency : int
    Latency correction (in samples) to ensure that stim OFF and stim ON periods are separated correctly.
baseline_artifact_freqs : list of int
    Harmonics of 60 Hz to be removed during filtering.
"""

PERIOD_KEYS = ["prestim", "stimresp"]
CHANNEL_KEYS = ["ch1", "ch2", "ch3", "ch4"]
METRIC_KEYS = ["mean", "std"]
SUBMETRIC_KEYS = ["doub_freq_mag", "SNR"]
sampling_frequency = 22050
latency = 2118

baseline_artifact_freqs = [
    60,
    120,
    180,
    240,
    300,
    360,
    420,
    480,
    540,
    600,
    660,
    720,
    780,
    840,
    900,
]
