import numpy as np
from fish_eeg.data import EEGDataset


def get_channels(eegdataset: EEGDataset) -> list[str]:
    """
    Get the list of channels for a given data.

    Args:
        eegdataset: The eegdataset to get the channels from.
    Returns:
        A list of channels.

    Example:
        get_channels(eegdataset) -> ["ch1", "ch2", "ch3", "ch4"]
    """
    unique_channels = set()
    for coord, dictionary in eegdataset.data.item().items():
        for key in list(dictionary.keys()):
            if key.startswith("ch") and key[-1].isdigit():
                unique_channels.add(key)
    return list(unique_channels)


def separate_periods(eegdataset, data_attr: str = "reconstructed_ica_data"):
    """
    Separate trial data into prestim (Stim OFF) and stimresp (Stim ON) periods for each channel.

    Splits the continuous trial data into two time windows based on the dataset's
    period length and latency parameters.

    Parameters
    ----------
    eegdataset : EEGDataset
        The EEG dataset containing trial data to be separated.
    data_attr : str, optional
        Name of the dataset attribute containing the data to separate
        (default: "reconstructed_ica_data").

    Returns
    -------
    EEGDataset
        The input dataset with added "separated_by_period" key in the specified
        data attribute, containing period-separated data for each coordinate.

    Raises
    ------
    ValueError
        If the specified data_attr does not exist in the dataset.

    Notes
    -----
    The separation uses:
    - prestim period: data[:, latency : latency + period_len]
    - stimresp period: data[:, latency + period_len : latency + period_len * 2]
    """
    try:
        data = getattr(eegdataset, data_attr)  # <–– dynamically fetch attribute
    except AttributeError:
        raise ValueError(f"Invalid data attribute: {data_attr}")

    separated_data = {}

    period_len = eegdataset.period_len
    latency = eegdataset.latency

    for coord in data.keys():
        separated_dict = {"prestim": {}, "stimresp": {}}
        for period in eegdataset.period_keys:
            for channel in eegdataset.channel_keys:
                if period == "prestim":
                    separated_dict[period][channel] = data[coord][channel][
                        :, latency : latency + period_len
                    ]
                elif period == "stimresp":
                    separated_dict[period][channel] = data[coord][channel][
                        :, latency + period_len : latency + period_len * 2
                    ]
        separated_data[coord] = separated_dict

    getattr(eegdataset, data_attr)["separated_by_period"] = separated_data

    return eegdataset


def collapse_channels(eegdataset, attr: str = "reconstructed_ica_fft_output"):
    """
    Collapse channel data by vertically stacking all channels for each period.

    Combines data from all channels into a single array for each experimental
    period, facilitating cross-channel analysis.

    Parameters
    ----------
    eegdataset : EEGDataset
        The EEG dataset containing channel data to collapse.
    attr : str, optional
        Name of the dataset attribute containing the data to collapse
        (default: "reconstructed_ica_fft_output").

    Returns
    -------
    EEGDataset
        The input dataset with added "collapsed_channels" key in the specified
        attribute, containing vertically stacked channel data for each coordinate
        and period.

    Notes
    -----
    The collapsed data shape is (n_channels * n_samples, ...) where channels
    are stacked vertically using np.vstack.
    """
    data = getattr(eegdataset, attr)

    for coord in data.keys():
        collapsed_dict = {"prestim": None, "stimresp": None}
        for period in eegdataset.period_keys:
            tmp = []
            for channel in eegdataset.channel_keys:
                tmp.append(data[coord][0][period][channel])
            collapsed_dict[period] = np.vstack(tmp)
        getattr(eegdataset, attr)[coord][0]["collapsed_channels"] = collapsed_dict

    return eegdataset


def select_doub_freq_bin(data, frequencies, period_keys, myfreq, window_size=100):
    """
    Calculate double frequency magnitude and SNR from FFT data.

    Identifies the signal power at double the stimulation frequency and computes
    the signal-to-noise ratio by comparing it to surrounding frequency bins while
    excluding artifact frequencies (particularly 60 Hz and its harmonics).

    Parameters
    ----------
    data : np.ndarray or dict
        FFT magnitude data. Can be array (single period) or dict with period keys.
    frequencies : np.ndarray or dict
        Frequency vectors corresponding to the data.
    period_keys : list
        List of period identifiers (e.g., ['prestim', 'stimresp']). Empty list
        indicates single-period processing.
    myfreq : float
        Sound stimulus frequency in Hz.
    window_size : float, optional
        Frequency window size around target frequency in Hz (default: 100).

    Returns
    -------
    dict
        If period_keys is empty: Dictionary with 'doub_freq_mag' and 'SNR' arrays.
        If period_keys provided: Nested dictionary with period keys, each containing
        'doub_freq_mag' and 'SNR' arrays.

    Notes
    -----
    - Target frequency is 2 * myfreq
    - Double frequency mask uses ±3 Hz tolerance
    - Artifact frequencies (60 Hz harmonics and stimulation frequencies) are
    excluded from the background window with ±3 Hz tolerance
    - SNR is calculated as: 10 * log10(doub_mag / remain_mag)
    - Includes debug print statements for frequency masking verification
    """
    if len(period_keys) == 0:
        doub_freq_dict = {}

        target_freq = 2 * myfreq
        part_window = window_size / 2
        artifact_freqs = [
            myfreq,
            target_freq,
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
        freq_vec = frequencies[0]

        # Double frequency mask (3 Hz tolerance)
        doub_mask = np.abs(freq_vec - target_freq) <= 3
        print(doub_mask.shape)
        print(freq_vec[doub_mask])
        window_mask = (freq_vec >= target_freq - part_window) & (
            freq_vec <= target_freq + part_window
        )
        print(freq_vec[window_mask])

        for freq in artifact_freqs:
            art_mask = np.abs(freq_vec - freq) <= 3
            window_mask[art_mask] = False
        print(freq_vec[window_mask])

        num_it = data.shape[0]

        doub_freq_tmp = []
        snr_tmp = []

        for cur_it in range(num_it):
            cur_data = data[cur_it]
            doub_mag = cur_data[doub_mask]
            remain_mag = cur_data[window_mask]

            doub_mag = np.mean(doub_mag)
            remain_mag = np.mean(remain_mag)

            doub_freq_tmp.append(doub_mag)
            snr_tmp.append(10 * np.log10((doub_mag) / (remain_mag)))

        # Move this inside the period loop
        doub_freq_dict = {
            "doub_freq_mag": np.hstack(doub_freq_tmp),
            "SNR": np.hstack(snr_tmp),
        }

    elif len(period_keys) > 0:
        doub_freq_dict = {"prestim": {}, "stimresp": {}}

        target_freq = 2 * myfreq
        part_window = window_size / 2
        artifact_freqs = [
            myfreq,
            target_freq,
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
        freq_vec = frequencies["prestim"]["ch1"]
        # Double frequency mask (3 Hz tolerance)
        doub_mask = np.abs(freq_vec - target_freq) <= 3
        window_mask = (freq_vec >= target_freq - part_window) & (
            freq_vec <= target_freq + part_window
        )
        print(window_mask.shape)

        for freq in artifact_freqs:
            art_mask = np.abs(freq_vec - freq) <= 3
            window_mask[art_mask] = False

        num_it = len(data["prestim"])
        doub_mask = doub_mask[
            0
        ]  # I am doing this since freq_vec is a matrix? check this...
        print(freq_vec[0][doub_mask])
        window_mask = window_mask[0]
        print(window_mask.shape)
        print(freq_vec[0][window_mask])

        print(num_it)

        for period in period_keys:
            doub_freq_tmp = []
            snr_tmp = []
            for cur_it in range(num_it):
                cur_data = data[period][cur_it]
                doub_mag = cur_data[doub_mask]
                remain_mag = cur_data[window_mask]

                doub_mag = np.mean(doub_mag)
                remain_mag = np.mean(remain_mag)

                doub_freq_tmp.append(doub_mag)
                snr_tmp.append(10 * np.log10((doub_mag) / (remain_mag)))

            # Move this inside the period loop
            doub_freq_dict[period] = {
                "doub_freq_mag": np.hstack(doub_freq_tmp),
                "SNR": np.hstack(snr_tmp),
            }

    return doub_freq_dict


class dotdict(dict):
    """dot.notation access to dictionary attributes
    Example:
    >>> my_dict = dotdict({"key1": "value1", "key2": "value2"})
    >>> my_dict.key1
    'value1'
    >>> my_dict.key2
    'value2'
    >>> my_dict.key3
    None
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
