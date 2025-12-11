import numpy as np
from fish_eeg.data import EEGDataset
from fish_eeg.utils import dotdict


class Bootstrap:
    """
    Perform bootstrap resampling analysis on reconstructed ICA FFT data.

    This class implements paired bootstrap resampling to estimate confidence
    intervals for EEG signal statistics across experimental periods.

    Parameters
    ----------
    eegdataset : EEGDataset
        The EEG dataset containing reconstructed ICA FFT outputs.
    data : dict
        Reconstructed ICA FFT output data from the dataset.
    period_keys : list
        List of experimental period identifiers (e.g., ['prestim', 'stimresp']).
    channel_keys : list
        List of channel identifiers.
    cfg : ConfigAccessor
        Configuration accessor for bootstrap parameters.
    """
    def __init__(self, eegdataset: EEGDataset, cfg: ConfigAccessor | None = None):
        """
        Initialize the Bootstrap analyzer with EEG dataset and configuration.

        Parameters
        ----------
        eegdataset : EEGDataset
            The EEG dataset containing reconstructed ICA FFT outputs to bootstrap.
        cfg : ConfigAccessor | None, optional
            Configuration accessor for bootstrap parameters. If None, uses default
            configuration (default: None).
        """
        self.eegdataset = eegdataset
        self.data = eegdataset.reconstructed_ica_fft_output
        self.period_keys = eegdataset.period_keys
        self.channel_keys = eegdataset.channel_keys
        cfg = cfg or dotdict({})  # if None, use empty
        if not isinstance(cfg, dotdict):
            cfg = dotdict(cfg)
        statistics_cfg = cfg.get("statistics", dotdict({}))
        self.cfg = statistics_cfg.get("params", dotdict({}))

    def calculate_bootstrap(
        self, data, period_keys, channel_keys, n_iterations=100, seed=42
    ):
        """
        Perform paired bootstrap resampling on data across experimental periods.

        Uses the same random sample indices across all periods to maintain paired
        relationships between conditions. Computes bootstrap distributions of means
        and standard deviations.

        Parameters
        ----------
        data : dict
            Dictionary with period keys mapping to data arrays of shape (n_samples, ...).
        period_keys : list
            List of period identifiers to bootstrap.
        channel_keys : list
            List of channel identifiers (currently unused in implementation).
        n_iterations : int, optional
            Number of bootstrap iterations to perform (default: 100).
        seed : int, optional
            Random seed for reproducibility (default: 42).

        Returns
        -------
        tuple
            Two-element tuple containing:
            - boot_means : dict
                Dictionary with period keys mapping to lists of bootstrap mean arrays.
                Each list has length n_iterations.
            - boot_std : dict
                Dictionary with period keys mapping to lists of bootstrap std arrays.
                Each list has length n_iterations.

        Notes
        -----
        The method uses paired bootstrap by generating a single set of sample indices
        that is applied consistently across all periods, preserving relationships
        between conditions.
        """
        # Ensure consistent random sampling across iterations
        rng = np.random.default_rng(
            self.cfg.get("seed", seed)
        )  # create a standalone random number generator object
        # Determine the maximum number of samples across periods
        max_samples = max(len(data[period]) for period in period_keys)

        boot_means = {period: [] for period in period_keys}
        boot_std = {period: [] for period in period_keys}

        for _ in range(self.cfg.get("n_iterations", n_iterations)):
            # Generate a single set of indices to use for all periods (i.e., paired bootstrap)
            sample_indices = rng.choice(max_samples, size=max_samples, replace=True)

            for period in period_keys:
                cur_data = data[period]

                # Resample using the same indices for each period
                resampled_data = cur_data[sample_indices]

                boot_means[period].append(np.mean(resampled_data, axis=0))
                boot_std[period].append(np.std(resampled_data, axis=0))

        return boot_means, boot_std

    def pipeline(self, n_iterations=100, random_state=42):
        """
        Execute the bootstrap analysis pipeline for all coordinates in the dataset.

        Processes each coordinate (frequency, amplitude pair) by performing bootstrap
        resampling on collapsed channel data and computing bootstrap distributions.

        Parameters
        ----------
        n_iterations : int, optional
            Number of bootstrap iterations to perform (default: 100).
        random_state : int, optional
            Random seed for reproducibility (default: 42).

        Returns
        -------
        EEGDataset
            The input dataset with added 'bootstrap_data' attribute containing
            bootstrap means and standard deviations for all coordinates and periods.

        Notes
        -----
        The method extracts collapsed channel data from the reconstructed ICA FFT
        output and applies bootstrap resampling to generate confidence intervals.
        """
        bootstrap_data = {}
        for coord in self.data.keys():
            data = self.data[coord][0]["collapsed_channels"]
            boot_means, boot_std = self.calculate_bootstrap(
                data,
                self.period_keys,
                self.channel_keys,
                n_iterations,
                random_state,
            )
            bootstrap_data[coord] = {
                "bootstrap_means": boot_means,
                "bootstrap_std": boot_std,
            }

        self.eegdataset.bootstrap_data = bootstrap_data

        return self.eegdataset
