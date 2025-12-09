import numpy as np
from fish_eeg.data import ConfigAccessor, EEGDataset


class Bootstrap:
    def __init__(self, eegdataset: EEGDataset, cfg: ConfigAccessor | None = None):
        self.eegdataset = eegdataset
        self.data = eegdataset.reconstructed_ica_fft_output
        self.period_keys = eegdataset.period_keys
        self.channel_keys = eegdataset.channel_keys
        self.cfg = cfg.get("statistics", "params", default=ConfigAccessor(None))

    def calculate_bootstrap(
        self, data, period_keys, channel_keys, n_iterations=100, seed=42
    ):
        # Ensure consistent random sampling across iterations
        rng = np.random.default_rng(
            self.cfg.get("statistics", "seed", default=seed)
        )  # create a standalone random number generator object
        # Determine the maximum number of samples across periods
        max_samples = max(len(data[period]) for period in period_keys)

        boot_means = {period: [] for period in period_keys}
        boot_std = {period: [] for period in period_keys}

        for _ in range(
            self.cfg.get("statistics", "n_iterations", default=n_iterations)
        ):
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
