from fish_eeg.data import EEGDataset
from fish_eeg.utils import get_channels
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA


class Denoiser:
    def __init__(self, eegdataset: EEGDataset):
        self.eegdataset = eegdataset
        self.data = self.eegdataset.bandpass_data
        self.channels = get_channels(eegdataset)
        self.period_keys = eegdataset.period_keys

    def ICA(self, dictionary: dict):
        def reshape_the_data(data, bootstrapped=False):
            # Reshape to (trials * samples) x channels
            reshaped_list = []
            for channel in self.channels:
                cur_set = data[channel]
                reshaped_list.append(cur_set.reshape(-1, 1))

            reshaped_data = np.hstack(reshaped_list)
            return reshaped_data

        def perform_ICA(data):
            start = time.perf_counter()

            # Standardize the data
            # Removes the mean and sets variance across ICs to 1, necessary for ICA to work
            # Standardization of a dataset is a common requirement for many machine learning estimators:
            # they might behave badly if the individual features do not
            # more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # Improved FastICA configuration
            # Use default variables more or less except max_iter set to 500 instead of 200
            ica = FastICA(
                n_components=len(self.channels),
                random_state=42,  # More robust random seed
                max_iter=500,  # Increased iterations
                tol=1e-4,  # Tightened tolerance???
                whiten="unit-variance",  # Corrected whiten parameter
            )

            S = ica.fit_transform(data_scaled)
            A = ica.mixing_

            end = time.perf_counter()
            elasped = end - start

            # Convergence check
            print(f"Elapsed time: {elasped:.2f} seconds")
            print(f"ICA Convergence: {ica.n_iter_}")

            # Store results in dictionary
            ica_results = {
                "S": S,
                "A": A,
                "n_iter": ica.n_iter_,
                "elapsed_time": elasped,
            }

            return ica_results

        reshaped_data = reshape_the_data(dictionary)
        ica_results = perform_ICA(reshaped_data)
        return ica_results

    def your_new_denoiser(self):
        pass

    def pipeline(self):
        ica_dictionary = {}
        for coord, dictionary in self.data.item().items():
            ica_results = self.ICA(dictionary)
            ica_dictionary[coord] = ica_results
        self.eegdataset.ica_output = ica_dictionary
        return self.eegdataset
