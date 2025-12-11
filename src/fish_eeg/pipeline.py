from fish_eeg.constants import sampling_frequency
from fish_eeg.data import load_data, read_yaml_config
from fish_eeg.preprocess import Preprocessor
from fish_eeg.denoisers import Denoiser
from fish_eeg.filters import Filter
from fish_eeg.plotting import Plotter
from fish_eeg.fft import FFT
from fish_eeg.reconstruct import Reconstructor
from fish_eeg.utils import (
    separate_periods,
    dotdict,
)
from fish_eeg.statistics import Bootstrap
import logging
import argparse
from fish_eeg.utils import collapse_channels
import os

"""
Main pipeline script for EEG data processing and analysis.

This script orchestrates the complete EEG analysis workflow including:
preprocessing, filtering, denoising with ICA, FFT analysis, signal
reconstruction, period separation, and statistical bootstrapping.

Usage:
    python main.py --config_path path/to/config.yaml
"""

def main(config_path: str):
    """
    Execute the complete EEG processing pipeline.

    Orchestrates the full analysis workflow from raw data loading through
    preprocessing, filtering, ICA denoising, FFT analysis, reconstruction,
    period separation, and bootstrap statistics. Generates diagnostic plots
    at each major stage.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file containing pipeline parameters,
        dataset paths, and processing options.

    Returns
    -------
    None
        Results are saved to disk and plotted according to configuration.

    Pipeline Steps
    --------------
    1. Load configuration and dataset
    2. Artefact rejection
    3. Bandpass filter
    4. ICA denoising
    5. FFT analysis on ICA components
    6. Reconstruct signals from ICA
    7. Separate into prestim/stimresp periods
    8. FFT analysis on reconstructed data
    9. Collapse channels and compute bootstrap statistics

    Notes
    -----
    Plots are saved to subdirectories under the configured save_path for each
    processing stage. The final eegdataset can optionally be pickled (currently
    commented out).

    Raises
    ------
    Exception
        Any errors during pipeline execution are logged and re-raised.
    """
    logging.info(f"Starting pipeline with config path: {config_path}")

    # read the config file
    config = read_yaml_config(config_path)
    pipeline_config = dotdict(config)

    logging.info("Loaded pipeline configuration successfully")

    path = pipeline_config.dataset.get("path")
    subjid = pipeline_config.dataset.get("subjid")
    eegdataset = load_data(path, subjid)

    logging.info(f"Loaded data successfully from {path} for subject {subjid}")

    preprocessor = Preprocessor(eegdataset, pipeline_config)
    eegdataset = preprocessor.pipeline()
    logging.info("Preprocessed data successfully")
    #### big plot for all freqs and amps for a global view
    Plotter(eegdataset).plot_waveforms_by_frequency_rows(
        attr="rms_subsampled_data",
        channel_keys=eegdataset.channel_keys,
        save_path=pipeline_config.save_path + f"/{subjid}/preprocessed_data",
    )

    filters = Filter(eegdataset, pipeline_config)
    eegdataset = filters.pipeline()
    logging.info("Filtered data successfully")
    #### big plot for all freqs and amps for a global view
    Plotter(eegdataset).plot_waveforms_by_frequency_rows(
        attr="bandpass_data",
        channel_keys=eegdataset.channel_keys,
        save_path=pipeline_config.save_path + f"/{subjid}/bandpass_data",
    )

    denoiser = Denoiser(eegdataset, pipeline_config)
    eegdataset = denoiser.pipeline()
    logging.info("Denoised data successfully")
    #### big plot for all freqs and amps for a global view
    Plotter(eegdataset).plot_waveforms_by_frequency_rows(
        attr="ica_output",
        channel_keys=[],
        num_samples=[],
        save_path=pipeline_config.save_path + f"/{subjid}/ICA_plots",
    )

    fft = FFT(eegdataset, pipeline_config)
    eegdataset = fft.pipeline(sampling_frequency)
    logging.info("FFT data successfully")
    #### big plot for all freqs and amps for a global view
    Plotter(eegdataset).plot_fft_all_ica(
        attr="ica_fft_output",
        channel_keys=[],
        period_keys=[],
        save_path=pipeline_config.save_path + f"/{subjid}/FFT_on_ICA",
        subjid=subjid,
        dataset_index=0,
    )

    reconstructor = Reconstructor(eegdataset, pipeline_config)
    eegdataset = reconstructor.pipeline()
    logging.info("Reconstructed data successfully")
    Plotter(eegdataset).plot_compare_denoised_by_frequency_rows(
        attr="reconstructed_ica_data"
    )
    Plotter(eegdataset).plot_compare_denoised_fft_by_frequency_rows()

    eegdataset = separate_periods(eegdataset)
    fft = FFT(eegdataset, pipeline_config)
    eegdataset = fft.pipeline(sampling_frequency)
    logging.info("Separated periods successfully")
    #### big plot for all freqs and amps for a global view
    Plotter(eegdataset).plot_fft_all_ica(
        attr="reconstructed_ica_fft_output",
        channel_keys=eegdataset.channel_keys,
        period_keys=eegdataset.period_keys,
        save_path=pipeline_config.save_path + f"/{subjid}/reconstructedFFT",
        subjid=subjid,
        dataset_index=0,
    )

    eegdataset = collapse_channels(eegdataset)
    statistics = Bootstrap(eegdataset, pipeline_config)
    eegdataset = statistics.pipeline()
    logging.info("Statistics calculated successfully")

    save_path = os.path.join(  # noqa: F841
        pipeline_config.save_path, f"{subjid}/final_eegdataset.pkl"
    )

    # with open(save_path, "wb") as f:
    #     pickle.dump(eegdataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    return None


def parse_args():
    """
    Parse command-line arguments for the pipeline script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing:
        - config_path : str
            Path to the YAML configuration file.

    Examples
    --------
    Run the pipeline with a config file:
        python main.py --config_path configs/experiment1.yaml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        main(args.config_path)
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        raise e
    logging.info("Pipeline completed successfully")
