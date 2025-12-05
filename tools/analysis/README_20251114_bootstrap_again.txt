================================================================================
    AUDITORY EVOKED POTENTIAL (AEP) ANALYSIS PIPELINE
================================================================================

OVERVIEW
--------
This pipeline analyzes auditory evoked potentials (AEPs) in Hydrolagus colliei 
(ratfish) by processing multi-channel electrophysiological recordings. The 
analysis focuses on detecting the double-frequency response to acoustic stimuli 
using ICA-based denoising, bootstrapping, and SNR calculations.

KEY FEATURES
------------
- Multi-channel electrophysiological data processing from MATLAB .mat files
- Artifact rejection based on RMS thresholds
- Bandpass filtering for noise reduction
- Independent Component Analysis (ICA) for signal denoising
- Bootstrap analysis for statistical confidence
- FFT-based frequency domain analysis
- SNR calculation at double-frequency response
- Automated visualization and result export

REQUIREMENTS
------------
- pandas
- numpy
- altair
- scipy
- matplotlib
- scikit-learn
- pywt
- h5py
- pickle

WORKFLOW
--------

1. DATA LOADING
   
   subjid = 'hydrolagusColliei_4'
   data, freq_amp_table = read_data(subjid)
   
   - Reads .mat files from specified directory
   - Extracts trials organized by frequency and amplitude
   - Returns dictionary organized by (frequency, amplitude) keys

2. DATA CLEANING
   
   cleaned_data = remove_artefacts(data)
   
   - Removes trials with excessive RMS (>3 standard deviations)
   - Balances trial counts across channels

3. STIMULUS SELECTION
   
   myfreq = 360  # Hz
   myamp = 135   # dB
   dataset_index = 0
   current_cond = select_stimulus(cleaned_data, myfreq, myamp, dataset_index)

4. BANDPASS FILTERING
   
   low, high = 70, 1400  # Hz
   filt_data = bandpass(current_cond, low, high, fs)
   
   - Removes frequencies outside auditory range of interest

5. ICA DENOISING
   
   reshaped_data = reshape_the_data(filt_data, channel_keys, period_keys)
   ica_results = perform_ICA(reshaped_data, channel_keys)
   
   - Separates independent signal components
   - Identifies components with strongest double-frequency response
   - Weights components by SNR for reconstruction

6. PERIOD SEPARATION
   
   separated_data = separate_periods(recon_restruct_data, current_cond, 
                                     period_keys, channel_keys, latency)
   
   - Splits data into prestim (stimulus OFF) and stimresp (stimulus ON) periods
   - Accounts for system latency (2118 samples at 22050 Hz)

7. BOOTSTRAP ANALYSIS
   
   collapsed_dict = collapse_channels(weighted_ffts, period_keys, channel_keys)
   bootstrap_means, bootstrap_stds = calculate_bootstrap(collapsed_dict, 
                                                         period_keys, 
                                                         channel_keys, 
                                                         n_iterations=100)
   
   - Generates 100 resampled estimates of FFT magnitude
   - Creates distribution for statistical comparison

8. STATISTICAL COMPARISON
   
   doub_freq_dict = select_doub_freq_bin(bootstrap_means, weighted_freq_vec, 
                                         period_keys, myfreq, window_size=100)
   diff_CI_results = calculate_diff_CI(doub_freq_dict, 'SNR')
   
   - Calculates SNR at double-frequency (2x stimulus frequency)
   - Uses 95% confidence interval on bootstrapped difference
   - If CI excludes 0, response is significant

OUTPUT FILES
------------

Plots (saved to plots/{subjid}/)
- *_compare_ICA_denoised_waveform_*.png - Time domain comparison
- *_compare_ICA_denoised_fft_*.png - Frequency domain comparison
- *_grand_avg_comparison_*.png - Bootstrap mean +/- SD by period
- *_2x_resp_hist_magnitude_*.png - SNR distribution histograms

Data Files
- weights_csv/{subjid}/*_weights_*.csv - ICA component weights per channel
- CI_csv/{subjid}/*_CI_*.csv - 95% confidence intervals for SNR difference

KEY PARAMETERS
--------------
Parameter               Value           Description
--------------------------------------------------------------------------------
fs                      22050 Hz        Sampling frequency
latency                 2118 samples    System delay (~96 ms)
channels                ch1-ch4         Recording electrode channels
artifact_threshold      mean + 3xSD     RMS rejection criterion
bandpass                70-1400 Hz      Filter range
bootstrap_n             100             Number of resamples
SNR_window              +/-100 Hz       Frequency window around 2f

INTERPRETATION
--------------
- Significant response: 95% CI of (stimresp - prestim) SNR excludes 0
- Component weights: Higher weight = more stable double-frequency response
- SNR: Signal power at 2f relative to surrounding noise floor (in dB)

NOTES
-----
- All frequency analysis focuses on the SECOND HARMONIC (2f) of the stimulus
- Artifact frequencies (60 Hz harmonics, fundamental, and 2f) are excluded 
  from noise floor calculation
- ICA denoising preserves phase relationships while reducing noise
- Bootstrap provides non-parametric confidence intervals suitable for 
  non-normal distributions

AUTHOR
------
Aoi Hunsaker
Contributions by: Andrew Brown, Joseph Sisneros, Michael James
================================================================================