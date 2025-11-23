================================================================================
FISH EEG ANALYSIS AND VISUALIZATION
================================================================================

OVERVIEW
--------
This script analyzes and visualizes EEG data from fish subjects (Hydrolagus 
colliei - spotted ratfish). It processes confidence interval data and ICA 
(Independent Component Analysis) weights to understand how fish respond to 
auditory stimuli at different frequencies and amplitudes.

REQUIREMENTS
------------
- Python 3.x
- pandas
- numpy
- altair
- vl-convert (for saving PNG files)
- os

INPUT DATA
----------
The script reads two types of CSV files:

1. Confidence Interval (CI) Data:
   Location: /home/sphsc/cse583/fish_eeg/analysis/CI_csv/{subjid}/
   Format: Files named "CI_*.csv" containing:
   - freq: Stimulus frequency (Hz)
   - amp: Stimulus amplitude
   - lower_CI: Lower bound of 95% confidence interval
   - upper_CI: Upper bound of 95% confidence interval

2. ICA Weights Data:
   Location: /home/sphsc/cse583/fish_eeg/analysis/weights_csv/{subjid}/
   Format: Files named "weights_*.csv" containing:
   - freq: Stimulus frequency (Hz)
   - amp: Stimulus amplitude
   - channel: EEG channel identifier
   - weight: ICA weight value

SUBJECTS
--------
The analysis includes 7 subjects:
- deadFish (control/baseline)
- hydrolagusColliei_4
- hydrolagusColliei_5
- hydrolagusColliei_6
- hydrolagusColliei_7
- hydrolagusColliei_8
- hydrolagusColliei_9

Note: The deadFish subject and certain problematic data points are filtered 
out during analysis.

DATA PROCESSING
---------------
1. Data Loading:
   - Reads all CSV files for each subject
   - Combines data from multiple tests per subject
   - Adds subject ID and test ID to each record

2. Data Filtering:
   - Removes frequencies 50 Hz and 385 Hz (likely artifacts)
   - Removes amplitudes 145 and 150 (problematic data)
   - Excludes 55 Hz and 100 Hz data for hydrolagusColliei_7
   - Excludes all deadFish data from visualizations

3. Calculations:
   - Computes midpoint of confidence intervals: (CI_lower + CI_upper) / 2
   - Calculates median CIs across subjects for each frequency/amplitude pair

VISUALIZATIONS
--------------

1. 95% Confidence Intervals - All Data (95CI_all.html/png):
   - Faceted by stimulus frequency
   - X-axis: Stimulus amplitude (70-145)
   - Y-axis: 95% confidence interval values
   - Shows:
     * Shaded area representing CI range
     * Line connecting midpoints
     * Circles at data points
     * Zero reference line
   - Interactive tooltips show subject ID, test ID, and values

2. Median 95% Confidence Intervals (95CI_median.html/png):
   - Same layout as above
   - Shows median CI values across all subjects
   - Useful for identifying general trends without individual variation

3. ICA Weights (ICAweights.html/png):
   - Faceted by stimulus frequency
   - X-axis: Stimulus amplitude (70-145)
   - Y-axis: ICA weight values
   - Color-coded by EEG channel
   - Hollow circles show individual measurements
   - Helps identify which channels contribute most to responses

OUTPUT FILES
------------
All visualizations are saved to:
/home/sphsc/cse583/fish_eeg/analysis/plots/summary_plots/

Files generated:
- 95CI_all.html (interactive)
- 95CI_all.png (high resolution, 3x scale factor)
- 95CI_median.html (interactive)
- 95CI_median.png (high resolution, 3x scale factor)
- ICAweights.html (interactive)
- ICAweights.png (high resolution, 3x scale factor)

INTERPRETATION
--------------
- Positive CI midpoints suggest neural response to stimulus
- Larger confidence intervals indicate more variability in response
- Different frequencies may show different response patterns
- ICA weights reveal which channels capture the stimulus response
- Amplitude effects show how response strength changes with stimulus intensity

USAGE
-----
Simply run the script in a Python environment with all dependencies installed:
    python script_name.py

Or execute cells sequentially in a Jupyter notebook environment.

NOTES
-----
- The script uses Altair for visualization, which creates interactive charts
- PNG exports require the vl-convert engine
- Display settings show up to 50 rows when printing dataframes
- All charts use the 'viridis' color scheme for frequency encoding
- Charts are interactive when viewed in HTML format

AUTHOR
-------
Aoi Hunsaker
Contributions by: Andrew Brown, Joseph Sisneros, Michael James

================================================================================