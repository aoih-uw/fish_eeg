# fish_eeg
Let's use different analysis and visualization tools to find interesting patterns of responses in my fish EEG dataset.

# Authors

**Aoi Hunsaker**
University of Washington

Psychology - PhD Student  - adv. Joseph Sisneros; Andrew Brown

www.sisneroslab.org

[aoih@uw.edu](mailto:aoih@uw.edu)

**Michael James**

University of Washington

Civil and Environmental Engineering - PhD Student  - adv. Jim Thomson; Kristin Zeiden

[Environmental Fluid Mechanics Group](http://depts.washington.edu/uwefm/wordpress/)

[mkj29@uw.edu](mailto:mkj29@uw.edu)

# Information about "data" structure

Data is a collection of python dictionaries.

data.item().keys() will tell you what stimulus combinations we have data for (e.g., 100 Hz, 115 dB)

You can access the data for a single stimulus combination like this:
data.item()[100,115]

Included in a single stimulus combination dictionary are the following:
1. filename: Which file did the data come from?
2. decision: ignore for now
3. period_len: ignore for now
4. ch1, ch2, ch3, ch4: The actual electrode signals from 4 separate electrodes
	shape = (number of trials, number of samples) 
