# fish_eeg
## Website Link
https://aoih-uw.github.io/fish_eeg/

## Description
Let's use different analysis and visualization tools to find interesting patterns of responses in my fish EEG dataset.

## Authors

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

## Information about "data" structure

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

## Repository Folder Structure

### src 
"source code" 
Contains all code for running programs. Each subdir represents a different process and contains a README for functions specific to that process. 

### deps
"dependencies" (.gitignored)
Storage location for auxillary programs that are not contained within conda or python. 

### data 
"data" (.gitignored)
Location for users to store their data used in this repo. 

### docs 
"documentation"
Location of pdfs and other documents which provide contextual information to the repository. 

### analysis
"analysis"
Location of jupyter notebooks which explore different analysis methods (not to be used in production)

## results
Final location of src products and project deliverables

# Initializing anaconda environment

For each type of script in this repo that is a python script, there are different modules that need to be installed. This can be done through anaconda or via "PIP". We tend to opt for anaconda because the modules can be stored in an "environment", where each conda environment has the right modules and versions for the job. In this folder we have two files:

**environment.yml** - lists out the modules in the conda environment which will cover all installed modules needed for **all** python scripts in this repo. 
**condainit.bash** - this bash file contains the commands to create and select this environment from the CLI. Prior to running any code, the user should make sure to run this script in order to have the right dependencies to run the rest of the scripts. 
