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

[SWIFT Lab](https://www.apl.washington.edu/project/project.php?id=swift)

[Environmental Fluid Mechanics Group](http://depts.washington.edu/uwefm/wordpress/home/people/)

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
Contains the fish_eeg python package with tools to be used specifically for this test analysis. Prior to using, please initialize anaconda environment using "condainit.bash" (see below) and then install the package using "pip install ." while in the home directory of "fish eeg". To download Anaconda, please visit [their distrobution]("https://github.com/conda-forge/miniforge").

### data 
"data" (.gitignored)
Location for users to store their data used in this repo. 

### docs 
"documentation"
Location of pdfs and other documents which provide contextual information to the repository. 

### tools
"tools"
Location of different analysis methods and auxillary scripts that serve as a log of analysis development as well as contextual support scripts for ratfish auditory analysis. 

### tests
"tests"
Location of the tests used on the fish eeg python package. In [github actions]("https://github.com/aoih-uw/fish_eeg/actions") you can whether or not each test has passed for a given push to the repo. 

### results
Final location of src products and project deliverables.

## Initializing anaconda environment

For each type of script in this repo that is a python script, there are different modules that need to be installed. This can be done through anaconda or via "PIP". We tend to opt for anaconda because the modules can be stored in an "environment", where each conda environment has the right modules and versions for the job. As a tertiary environment, we make our conda environment initially to add onto while building the process. For this functionality we have 2 files:

**environment.yml** - lists out the modules in the conda environment which will cover all installed modules needed for python scripts excluding the python package in this repo. 
**condainit.bash** - this bash file contains the commands to create and select this environment from the command line. Prior to running any code, the user should make sure to run this script in order to have the right dependencies to run the rest of the scripts. 

**To initialize your environment, in a bash terminal (Linux) with the fisheeg directory selected, please run "source condainit.bash"**
