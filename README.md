```
      /`·.¸
     /¸...¸`:·         ________
 ¸.·´  ¸   `·.¸.·´)   /        \ 
: © ):´;      ¸  {   <   glug   |
 `·.¸ `·  ¸.·´\`·¸)   \________/
     `\\´´\¸.·´
```	 
# Project Name: fish_eeg
A data analysis pipeline and web visualization package for auditory fish EEG recordings.

This package provides a complete workflow for processing, analyzing, and visualizing auditory fish EEG recordings. It includes tools for data ingestion, preprocessing, feature extraction, statistical analysis, and interactive web-based visualization.

# Website Link
https://aoih-uw.github.io/fish_eeg/index.html  

# Intended Audience
1. Researchers who need to analyze auditory fish EEG recordings  
2. Collaborators of researchers who want to understand the data analysis pipeline and its results  
3. Generally scientifically minded curious people who want to know about what sounds fish can hear  

# How to install the package
Option 1: Install directly from GitHub  
```bash
pip install git+https://github.com/aoih-uw/fish_eeg.git
```

Option 2: Clone the repository and install locally
```bash
git clone https://github.com/aoih-uw/fish_eeg.git
cd fish_eeg
pip install .
```
These commands will build and install the package from `pyproject.toml`

# Example of use
1. Ensure your dataset is in the right format. This package expects `.npz` file containing the following structure:  
```
loaded = np.load(f"{path}/{subjid}_data.npz", allow_pickle=True)
    data = loaded["data"]
    freq_amp_table = loaded["freq_amp_table"]
    latency = loaded["latency"].item()
    channel_keys = loaded["channel_keys"].tolist()
```
2. Place your dataset in the `data/` directory
```
 project_root/
	data/
		fish01.mat
```
3. Run the analysis pipeline
   You may run it directly by:  
```
   python pipeline.py --config_path path/to/config.yaml
```
or for beginners run this tutorial jupyter notebook:  
```
end_to_end_analysis.ipynb
```
This notebook is in /examples

4. Edit the html files as needed to display results from analysis pipeline
```
index.html (must stay at top level for github web host to detect it)
website/
	overview.html
	pipeline.html
	team.html
	visuals.html
```
You may edit these files to adjust labels, formatting, or visual elements before publishing or sharing the results.  

# Authors

### **Aoi Hunsaker**  
Role: Designed data analysis pipeline, wrote project documentation (Functional and design specification, docstrings, README.md), and lead coordination of project 

Psychology - PhD Student  - adv. Joseph Sisneros; Andrew Brown  
www.sisneroslab.org  
[aoih@uw.edu](mailto:aoih@uw.edu)

### **Michael James**  
Role: Added tools/passive_acoustic, wrote continuous integration, set up .toml and env.yml files for dependency coverage. 

Civil and Environmental Engineering - PhD Student  - adv. Jim Thomson; Kristin Zeiden

[Swift Lab](https://www.apl.washington.edu/project/project.php?id=swift)

[Environmental Fluid Mechanics Group](http://depts.washington.edu/uwefm/wordpress/)

[mkj29@uw.edu](mailto:mkj29@uw.edu)

### **Yash Sonthalia**

### **Jeffrey Jackson**
Role: Look over underlying statistical methods to ensure effective statistical methodology. Lead in testing, ensure tests function during continuous integration. Writeing docstrings and comments.

### **Christopher Tritt**

# Repository Folder Structure
---
```
├── assets
│   ├── css
│   ├── downloads
│   └── images
├── data
│   ├── PAcoustic
│   └── google-cloud-sdk
├── docs
├── examples
│   ├── JupyterNotebook
│   └── PipelineScript
├── pages
├── results
│   ├── aoi
│   ├── christopher
│   ├── jeff
│   ├── mike
│   └── yash
├── src
│   └── fish_eeg
├── test_data
├── tests
│   └── __pycache__
└── tools
    └── passive_acoustic
```
## assets

"Assets"

Everything that website uses as a resource for showing data

## data

"data" (.gitignored)

Location for users to store their data used in this repo.

## docs

"documentation"

Location of pdfs and other documents which provide contextual information to the repository.

## examples

"examples"

Examples of using the pipeline 

## pages

"webpages"

Webpage structure for fisheeg project information

## results

"results"

Final location of src products and project deliverables

## src

"source code"

Contains all code for running programs. Each subdir represents a different process and contains a README for functions specific to that process.

## test_data

"test data"

Example dataset used for testing

## tests

"tests"

List of tests used for pipeline.

## tools/passive_acoustic

"tools" "passive acoustic data"

Auxillary folder for loading in public acoustic data to compare to audiogram.
