# Read me for initializing anaconda environment

For each type of script in this repo that is a python script, there are different modules that need to be installed. This can be done through anaconda or via "PIP". We tend to opt for anaconda because the modules can be stored in an "environment", where each conda environment has the right modules and versions for the job. In this folder we have two files:

**environment.yml** - lists out the modules in the conda environment which will cover all installed modules needed for **all** python scripts in this repo. 
**condainit.bash** - this bash file contains the commands to create and select this environment from the CLI. Prior to running any code, the user should make sure to run this script in order to have the right dependencies to run the rest of the scripts. 