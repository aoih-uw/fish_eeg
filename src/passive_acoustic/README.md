# Read me for the loading of passive acoustic data. 

This folder allows the user to load in their own passive acoustic data via NOAA's google cloud storage. 

For the setup:
**gsutilinit.bash** - Adds the google cloud service utility to the user's CLI. This is the API used to pull down out acoustic dataset for the Puget Sound. 
**getacousticdata.bash** - Uses the google cloud service utility to load orcasound labs passive acoustic data from the Puget Sound. Stores data in a dir called "data" which is ignored by git. 
**plotspectra.py** - with folders with orcasound labs data for different locations from the Puget Sound, the user will input what location they want to run through each audio file. There is also an oppertunity to choose the type of fish by providing the path to the fish's test file. For each, the script will create Sound Pressure Level frequency spectra comparision of the audio file, including comparision of the inputs that were given to the fish. 