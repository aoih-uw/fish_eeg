Information about "data" structure

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