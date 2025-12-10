# Functional Specification

## Background
The goal of this work is to determine the sound frequencies that the spotted ratfish (Hydrolagus colliei) can hear and to present the findings to a broad scientific audience. The challenge is that the collected EEG data require specialized analysis and visualization methods to extract reliable auditory responses. The solution we implement is to develop custom signal-processing and data-analysis scripts, along with an interactive data-visualization website, to show the process of the analysis pipeline and clearly display the findings from this unique dataset.

## User profile
The **primary experimenter** and data analyst is deeply familiar with the dataset, the analysis scripts, and the relevant background information, and does not need the website to clarify terminology for themselves. Rather, they will use this software package mainly to analyze their datasets and use the website to help explain the general analysis pipeline to others including their PIs, collaborators, and friends. They have coding experience and strong general computer skills.

The **principal investigator** overseeing the experimenter understands the dataset but not at the same level of detail. For them, clear documentation in the analysis scripts and the data-visualization website will efficiently communicate the structure of the analysis pipeline and its key results. They do not code but can use a computer comfortably.

A **collaborator of the primary experimenter** has a working understanding of the project and will also use this software package to analyze their own related datasets. They are not involved in the daily processing of the primary experimenter’s data, so they rely on clear summaries of the dataset structure, the analysis workflow, and the resulting findings. Documentation in the analysis scripts and the website will support both their understanding of the existing work and their ability to run the pipeline independently. They have general coding experience and strong general computer skills.

A **generally curious friend of the analyst** has no background knowledge of the dataset and is mainly interested in the results rather than the analytical process. The data-visualization website will help them quickly grasp the main findings. They may have some coding experience that shapes how they interpret the visualizations, and they have basic computer skills.

## Data sources
1. Fish EEG recordings collected by Aoi Hunsaker in June 2025
2. Passive acoustic data via NOAA's google cloud storage

## Use Cases

### 1. Analyzing EEG data
Actors: Primary experimenter, collaborator  
Goal: To run custom pipeline of analysis scripts on fish EEG datasets to identify audible frequencies of fish
Stakeholder: Primary experimenter, PI, collaborators, general audience  
Precondition: Access to the compatible datasets and data analysis pipeline  
Triggers: A need to process a new eeg dataset to determine the hearing thresholds of a given fish  
Expected interactions:Install fish_eeg repo, load data and process data via pipeline.py on a dev.  environment (VS Code)
Exceptions: Script errors, corrupted datasets, python incompatability issues  
Assumptions: User has coding experience and is fully familiar with the data and analysis procedures  

User story:
Aoi is a graduate PhD student in Psychology at University of Washington.
Aoi will use this code base to publish research on a spotted ratfish.
Aoi wants a code base that is clear for the audience and shows scientific achievement.
Aoi is pretty technical and has been coding for this project prior to this class.

### 2. Verifying analytical choices and monitoring results
Actors: PI  
Goal: To review dataset structure, analytical decisions, and results to stay aligned with the project and provide input  
System: Documentation of analysis scripts and the visualization website  
Stakeholder: Collaborator, primary experimenter, research team  
Precondition: Access to the documentation, website, and any shared datasets or code  
Triggers: Invitation from the experimenter to review progress or prepare for joint work  
Expected interactions: Read documentation describing the data and analysis workflow, open the visualization website to explore interactive summaries, examine plots and metrics to verify analysis decisions, provide feedback to the experimenter to refine next steps  
Exceptions: Incomplete documentation, website issues, or outdated information  
Assumptions: User does not have general coding knoweldge but is deeply familiar with the scientific context  

User story:
Joseph is the PI of Aoi. 
They want to use the site to learn better what their PhD student is doing for their work. 
They need to have the data shown and the methods used to be understandable but also includes all the details necessary to see if Aoi has made any errors in their work. 
Joseph is a neuroethologist.

### 3. Curious friend exploring results of data analysis
Actors: Curious friend  
Goal: To understand the main results of the study without needing deep knowledge of the dataset or analytical methods  
System: The internet (website hosted online)  
Stakeholder: Friend, experimenter  
Precondition: Computer or mobile device with internet access  
Triggers: The experimenter shares the link to the friend  
Expected interactions: Open the website using the provided link, browse the results sections and use the interactive website features, view simple explanations of findings and significance  
Exceptions: Website too jargony for non-experts, technical issues, or missing explanations  
Assumptions: User has basic computer skills and limited background knowledge but interest in the findings  
