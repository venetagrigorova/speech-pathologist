****************************************************************

Welcome to SpeechTeach, a speech therapy assitive program! 

****************************************************************

by Capstone Group 13: Raphael Capon, Veneta Grigorova, Hira Nadeem, Ben Raubvogel

*NOTE: due to the large number of depencies, this project is unlikely to run
on machines that have not been appropriately configured.*

Project was programmed in python 3.7

->The following dependencies are needed:

packages for GUI
- tkinter as tk
- tkinter ttk
- tkinter filedialog

packages for audio
- wave
- soundfile
- sounddevice
- threading
- aubio

packages for file management
- signal
- os
- io

packages for plotting
- numpy
- matplotlib.pyplot as plt
- matplotlib.backends.backend_tkagg (FigureCanvasTkAgg, NavigationToolbar2Tk)
- PIL

Other
- time
- scipy.io.wavfile
- math

machine learning
- keras
- tensorflow
- praat-parselmouth
- scikitlearn

text-to-speech (implemented, but depends too heavily on external software)
- google.cloud speech
- google.cloud.speech enums
- google.cloud.speech types

->Instructions: basic usage scenario
1. Unzip all files. 
2. Open GUI.py in command prompt or a Python IDE.
3. Use "load" tab to locate files for analysis
	-Two files can be loaded at once, for comparison, but only one of them can be analyzed with the Machine Learning model
	-File properties will populate upon successful import
4. Switch to the "view" tab
	-Select plotting options and press "refresh" to view plot of loaded data
	-use play/stop buttons to listen to files
5. Remaining on view tab, select a Machine Learning model from the dropdown menu on the right
	- each model was trained with many samples of the same sentence. The text corresponding to each model are displayed below
	- click the "analyze" button to detect stuttering the sound file. Only file slot A can be analyzed.
	- Analysis may take around 30 seconds, the program will be unavailable during that time.
6. Switch to the record tab. Enter a filename (omitting extension) and a desired recording duration
	- if the recorded sample is to be analyzed, check the box on the right to immediately load the new file into slot A.
	- Click the "record" button start recording. Recording stops automatically after specified time.
	- Ideally, the software also produces a transcript of the the textfile using google speech-to-text services
	due to portability issues, this functionality has been disabled but can be examined in the "speech-to-text" folder
7. The recorded file can be accessed in the view tab. File properties will be available in the "Load" tab.
	-the files can be accessed at any time in the "userrecording" subfolder.

->Directory Contents:

>Files:

GUI.py ->This is the main module, most code is contained in this file. Run this file to access program

imageclassify.py -> contains machine learning functions for analyzing data using existing models

TemporalAnalysisLibrary_V3_2.py -> collection of numerical analysis functions

full_transcript_ALL_FILES.txt -> transcript of entire text from which samples were drawn

>Folders:

userspectrograms -> contains spectrograms of each soundfile analyzed. essential to the categorization algorithm

userrecordings -> initially empty, contains recordings created by the user

test_samples -> contains a number of samples to help in evaluating the project

model -> contains the datafiles describing machine learning models. Three models are available.

working_files -> contains code that was used to train the models. The dataset used for training was not included but is available upon request.

speech-to-text -> contains code used to enable speech to text functionality. The configuration requirements for these functions are significant, but the code is included for reference. For more information, search for Google Speech-to-text API.
