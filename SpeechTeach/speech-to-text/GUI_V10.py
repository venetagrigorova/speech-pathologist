#packages for GUI
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

#packages for audio
import wave
import scipy.io.wavfile as wavf

#packages for plotting
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import TemporalAnalysisLibrary_V3_2 as TAL

#packages for speech to text
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

#packages for palying and recording audio
import sounddevice
import soundfile
from scipy.io.wavfile import write

#packages for file management
import io
import os
#import threading
#import queue

#File loader functions --------------------------------------------------------
def open_fileA():
    
    try:
        #open a file dialog to browse for desired file
        path = filedialog.askopenfilename(initialdir="", 
                                          title="select file",
                                          filetypes=(("WAV files","*.wav"),("all files","*.*")))
        
        #print("testing: path = ", path)
        #check if file was found
        if(path == ""):
            console_write("No file selected")
        else:
            console_write("Selected File A:\n " + get_filename_from_path(path))
        
        #update StringVar object linked to the form
        window.filename_A.set(path)
                
        try: 
            
            #open file for reading
            window.sound_file_A = wave.open(path,'r')
            
            sound_file_A = window.sound_file_A    
            
        except:
            console_write("Error reading file")    
            raise BaseException
            #TODO: find a better exception for this
                
        #unlock textboxes for editing and clear their contents
        for box in attribute_boxes_A: 
            box["state"] = "normal"
            box.delete(0, tk.END)
        
        attribute_boxes_A[0].insert(0,get_filename_from_path(path))                                                 #filename
        attribute_boxes_A[1].insert(0,"{:.3f}".format(sound_file_A.getnframes()/sound_file_A.getframerate()))#duration
        attribute_boxes_A[2].insert(0,sound_file_A.getnframes())                            #number of samples
        attribute_boxes_A[3].insert(0,sound_file_A.getframerate())                          #framerate, sampling rate
        attribute_boxes_A[4].insert(0,sound_file_A.getsampwidth() * 8)                      #bit depth
        
        #determine if file is mono or stereo    
        if(sound_file_A.getnchannels() == 2):
            attribute_boxes_A[5].insert(0,"2 (stereo)")
        else:
            attribute_boxes_A[5].insert(0,"1 (mono)")
        
        #lock textboxes again
        for box in attribute_boxes_A: 
            box["state"] = "readonly"
            
    except:
        console_write("\nSomething went wrong with file selection...")
        
    #print("Window sound file = ", window.sound_file_A)
    window.sound_files[0] = window.sound_file_A
    window.signals[0] = window.sound_files[0].readframes(window.sound_files[0].getnframes())
    
def open_fileB():
    
    try:
        #open a file dialog to browse for desired file
        path = filedialog.askopenfilename(initialdir="", 
                                          title="select file",
                                          filetypes=(("WAV files","*.wav"),("all files","*.*")))
        
        #check if file was found
        if(path == ""):
            console_write("No file selected")
        else:
            console_write("Selected File B:\n " + get_filename_from_path(path))
        
        #update StringVar object linked to the form
        window.filename_B.set(path)
        
        try: 
            #open file for reading
            window.sound_file_B = wave.open(path,'r')
            
            sound_file_B = window.sound_file_B
            
        except:
            console_write("Error reading file")    
            raise BaseException
            #TODO: find a better exception for this
        
        #unlock textboxes for editing and clear their contents
        for box in attribute_boxes_B: 
            box["state"] = "normal"
            box.delete(0, tk.END)
        
        attribute_boxes_B[0].insert(0,get_filename_from_path(path))                                                 #filename
        attribute_boxes_B[1].insert(0,"{:.3f}".format(sound_file_B.getnframes()/sound_file_B.getframerate()))#duration
        attribute_boxes_B[2].insert(0,sound_file_B.getnframes())                            #number of samples
        attribute_boxes_B[3].insert(0,sound_file_B.getframerate())                          #framerate, sampling rate
        attribute_boxes_B[4].insert(0,sound_file_B.getsampwidth() * 8)                      #bit depth
        
        #determine if file is mono or stereo    
        if(sound_file_B.getnchannels() == 2):
            attribute_boxes_B[5].insert(0,"2 (stereo)")
        else:
            attribute_boxes_B[5].insert(0,"1 (mono)")
        
        #lock textboxes again
        for box in attribute_boxes_B: 
            box["state"] = "readonly"
               
    except:
        console_write("\nSomething went wrong with file selection...")
        
    window.sound_files[1] = window.sound_file_B
    window.signals[1] = window.sound_files[1].readframes(window.sound_files[1].getnframes())
    
#Viewing Tab-------------------------------------------------------------------
def refresh_plot_A():
    refresh_plot(0)

def refresh_plot_B():
    refresh_plot(1)


def refresh_plot(index):
     
     window.figures[index] = plt.Figure(figsize=(8,3), dpi=100)
     
     window.view_canvases[index] = FigureCanvasTkAgg(window.figures[index], master=window.view_frames[index])
     
     #extract raw audio from the .wav file
     try:   
         signal = window.signals[index]
         console_write("read new file contents")
         #convert file data to numpy array
         window.signals[index] = numpy.fromstring(window.signals[index], "Int16")
     
         window.framerates[index] = window.sound_files[index].getframerate()
     
         window.times[index] = numpy.linspace(0, len(window.signals[index]) / window.framerates[index], num=len(window.signals[index]))
         
     except ValueError:
         console_write("using same file as previous operation")
 
     #Get silences
     threshold = window.dB_silence_threshold
     min_silence_samp = 1000 
     min_chunk_samp = 1000
     
     console_write("\n (working) Getting Silences\nTheshold= "+str(threshold) + "\nmin. silence samples= " + str(min_silence_samp) + "\nmin. chunk samples= "+str(min_chunk_samp))
     
     print(window.signals[index])
     silence_timestamps, silence_signal= TAL.listSilences(window.signals[index], threshold_dB=threshold, min_silence_samples=min_silence_samp)
     
     console_write("\n(working) Getting Chunks from silences...")
     chunks = TAL.silencesToSoundChunks(silence_timestamps, min_chunk_samples = min_chunk_samp)
     
     console_write("\nGetting Peak sample for each chunk..")       
     peaks_list, peak_locations = TAL.findPeaks(signal, chunks)
 
     window.audioSubplots[index] = window.figures[index].add_subplot(111)
     
     console_write("potting waveform...")
     filename = get_filename_from_path(window.filenames[index].get())
     
     window.audioSubplots[index].set_title(filename + " waveform")
     window.audioSubplots[index].plot(window.times[index], window.signals[index], 'grey', alpha = 0.5)
     window.audioSubplots[index].plot(window.times[index], silence_signal, alpha = 0.5)
     
     if window.option_checkbox_values[index]["chunk_start"].get() == 1:
         console_write("adding event lines - chunk start")
         for chunk in chunks:    
             TAL.plotEvents_seconds([chunk[0]], colour = 'blue', plot_object = window.audioSubplots[index])
     
     if window.option_checkbox_values[index]["chunk_end"].get() == 1:
         console_write("adding event lines - chunk end")
         for chunk in chunks:        
             TAL.plotEvents_seconds([chunk[1]], colour = 'red', plot_object = window.audioSubplots[index])
     
     if window.option_checkbox_values[index]["show_peaks"].get() == 1:
         console_write("adding event lines - peaks")
         for peak in peak_locations:
                 TAL.plotEvents_seconds([peak], colour = 'black', plot_object = window.audioSubplots[index])
     
     if window.option_checkbox_values[index]["silence_start"].get() == 1:
         for silence in silence_timestamps:
             TAL.plotEvents_seconds([silence[0]], colour = 'green', plot_object = window.audioSubplots[index])
     
     if window.option_checkbox_values[index]["silence_end"].get() == 1:
         for silence in silence_timestamps:
             TAL.plotEvents_seconds([silence[1]], colour = 'orange', plot_object = window.audioSubplots[index])
 
     window.view_canvases[index].draw()
     window.view_canvases[index].get_tk_widget().grid(row="0",column="0", rowspan=2, sticky="ewns")


#console related funtions -----------------------------------------------------
def console_write(message):
    console['state'] = 'normal'
    console.insert(tk.END,"\n" + message+"\n")
    console.see(tk.END)
    console['state'] = 'disabled'

def get_filename_from_path(path):    
    list = path.split("/")
    
    if len(list) == 1:
        return path
    else:
        return list[-1]

#plotting related functions----------------------------------------------------
def plotAudio(desiredTab, signal, framerate):   
    #determine x-axis
    record_time = numpy.linspace(0, len(signal) / framerate, num=len(signal))
    
    window.record_plot = plt.Figure(figsize=(8,3), dpi=100)  
    window.record_canvas = FigureCanvasTkAgg(window.record_plot, master=window.frame_recordSpeech)
    
    window.record_subplot = window.record_plot.add_subplot(111)
    window.record_subplot.plot(record_time, signal, 'grey', alpha = 0.5)

    window.record_canvas.draw()
    window.record_canvas.get_tk_widget().grid(row="0",column="0", rowspan=2, sticky="ewns")

#recording related functions---------------------------------------------------
def buttonStartRecording():
    fs = 44100  # Sample rate
    duration = 5  # Duration of recording (s)
    
    enteredFilename = filenameEntry.get() # get the name of the file that the user entered in the field
    recordFilename = window.path_UserRecordings + "/" + enteredFilename + ".wav" #make complete filepath with the entered filename
    
    print("Duration of recording (seconds): ", duration)
    print("Starting Recording")
    myrecording = sounddevice.rec(int(duration * fs), samplerate=fs, channels=1, dtype='Int16')
    sounddevice.wait()  # Wait until recording is finished
    write(recordFilename, fs,  myrecording)  # Save as WAV file
    print("Done Recording")
    
    #use speech-to-text to interpret what the speaker said
    window.spokenText = speechToText(recordFilename)
    print(window.spokenText)
    
    window.spokenText = tk.Label(window.frame_transcribeSpeech, text=window.spokenText, padx=10, pady=10)
    window.spokenText.grid(row="0", column="1", padx="0.25c", pady="0.25c", sticky="ew")
    
    #plot the speaker's audio
    data, samplerate = soundfile.read(recordFilename, dtype='Int16')
    plotAudio(record_tab, data, samplerate) #plot the .wav file  
    
#speech to text----------------------------------------------------------------
def speechToText(audioFile):
    # Instantiates a client
    client = speech.SpeechClient.from_service_account_json('speechteach-7f18496fe981.json')
    
    # Loads the audio into memory
    with io.open(audioFile, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)
    
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='en-US')
    
    # Detects speech in the audio file
    response = client.recognize(config, audio)
    transcribedSpeech = "{}"
    
    for result in response.results:
        outputSpeech = transcribedSpeech.format(result.alternatives[0].transcript)
    
    return outputSpeech
    
###############################################################################
#BUILD GUI
###############################################################################

#setup window
window = tk.Tk()
window.title("Speech Teach")
window.geometry("1000x600")

#GUI Variables 

#console
window.console = None

#for loader tab
window.attribute_boxes_A = []
window.attribute_boxes_B = []

#for plotting
window.i = 0
window.filename_A = tk.StringVar()
window.filename_B = tk.StringVar()
window.filenames = [window.filename_A,window.filename_B]
window.file_attributes = ["Filename", "Duration", "No. of Samples","Sampling Rate", "Bit Depth", "Channels"]
window.view_frames = [None, None]
window.view_option_frames = [None, None]
window.btns_refresh = [None, None]
window.figures = [None, None]
window.view_canvases= [None, None]
window.signals = [None, None]
window.framerates = [None, None]
window.audioSubplots = [None, None]
window.times = [None, None]

#options
window.option_checkboxes = [{},{}]
window.option_checkbox_values = [{},{}]
window.dB_silence_threshold = 55.0

#sound files
window.sound_file_A = None
window.sound_file_B = None
window.sound_files = [window.sound_file_A, window.sound_file_B]

#SETUP TABS--------------------------------------------------------------------
tab_parent = ttk.Notebook(window)

file_loader_tab = ttk.Frame(tab_parent)
view_tab = ttk.Frame(tab_parent)
record_tab = ttk.Frame(tab_parent)
placeholder_tab = ttk.Frame(tab_parent)

tab_parent.add(file_loader_tab, text="Load Files")
tab_parent.add(view_tab, text="View")
tab_parent.add(record_tab, text = "Record Speech")

tab_parent.pack(side="left",padx="0.25c",pady="0.25c", expand= True, fill=tk.BOTH)


#CONSOLE-----------------------------------------------------------------------
console_frame = tk.LabelFrame(window,text = "Console", padx=5,pady=5)

console_frame.pack(side="right",
                   padx="0.25c",
                   pady="0.25c", 
                   expand= True, 
                   fill=tk.BOTH)

console = tk.Text(console_frame,
                  width=20, 
                  height=30, 
                  wrap="word", 
                  background="black", 
                  foreground="white")

console.pack(side=tk.LEFT, fill=tk.BOTH, expand= True)

scrollbar = tk.Scrollbar(console_frame)
scrollbar.config(command=console.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.BOTH)

console.config(yscrollcommand=scrollbar.set, state="disabled")

# FILE LOADER #################################################################
# frame for both files
fileA_frame = tk.LabelFrame(file_loader_tab,text = "File A", padx=10,pady=10)
fileA_frame.pack(fill=tk.X, padx="0.25c",pady="0.25c")

fileB_frame = tk.LabelFrame(file_loader_tab,text = "File B", padx=10,pady=10)
fileB_frame.pack(fill=tk.X, padx="0.25c",pady="0.25c")

#browse button
btn_open_fileA= tk.Button(fileA_frame,text="Find File A", command = open_fileA)
btn_open_fileA.grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="ew")

btn_open_fileB= tk.Button(fileB_frame,text="Find File B", command = open_fileB)
btn_open_fileB.grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="ew")

#labels for displaying filename, path
fileA_textBox = tk.Entry(fileA_frame)
fileA_textBox.grid(row="0",column="1", columnspan=5, sticky="ew")
window.filename_A.set("<No File Selected>")
fileA_textBox["textvariable"] = window.filename_A

fileB_textBox = tk.Entry(fileB_frame)
fileB_textBox.grid(row="0", column="1", columnspan=5, sticky="ew")
window.filename_B.set("<No File Selected>")
fileB_textBox["textvariable"] = window.filename_B

#build a table of fields to be populated later
attribute_boxes_A = []
attribute_boxes_B = []

#configure first column weights --> filename field should expand more quickly
fileA_frame.columnconfigure(0,weight=2)
fileB_frame.columnconfigure(0,weight=2)

i=0
#all other columns should have equal weight
for i in range(1,5):
    fileA_frame.columnconfigure(i,weight=1)
    fileB_frame.columnconfigure(i,weight=1)

i=0
for field_name in window.file_attributes:
    
    #create labels
    tk.Label(fileA_frame,text= field_name).grid(column=str(i), row= 1)
    tk.Label(fileB_frame,text= field_name).grid(column=str(i), row= 1)    
    
    #create textbox to hold attribute information
    attribute_boxes_A.append(tk.Entry(fileA_frame, state="readonly", justify= "center"))
    attribute_boxes_B.append(tk.Entry(fileB_frame, state="readonly", justify= "center"))    
    
    #layout fields accross columns
    if(i==0):
        attribute_boxes_A[i].grid(column=str(i), row= 2, sticky="ew")    
        attribute_boxes_B[i].grid(column=str(i), row= 2, sticky="ew")    
    else:
        attribute_boxes_A[i].grid(column=str(i), row= 2, sticky="ew")
        attribute_boxes_B[i].grid(column=str(i), row= 2, sticky="ew")
        
    i+=1   

# VIEW TAB ####################################################################
#create frames inside tab
window.view_frames[0] = tk.LabelFrame(view_tab,text = "File A", padx=10,pady=10)
window.view_frames[0].grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="ewns")

window.view_frames[1] = tk.LabelFrame(view_tab,text = "File B", padx=10,pady=10)
window.view_frames[1].grid(row="1",column="0",padx="0.25c",pady="0.25c", sticky="ewns")

tk.Canvas(window.view_frames[0], width=500, height=200).grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="nsew")
tk.Canvas(window.view_frames[1], width=500, height=200).grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="nsew")

window.figures[0] = plt.Figure(figsize=(7,2), dpi=100)
window.view_canvases[0] = FigureCanvasTkAgg(window.figures[0], master=window.view_frames[0])

window.figures[1] = plt.Figure(figsize=(7,2), dpi=100)
window.view_canvases[1] = FigureCanvasTkAgg(window.figures[1], master=window.view_frames[1])

#options pane on view tab
window.view_option_frames[0] = tk.LabelFrame(view_tab,text = "File A Options", padx=10, pady=10)
window.view_option_frames[0].grid(row="0",column="1",padx="0.25c",pady="0.25c", sticky="nsew")

window.view_option_frames[1] = tk.LabelFrame(view_tab,text = "File A Options", padx=10, pady=10)
window.view_option_frames[1].grid(row="1",column="1",padx="0.25c",pady="0.25c", sticky="nsew")

#add option buttons
"""
Options:
    -show where silences start
    -show where silence end
    -show where chunks start
    -show where chunks end
    -show chunk peaks
    -plot in dB
"""
checkbox_metadata = []
checkbox_metadata.append(["silence_start","show where silent segments start"])
checkbox_metadata.append(["silence_end","show where silent segments end"])
checkbox_metadata.append(["chunk_start", "show where sound chunks start"])
checkbox_metadata.append(["chunk_end", "show where sound chunks end"])
checkbox_metadata.append(["show_peaks",  "show chunk peaks"])
checkbox_metadata.append(["plot_dB", "plot in dB"])
j=1
for item in checkbox_metadata:
    for i in range(0,2):
        window.option_checkbox_values[i][item[0]] = tk.IntVar()
        window.option_checkboxes[i][item[0]] = tk.Checkbutton(window.view_option_frames[i], 
                                                                text = item[1], 
                                                                variable = window.option_checkbox_values[i][item[0]],
                                                                onvalue=1,
                                                                offvalue=0,
                                                                justify= tk.LEFT).grid(row=str(j),column="0", sticky="w")
    j+=1 

#refresh buttons button
window.btns_refresh[0] = tk.Button(window.view_option_frames[0],text="Refresh A", command = refresh_plot_A)
window.btns_refresh[0].grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="nsew")

window.btns_refresh[1] = tk.Button(window.view_option_frames[1],text="Refresh B", command = refresh_plot_B)
window.btns_refresh[1].grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="nsew")
        
#define column weights to try to get window resizing
window.view_frames[0].columnconfigure(0,weight=2)
window.view_frames[1].columnconfigure(0,weight=2)

window.view_frames[0].columnconfigure(1,weight=1)
window.view_frames[1].columnconfigure(1,weight=1)

window.view_option_frames[0].columnconfigure(0,weight=1)
window.view_option_frames[1].columnconfigure(0,weight=1)

#RECORD TAB####################################################################
window.path_Script = os.path.dirname(__file__)
window.path_UserRecordings = window.path_Script + "/User Recordings" #where the user's recorded speech files are stored

window.stopRecording = 0

#frame for buttons
window.frame_buttonsrecordSpeech = tk.LabelFrame(record_tab, text = "Record Speech", padx=10, pady=10)
window.frame_buttonsrecordSpeech.grid(row="0", column="0", padx="0.25c", pady="0.25c", sticky="ew")

#frame for transcript
window.frame_transcribeSpeech = tk.LabelFrame(record_tab, text = "Transcript Recorded Speech", padx=10, pady=10)
window.frame_transcribeSpeech.grid(row="1", column="0", padx="0.25c", pady="0.25c", sticky="nsew")

#frame for plot
window.frame_recordSpeech = tk.LabelFrame(record_tab, text = "Plot of Recorded Speech", padx=10, pady=10)
window.frame_recordSpeech.grid(row="2", column="0", padx="0.25c", pady="0.25c", sticky="nsew")


tk.Canvas(window.frame_recordSpeech, width=800, height=200).grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="nsew")

#label for the filename field
window.filenameField = tk.Label(window.frame_buttonsrecordSpeech, text="Filename", padx=10, pady=10)
window.filenameField.grid(row="0", column="2", padx="0.25c", pady="0.25c", sticky="ew")

#creating the filename entry field
filenameEntry = tk.Entry(window.frame_buttonsrecordSpeech)
filenameEntry.grid(row="0", column="3", padx="0.25c", pady="0.25c", sticky="ew")

#label for the transcript field
window.transcriptLabel = tk.Label(window.frame_transcribeSpeech, text="You Said: ", padx=10, pady=10)
window.transcriptLabel.grid(row="0", column="0", padx="0.25c", pady="0.25c", sticky="ew")


#creating the plot canvas
window.record_plot = plt.Figure(figsize=(7,2), dpi=100)
window.record_canvas = FigureCanvasTkAgg(window.record_plot, master=window.frame_recordSpeech)

#button to start recording
window.btn_recordSpeech = tk.Button(window.frame_buttonsrecordSpeech, text="Record Speech", command=buttonStartRecording)
window.btn_recordSpeech.grid(row="0", column="0", padx="0.25c", pady="0.25c", sticky="ew")





console_write("SPEECH TEACH!\nInitialized...")
#launch main loop
window.mainloop()
