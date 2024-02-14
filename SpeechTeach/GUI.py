#Speech Teach
#April 9 2020

#Capstone group 13: Raphael Capon, Veneta Grigorova, Hira Nadeem, Ben Raubvogel

#packages for GUI
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

#packages for audio
import wave
import soundfile
import sounddevice
import threading

#packages for file management
import signal
import os
import io

#packages for plotting
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import TemporalAnalysisLibrary_V3_2 as TAL

#other stuff
import time
import scipy.io.wavfile as wavf
from scipy.io.wavfile import write

#machine learning code
#from imageclassify import *
import imageclassify
from keras.models import load_model
from keras.preprocessing import image

#text-to-speech (implemented, but depends too heavily on external software)
#from google.cloud import speech
#from google.cloud.speech import enums
#from google.cloud.speech import types

import parselmouth
import PIL


#GUI variables-----------------------------------------------------------------
#setup window
window = tk.Tk()
window.title("Speech Teach")
window.geometry("1920x1080")

#GUI Variables 
##console
window.console = None

##for loader tab
window.attribute_boxes = [None,None]
window.i = 0

#these store entire path, access filename using get_filename_from_path()
window.filename_A = tk.StringVar()
window.filename_B = tk.StringVar()
window.filenames = [window.filename_A,window.filename_B]
window.file_attributes = ["Filename", "Duration", "No. of Samples","Sampling Rate", "Bit Depth", "Channels"]

#build a table of fields to be populated later
window.attribute_boxes = [[],[]]

#plotting
window.figures = [None, None]
window.view_canvases= [None, None]
window.signals = [None, None]
window.framerates = [None, None]
window.audioSubplots = [None, None]
window.times = [None, None]
window.view_is_playing = False

#frames
window.view_frames = [None, None]
window.view_option_frames = [None, None]
window.view_progress_bar_frames = [None, None]
view_attribute_boxes = [None]

#heatmap
window.analyze_heatmap_canvas= None

#buttons
window.btns_refresh = [None, None]
window.btns_play = [None, None]
window.btns_stop = [None, None]


#progress bars
window.view_progress_bars = [None, None]
window.view_progress_bar_progress = [0,0]

#options
window.option_checkboxes = [{},{}]
window.option_checkbox_values = [{},{}]
window.dB_silence_threshold = 55.0

#ML related
window.btn_analyze = None
window.analyze_frame = None
window.analyze_result_label = None
window.selectedPrompt = tk.StringVar()


#sound files
window.sound_file_A = None
window.sound_file_B = None
window.sound_files = [window.sound_file_A, window.sound_file_B]

#audio
window.streams = [None,None]
window.current_position = 0

#recording
window.enteredFilename = tk.StringVar()
window.recordDuration_label = None
window.recordDuration_value_textbox = None
window.recordDuration_value = tk.StringVar()
window.record_checkbox = None
window.record_checkbox_value = tk.IntVar()



##############################################################################
# ______                _   _                 
# |  ___|              | | (_)                
# | |_ _   _ _ __   ___| |_ _  ___  _ __  ___ 
# |  _| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# | | | |_| | | | | (__| |_| | (_) | | | \__ \
# \_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
                
#File loader functions --------------------------------------------------------

def open_fileA():
    #open a file dialog to browse for desired file
    path = filedialog.askopenfilename(initialdir="", 
                                      title="select file",
                                      filetypes=(("WAV files","*.wav"),("all files","*.*")))
    #check if file was found  
    if(path == ""):
        console_write("No file selected")
        return
    
    
    window.filenames[0].set(path)
    console_write("Selected File A:\n " + get_filename_from_path(path))
    open_file(0)
    
    
def open_fileB():
    
    #open a file dialog to browse for desired file
    path = filedialog.askopenfilename(initialdir="", 
                                      title="select file",
                                      filetypes=(("WAV files","*.wav"),("all files","*.*")))
    
    #check if file was found  
    if(path == ""):
        console_write("No file selected")
        return
    
    window.filenames[1].set(path)
    console_write("Selected File B:\n " + get_filename_from_path(path))
    open_file(1)
    
def open_file(index):
    
    try:
        
        path = window.filenames[index].get()
                            
        try:     
            window.sound_files[index] = soundfile.SoundFile(path)
        except:
            console_write("Error reading file")    
            raise BaseException
            #TODO: find a better exception for this
                
        #unlock textboxes for editing and clear their contents
        for box in window.attribute_boxes[index]: 
            box["state"] = "normal"
            box.delete(index, tk.END)
        
        duration = 1.0*window.sound_files[index].frames / window.sound_files[index].samplerate
        window.attribute_boxes[index][0].insert(0,get_filename_from_path(window.filenames[index].get()))                                  #filename
        print(get_filename_from_path(window.filenames[index].get()))
        window.attribute_boxes[index][1].insert(0,"{:.3f}".format(duration))  #duration
        window.attribute_boxes[index][2].insert(0, window.sound_files[index].frames)                                           #number of samples
        window.attribute_boxes[index][3].insert(0, window.sound_files[index].samplerate)                                       #framerate, sampling rate
        window.attribute_boxes[index][4].insert(0, window.sound_files[index].subtype)                                          #bit depth
        
        #determine if file is mono or stereo    
        if(window.sound_files[index].channels == 2):
            window.attribute_boxes[index][5].insert(0,"2 (stereo)")
        else:
            window.attribute_boxes[index][5].insert(0,"1 (mono)")
        
        #lock textboxes again
        for box in window.attribute_boxes[index]: 
            box["state"] = "readonly"
            
    except:
        console_write("\nSomething went wrong with file selection...") 
        
    #window.signals[index] = soundfile.read(path, dtype = "float32")[0]
    window.signals[index] = soundfile.read(path, dtype = "int16")[0]
        
#Plotting----------------------------------------------------------------------

def plotAudio(desiredTab, signal, framerate):   
    
    #determine x-axis
    record_time = numpy.linspace(0, len(signal) / framerate, num=len(signal))
    
    window.record_plot = plt.Figure(figsize=(8,3), dpi=100)  
    window.record_canvas = FigureCanvasTkAgg(window.record_plot, master=window.frame_recordSpeech)
    
    window.record_subplot = window.record_plot.add_subplot(111)
    window.record_subplot.plot(record_time, signal, 'grey', alpha = 0.5)

    window.record_canvas.draw()
    window.record_canvas.get_tk_widget().grid(row="0",column="0", rowspan=2, sticky="ewns")

def refresh_plot_A():
    refresh_plot(0)

def refresh_plot_B():
    refresh_plot(1)

def refresh_plot(index):
    
    window.figures[index] = plt.Figure(figsize=(7,2), dpi=100)
    
    window.view_canvases[index] = FigureCanvasTkAgg(window.figures[index], master=window.view_frames[index])
    
    #extract raw audio from the .wav file
    try:
        signal = window.signals[index]
        console_write("read new file contents")
    
        window.framerates[index] = window.sound_files[index].samplerate
    
        window.times[index] = numpy.linspace(0, len(window.signals[index]) / window.framerates[index], num=len(window.signals[index]))
        
    except ValueError:
        console_write("using same file as previous operation")
    
    #Get silences
    threshold = window.dB_silence_threshold
    min_silence_samp = 1000 
    min_chunk_samp = 1000
    
    console_write("\n (working) Getting Silences\nTheshold= "+str(threshold) + "\nmin. silence samples= " + str(min_silence_samp) + "\nmin. chunk samples= "+str(min_chunk_samp))
    
    silence_timestamps, silence_signal= TAL.listSilences(window.signals[index], threshold_dB=threshold, min_silence_samples=min_silence_samp)
    
    console_write("\n(working) Getting Chunks from silences...")
    chunks = TAL.silencesToSoundChunks(silence_timestamps, min_chunk_samples = min_chunk_samp)
    
    console_write("\nGetting Peak sample for each chunk..")       
    peaks_list, peak_locations = TAL.findPeaks(signal, chunks)

    window.audioSubplots[index] = window.figures[index].add_subplot(111)
    
    console_write("potting waveform...")
    filename = get_filename_from_path(window.filenames[index].get())
    
    window.audioSubplots[index].set_title(get_filename_from_path(filename) + " waveform")
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
    
    #update image
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
    
def get_path(file_path):
    
    list = file_path.split("/")
    
    if len(list) == 1:
        return file_path
    else:
        return list[0:-1]
    
#button related functions------------------------------------------------------
def buttonListenPressed_A(): 
    if(window.view_is_playing == False):
        buttonListenPressed(0)
    else:
        console_write("Playback already in progress")
    
def buttonListenPressed_B(): 
    if(window.view_is_playing == False):
        buttonListenPressed(1)
    else:
        console_write("Playback already in progress")
        
def buttonListenPressed(index):  
    if(not window.view_is_playing):    
        window.sound_files[index] = soundfile.SoundFile(window.filenames[index].get())    
        switch_buttons_to_pause(index)

        window.view_is_playing = True
        start_progress_bar(index)
        
    else:
        console_write("Playback already in progress")
        
def switch_buttons_to_pause(index):
    i=0
    for button in window.btns_play:
        button["bg"] = "Orange"
        button["text"] =  "Playing..."
        i+=1

def buttonStopPressed():
    i=0
    window.view_is_playing = False
    window.current_position 
    
    for button in window.btns_play:
        button["bg"] = "green"
        button["text"] =  "Play Audio"
        if i == 0:
            button["command"] = buttonListenPressed_A
        else:
            button["command"] = buttonListenPressed_B
        i+=1
        
def buttonAnalyzePressed():
    
    try:
        if(window.selectedPrompt.get() == ""):
            console_write("No model Selected. \nUse Dropdown.")
            return
        elif(window.selectedPrompt.get() == "3"):
            result = imageclassify.classify(3, window.filenames[0].get())
        elif(window.selectedPrompt.get() == "10"):
            result = imageclassify.classify(10, window.filenames[0].get())
        elif(window.selectedPrompt.get() == "13"):
            result = imageclassify.classify(13, window.filenames[0].get())
    except:
        console_write("Error: Could not perform analysis")
    
    if result:
        window.analyze_result_label["text"] = "Stutter Detected"
        window.analyze_result_label["background"] = "red"
    else:
        window.analyze_result_label["text"] = "No Stutter Detected"
        window.analyze_result_label["background"] = "green"
        
    
def buttonStartRecording():
    fs = 44100  # Sample rate
    
    filename = window.enteredFilename.get()
    
    if(filename == ""):
        console_write("No filename provided. Could not open")
    else:
        
        duration = int(window.recordDuration_value.get())  # Duration of recording (s)
        #window.enteredFilename = window.enteredFilename.get() # get the name of the file that the user entered in the field    
        recordFilename = window.path_UserRecordings + '\\' + filename + ".wav" #make complete filepath with the entered filename
        
        print(recordFilename)
        
        print("Duration of recording (seconds): ", duration)
        print("Starting Recording")
        myrecording = sounddevice.rec(int(duration * fs), samplerate=fs, channels=1, dtype='Int16')
        sounddevice.wait()  # Wait until recording is finished
        write(recordFilename, fs, myrecording)  # Save as WAV file
        console_write("Done Recording")
        
        if window.record_checkbox_value.get():
            console_write("Loading recorded file to Slot A")
            window.filename_A.set(recordFilename)
            open_file(0)
        
        #use speech-to-text to interpret what the speaker said
        #window.spokenText = speechToText(recordFilename)
        #print(window.spokenText)
        
        #window.spokenText = tk.Label(window.frame_transcribeSpeech, text=window.spokenText, padx=10, pady=10)
        #window.spokenText.grid(row="0", column="1", padx="0.25c", pady="0.25c", sticky="ew")
        
        #plot the speaker's audio
        data, samplerate = soundfile.read(recordFilename, dtype='Int16')
        plotAudio(record_tab, data, samplerate) #plot the .wav file    

#Progressbar and Threading----------------------------------------------------
def handle_progress_barA(indata, outdata, frames, ctime, status):
    
    if not window.view_is_playing:
        raise sounddevice.CallbackStop
    
    data = window.sound_files[0].buffer_read(frames, dtype = "float32")
    print("in length=", len(data), "frames=", frames)

    try:
        outdata[:] = data
    except ValueError:
        print("incomplete final frame")
    
    window.view_progress_bar_progress[0] += frames
    #print("increment=",frames," progress=", window.view_progress_bar_progress[0])
    
    new_value = int(100*window.view_progress_bar_progress[0]/window.sound_files[0].frames) + 1
    window.view_progress_bars[0]["value"] = new_value
    
def handle_progress_barB(indata, outdata, frames, ctime, status):
    
    data = window.sound_files[1].buffer_read(frames, dtype = "float32")
    print("in length=", len(data), "frames=", frames)
    outdata[:] = data
    
    window.view_progress_bar_progress[1] += frames
    #print("increment=",frames," progress=", window.view_progress_bar_progress[0])
    
    new_value = int(100*window.view_progress_bar_progress[1]/window.sound_files[1].frames) + 1
    window.view_progress_bars[1]["value"] = new_value
    
def playback_finished():

    console_write("Playback Stopped")
    

def thread_function(index):
    frames = window.sound_files[index].frames
    frame_rate = window.sound_files[index].samplerate
    
    if index == 0:
        with sounddevice.RawStream(channels=1, callback=handle_progress_barA, finished_callback=playback_finished):
            sounddevice.sleep(int(1000*frames/frame_rate))
    else:
        print("index = ",index)
        with sounddevice.RawStream(channels=1, callback=handle_progress_barB, finished_callback=playback_finished):
            sounddevice.sleep(int(1000*frames/frame_rate))
        
def start_progress_bar(index):
    window.view_progress_bars[index]["value"] = 0
    window.view_progress_bar_progress[index] = 0
    
    x = threading.Thread(target=thread_function,args=(index,), daemon = True)
    x.start()
    
    console_write("Starting Playback")
    
#Machine Learning
     
# #speech to text----------------------------------------------------------------
# def speechToText(audioFile):
#     # Instantiates a client
#     client = speech.SpeechClient.from_service_account_json('speechteach-7f18496fe981.json')
    
#     # Loads the audio into memory
#     with io.open(audioFile, 'rb') as audio_file:
#         content = audio_file.read()
#         audio = types.RecognitionAudio(content=content)
    
#     config = types.RecognitionConfig(
#         encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=44100,
#         language_code='en-US')
    
#     # Detects speech in the audio file
#     response = client.recognize(config, audio)
#     transcribedSpeech = "{}"
    
#     for result in response.results:
#         outputSpeech = transcribedSpeech.format(result.alternatives[0].transcript)
    
#     return outputSpeech
    
##############################################################################
# ______                      _____ _   _ _____ 
# |  _  \                    |  __ \ | | |_   _|
# | | | |_ __ __ ___      __ | |  \/ | | | | |  
# | | | | '__/ _` \ \ /\ / / | | __| | | | | |  
# | |/ /| | | (_| |\ V  V /  | |_\ \ |_| |_| |_ 
# |___/ |_|  \__,_| \_/\_/    \____/\___/ \___/ 
                                              
#SETUP TABS--------------------------------------------------------------------
tab_parent = ttk.Notebook(window)

file_loader_tab = ttk.Frame(tab_parent)
view_tab = ttk.Frame(tab_parent)
record_tab = ttk.Frame(tab_parent)
#placeholder_tab = ttk.Frame(tab_parent)

tab_parent.add(file_loader_tab, text="Load Files")
tab_parent.add(view_tab, text="View")
tab_parent.add(record_tab, text="Record")
#tab_parent.add(placeholder_tab, text="placeholder")

tab_parent.pack(side="left",padx="0.25c",pady="0.25c", expand= True, fill=tk.BOTH)

# placeholder = tk.Canvas(placeholder_tab, width=700, height=500)
# placeholder.pack()

#CONSOLE-----------------------------------------------------------------------
console_frame = tk.LabelFrame(window,text = "Console", padx=5,pady=5)
window.columnconfigure(1,weight=2)

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

# FILE LOADER ----------------------------------------------------------------
# frame for both files
fileA_frame = tk.LabelFrame(file_loader_tab,text = "File A", padx=10,pady=10)
fileA_frame.pack(fill=tk.X, padx="0.25c",pady="0.25c")

fileB_frame = tk.LabelFrame(file_loader_tab,text = "File B", padx=10,pady=10)
fileB_frame.pack(fill=tk.X, padx="0.25c",pady="0.25c")

#browse button
btn_open_fileA = tk.Button(fileA_frame,text="Find File A", command = open_fileA)
btn_open_fileA.grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="ew")

btn_open_fileB= tk.Button(fileB_frame,text="Find File B", command = open_fileB)
btn_open_fileB.grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="ew")

#TODO: consider adding a weight of 1 for each column and row to fix the
#stretching problem

#labels for displaying filename, path
fileA_textBox = tk.Entry(fileA_frame)
fileA_textBox.grid(row="0",column="1", columnspan=5, sticky="ew")
window.filename_A.set("<No File Selected>")
fileA_textBox["textvariable"] = window.filename_A

fileB_textBox = tk.Entry(fileB_frame)
fileB_textBox.grid(row="0", column="1", columnspan=5, sticky="ew")
window.filename_B.set("<No File Selected>")
fileB_textBox["textvariable"] = window.filename_B

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
    window.attribute_boxes[0].append(tk.Entry(fileA_frame, state="readonly", justify= "center"))
    window.attribute_boxes[1].append(tk.Entry(fileB_frame, state="readonly", justify= "center"))    
    
    #layout fields accross columns
    if(i==0):
        window.attribute_boxes[0][i].grid(column=str(i), row= 2, sticky="ew")    
        window.attribute_boxes[1][i].grid(column=str(i), row= 2, sticky="ew")    
    else:
        window.attribute_boxes[0][i].grid(column=str(i), row= 2, sticky="ew")
        window.attribute_boxes[1][i].grid(column=str(i), row= 2, sticky="ew")
        
    i+=1   
    #TODO: need to associate these to their own StringVar objects

# VIEW TAB -------------------------------------------------------------------
#create frames inside tab
window.view_frames[0] = tk.LabelFrame(view_tab,text = "File A", padx=10,pady=10)
window.view_frames[0].grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="ewns")

window.view_frames[1] = tk.LabelFrame(view_tab,text = "File B", padx=10,pady=10)
window.view_frames[1].grid(row="2",column="0",padx="0.25c",pady="0.25c", sticky="ewns")

#progress bars
window.view_progress_bar_frames[0] = tk.LabelFrame(view_tab,text = "File A Playback Progress", padx=10,pady=10)
window.view_progress_bar_frames[0].grid(row="1",column="0", columnspan = 2,padx="0.25c",pady="0.25c", sticky="ewns")

window.view_progress_bar_frames[1] = tk.LabelFrame(view_tab,text = "File B Playback Progress", padx=10,pady=10)
window.view_progress_bar_frames[1].grid(row="3",column="0",
                                        columnspan = 2,
                                        padx="0.25c",pady="0.25c",
                                        sticky="ewns")

#options pane on view tab
window.view_option_frames[0] = tk.LabelFrame(view_tab,text = "File A Options", padx=10, pady=10)
window.view_option_frames[0].grid(row="0",column="1",
                                  padx="0.25c",pady="0.25c",
                                  sticky="nsew")

window.view_option_frames[1] = tk.LabelFrame(view_tab,text = "File A Options", padx=10, pady=10)
window.view_option_frames[1].grid(row="2",column="1",
                                  padx="0.25c",pady="0.25c",
                                  sticky="nsew")

#analysis frames
window.analyze_frame = tk.LabelFrame(view_tab,text = "File A Analysis", padx=10,pady=10)
window.analyze_frame.grid(row="0",column="2",
                          rowspan = 4, 
                          padx="0.25c",pady="0.25c",
                          sticky="ewns")

#create figures for plotting
window.figures[0] = plt.Figure(figsize=(7,2), dpi=100)
window.view_canvases[0] = FigureCanvasTkAgg(window.figures[0], master=window.view_frames[0])

window.figures[1] = plt.Figure(figsize=(7,2), dpi=100)
window.view_canvases[1] = FigureCanvasTkAgg(window.figures[1], master=window.view_frames[1])

window.view_canvases[0].get_tk_widget().grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="nsew")
window.view_canvases[1].get_tk_widget().grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="nsew")

#Create buttons
window.btns_play[0] = tk.Button(window.view_option_frames[0],text="Play Audio A", command = buttonListenPressed_A)
window.btns_play[1] = tk.Button(window.view_option_frames[1],text="Play Audio B", command = buttonListenPressed_B)

window.btns_play[0].grid(row="10",column="0",padx="0.25c",pady="0.25c", sticky="nsew")
window.btns_play[1].grid(row="10",column="0",padx="0.25c",pady="0.25c", sticky="nsew")

#modify button appearance -> green
i=0
for button in window.btns_play:
    button["bg"] = "green"

window.btns_stop[0] = tk.Button(window.view_option_frames[0],text="Stop", command = buttonStopPressed)
window.btns_stop[1] = tk.Button(window.view_option_frames[1],text="Stop", command = buttonStopPressed)

window.btns_stop[0].grid(row="11",column="0",padx="0.25c",pady="0.25c", sticky="nsew")
window.btns_stop[1].grid(row="11",column="0",padx="0.25c",pady="0.25c", sticky="nsew")

#modify stop button appearance -> red
i=0
for button in window.btns_stop:
    button["bg"] = "red"

window.btn_analyze = tk.Button(window.analyze_frame,text="Perform Stutter Analysis", command = buttonAnalyzePressed)
window.btn_analyze.grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="nsew")

#configure results box
window.analyze_result_label = tk.Label(window.analyze_frame,text= "-")
window.analyze_result_label.grid(column="0", row= "1")

window.analyze_result_label["height"] = "20"
window.analyze_result_label["width"] = "14"
window.analyze_result_label["anchor"] = tk.CENTER
window.analyze_result_label["relief"] = tk.SUNKEN
window.analyze_result_label["background"] = "blue"
window.analyze_result_label["foreground"] = "white"
window.analyze_result_label["wraplength"] = 100
window.analyze_result_label["font"] = ("Arial", 16, "bold")

#progress bar
window.view_progress_bars[0] = ttk.Progressbar(window.view_progress_bar_frames[0],
                                              orient=tk.HORIZONTAL,
                                              length=100,
                                              mode = "determinate")
window.view_progress_bars[1] = ttk.Progressbar(window.view_progress_bar_frames[1],
                                              orient=tk.HORIZONTAL,
                                              length=100,
                                              mode = "determinate")

window.view_progress_bars[0]["value"] = 0
window.view_progress_bars[0].grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="ew")
window.view_progress_bar_frames[0].columnconfigure(0,weight=1)

window.view_progress_bars[1]["value"] = 0
window.view_progress_bars[1].grid(row="0",column="0",padx="0.25c",pady="0.25c", sticky="ew")
window.view_progress_bar_frames[1].columnconfigure(0,weight=1)

#add checkboxes for option buttons
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
#checkbox_metadata.append(["plot_dB", "plot in dB"])
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

promptsList = ["3","10","13"]

preamble = "Each model was trained using a single prompt: \n\n"
sentence1 = "Model 3 \n And that's what sold me on Friuli.\n"
sentence2 = "Model 10\n As a result, things look different here.\n"
sentence3 = "Model 13\n Here you'll find grey stone castles rather than sundrenched villas.\n"
combined_sentence = preamble + sentence1 +"\n" + sentence2 +"\n" + sentence3 + "\n\n\nSelect a model:"

tk.Label(window.analyze_frame,text= combined_sentence, justify = tk.CENTER).grid(row="7", column="0", padx="0.25c", pady="0.25c", sticky="ewns")

#dropdown menu with prompts
promptMenu = tk.OptionMenu(window.analyze_frame, window.selectedPrompt, *promptsList) #make the option menu
promptMenu.grid(row="8", column="0", padx="0.25c", pady="0.25c", sticky="ew")

tk.Label(window.analyze_frame,text= "").grid(row="9", column="0", padx="0.25c", pady="0.25c", sticky="ew")

#RECORD TAB####################################################################
window.path_Script = os.path.dirname(__file__)
window.path_UserRecordings = window.path_Script + '\\userrecordings' #where the user's recorded speech files are stored
#window.path_PromptAudio = window.path_Script + "/PromptAudio" #where the prompt audio files are stored

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
window.filenameField = tk.Label(window.frame_buttonsrecordSpeech, text="Filename (excluding extension)", padx=10, pady=10)
window.filenameField.grid(row="0", column="2", padx="0.25c", pady="0.25c", sticky="ew")

#creating the filename entry field
filenameEntry = tk.Entry(window.frame_buttonsrecordSpeech, textvariable = window.enteredFilename)
filenameEntry.grid(row="0", column="3", padx="0.25c", pady="0.25c", sticky="ew")

#label for the record duration field
window.recordDuration_label = tk.Label(window.frame_buttonsrecordSpeech, text="Recording Length (seconds)", padx=10, pady=10)
window.recordDuration_label.grid(row="0", column="4", padx="0.25c", pady="0.25c", sticky="ew")

#creating the record duration field
window.recordDuration_value_textbox = tk.Entry(window.frame_buttonsrecordSpeech, textvariable = window.recordDuration_value)
window.recordDuration_value.set("3") #default is 3 seconds

window.recordDuration_value_textbox.grid(row="0", column="5", padx="0.25c", pady="0.25c", sticky="ew")

#checkbox for recording -> decide whether to load resulting file 
tk.Checkbutton(window.frame_buttonsrecordSpeech, 
                text = "Load resulting file for analysis?", 
                variable = window.record_checkbox_value,
                onvalue=1,
                offvalue=0,
                justify= tk.LEFT).grid(row="0",column="6", sticky="w")

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