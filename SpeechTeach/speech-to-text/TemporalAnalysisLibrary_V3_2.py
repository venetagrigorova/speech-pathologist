import matplotlib.pyplot as plt
import math
import numpy as np
import wave
import sounddevice as sound_device
import sys
import scipy
from scipy import signal as sig
from aubio import source, onset
import aubio

###############################################################################
def listSilences(signal,fs=44100, threshold_dB = 65.0, number_of_segments = 512, min_silence_samples = 1000, ):
    
    signal = signal.copy()
    
    #define process parameters
    segments = number_of_segments  #evenly spaced
    increment = len(signal)//segments
    
    #generate array of silences
    chunks = []
    silence_array = []
    consecutive_silences = 0
    silence_timestamps = []
    current_silence = []
    for i in range(segments): 
        
        #we check the chunks individually
        is_silent = aubio.silence_detection(aubio.fvec(signal[i*increment:(i+1)*increment]),threshold_dB)
        chunks.append(i*increment*1/fs)
        
        #implement minimum number of consecutive silent chunks
        if(is_silent):
            
            if(consecutive_silences == 0):
                current_silence = []
                current_silence.append(i*increment)
            
            consecutive_silences +=1         
            
        else: #"not silent" encountered
            
            if(consecutive_silences > 0):
                
                current_silence.append(i*increment)
                
                #only append if silence is sufficiently long
                if(current_silence[1] - current_silence[0] > min_silence_samples):
                    silence_timestamps.append(current_silence)
                
            consecutive_silences = 0
            
        silence_array.append(is_silent)
    
    current_silence.append(i*increment)
            
    #only append if silence is sufficiently long
    if(current_silence[1] - current_silence[0] > min_silence_samples):
        silence_timestamps.append(current_silence)
            
    #replace silent segments with zeros
    for i in range(len(signal)):
        
        current_increment = i//increment
        
        if(current_increment>segments-2):
            break
        
        if(silence_array[current_increment] == 1):
            signal[i] = 0 #20000*np.sin(i), for testing, can replace with a sound
    
    return silence_timestamps, signal

###############################################################################
def silencesToSoundChunks(silence_timestamps, min_chunk_samples = 1000):
    
    #assume sound clip starts and ends with a silence
    i=0
    chunk_list = []
    current_sound = []
    
    #go through timestamps and "flip" inside-out...sortof
    for silence in silence_timestamps:
        if i==0:
            current_sound.append(silence[1])
        else:
            current_sound.append(silence[0])
            
            #only include a chunk if it is large enough
            if(current_sound[1] - current_sound[0] > min_chunk_samples):
                chunk_list.append(current_sound)
            
            current_sound = [silence[1]]
        i+=1
    
    return chunk_list

###############################################################################
def findPeaks(signal, chunks):
    
    peak_values = []
    peak_locations = []
    
    for chunk in chunks:
        current_chunk = list(signal[chunk[0]:chunk[1]])
        
        peak_values.append(max(current_chunk))
        index = current_chunk.index(max(signal[chunk[0]:chunk[1]]))
        peak_locations.append(index + chunk[0]  )
        
    return peak_values, peak_locations
 
###############################################################################
def plotEvents_seconds(eventList, colour = 'red', fs=44100, plot_object = plt ):
    
    for event in eventList:
        
        plot_object.axvline(x=event*1/fs, color = colour, alpha=50)
        
###############################################################################    
def playChunks_fromSingleIndices(signal, onsets):

    print("\n\nPlaying signal chunks...")
    print(len(onsets)-2, " chunks to play \n enter to continue...")
    input()
    
    j = 1
    for i in onsets:
        if(j<len(onsets)-1):
            print("Playing chunk",j)
            print("samples: ",i," to ", onsets[j])
            sound_device.play(signal[i: onsets[j]])
            j+=1
            input("enter to continue...\n\n")
        
###############################################################################    
def playChunks_fromIndexPairs(signal, index_pairs):

    print("\n\nPlaying signal chunks...")
    print(len(index_pairs), " chunks to play \n enter to continue...")
    input()
    
    j = 1
    for pair in index_pairs:
        print("Playing chunk",j)
        print("samples: ",pair[0]," to ", pair[1])
        sound_device.play(signal[pair[0]: pair[1]])
        j+=1
        input("enter to continue...\n\n")    

###############################################################################
def RMS(data, in_dB = False):
    '''return RMS value of a list'''
    squares = np.square(data, dtype='int64')
    sum_of_squares = np.sum(squares, dtype='int64')
    #print(sum_of_squares)
    
    RMS_value = np.sqrt(sum_of_squares / len(data))
    
    if in_dB:
        return 20*np.log10(RMS_value)
    else:
        return RMS_value
    
###############################################################################  
def normalizeToRMSValue(signal, RMS_value):
    #use RMS value to solve for gain factor for each sample
    
    n = len(signal)
    R = RMS_value
    sum_of_squares = np.sum(np.square(signal))
    
    print("\nNormalizing", n, "samples to RMS value",RMS_value)
    
    gain_array = []
    new_signal = []
    for sample in signal:
        gain_factor =np.sqrt(float(n*R**2)/sum_of_squares)         
        #sample = sample * gain_factor
        new_signal.append(sample*gain_factor)
        gain_array.append(gain_factor)
    
    print("average gain factor: ",np.average(gain_array))
    return new_signal

###############################################################################
def analyze_chunks(signal, chunks):
    
    peaks_list_1, peak_locations_1 = findPeaks(signal, chunks)
    signal_abs = list(np.abs(signal))
    prev_chunk = None
    chunk_data = []
    i=0
    
    for chunk in chunks:
        current_dict = {} #store information for a chunk in a dict 
      
        #Analyse current chunk
        current_dict["length"] = chunk[1]-chunk[0]
        current_dict["max_value"] = peaks_list_1[i]
        current_dict["max_position"] = peak_locations_1[i]
        current_dict["mean"] = np.round(np.average(signal_abs[chunk[0]:chunk[1]]),2)
        current_dict["median_value"] = np.round(np.median(signal_abs[chunk[0]:chunk[1]]),2)
        current_dict["standard_deviation"] = np.round(np.std(signal_abs[chunk[0]:chunk[1]]),2)
        current_dict["RMS_value"] = np.round(RMS(signal_abs[chunk[0]:chunk[1]]),2)
      
        #analyse preceding silennce
        preceding_silence = [None,chunk[0]]
      
        #handle case for first chunk
        if(prev_chunk != None):
            preceding_silence[0] = prev_chunk[1]
        else:
            preceding_silence[0] = 0 #assumes first silence starts at zero
    
        current_dict["preceding_silence_length"] = preceding_silence[1] - preceding_silence[0]
    
        prev_chunk = chunk
        chunk_data.append(current_dict)    
        i+=1
      
    return chunk_data

###############################################################################
def find_neighbours(chunks_1, chunks_2):
    
    '''  
        Returns a list of neighbours (by index number)
        Neighbours are chunks from othe other file that overlap with a given chunk
    
    '''
    neighbours_1=[]
    neighbours_2=[]
    
    #chunk_data_1 = analyze_chunks(signal_1,chunks_1)
    #chunk_data_2 = analyze_chunks(signal_2,chunks_2)
    
    i=0
    for chunk in chunks_1:
        current_chunk = []
        i=0
        for other_side in chunks_2: 
            #check if a chunk from other file ends in this chunk
            if(other_side[1] > chunk[0] and other_side[1] < chunk[1]):
                current_chunk.append(i)
            #check if a chunk from other file begins in this chunk
            elif(other_side[0] > chunk[0] and other_side[0] < chunk[1]):
                current_chunk.append(i)
            #check if chunk is contained by other chunks
            elif(other_side[0] < chunk[0] and other_side[1] > chunk[1]):
                current_chunk.append(i)
            i+=1
        neighbours_1.append(current_chunk)    
        
    for chunk in chunks_2:
        current_chunk = []
        i=0
        for other_side in chunks_1: 
            #check if a chunk from other file ends in this chuck
            if(other_side[1] > chunk[0] and other_side[1] < chunk[1]):
                current_chunk.append(i)
            #check if a chunk from other file begins in this chunk
            elif(other_side[0] > chunk[0] and other_side[0] < chunk[1]):
                current_chunk.append(i)
            #check if chunk is contained by other chunks
            elif(other_side[0] < chunk[0] and other_side[1] > chunk[1]):
                current_chunk.append(i)
            i+=1
        neighbours_2.append(current_chunk)
    
    return neighbours_1, neighbours_2

###############################################################################
def merge_chunks(signal_1,chunks_1,signal_2,chunks_2):
    
    #define maximum ratio
    max_ratio = 1.33
    
    if(len(chunks_1) == len(chunks_2)):
        print("Chunk Array Lengths are the same")
        return chunks_1,chunks_2
    
    #determine which file needs to have chunks merged
    if(len(chunks_1) > len(chunks_2)):
        merge_signal = signal_1
        merge_chunks = chunks_1
        ref_signal = signal_2
        ref_chunks = chunks_2
    else:
        merge_signal = signal_2
        merge_chunks = chunks_2
        ref_signal = signal_1
        ref_chunks = chunks_1
    
    merge_neighbours, ref_neighbours = find_neighbours(merge_chunks,ref_chunks)
    
    to_be_removed = []
    
    i=0
    for neighbour_set in ref_neighbours:
        i+=1
        print("ref chunk ", i-1, "has neighbours",neighbour_set)
        
        if(not len(neighbour_set)<=1):
            
            #Check that resulting chunk is not much bigger than reference chunk
            print(">> reference chunk size: ", ref_chunks[i-1][1]-ref_chunks[i-1][0], end="\t")
            print("combined chunk size: ", merge_chunks[neighbour_set[-1]][1] - merge_chunks[neighbour_set[0]][0], end="\t")
            print("ratio = ", (merge_chunks[neighbour_set[-1]][1] - merge_chunks[neighbour_set[0]][0])/(ref_chunks[i-1][1]-ref_chunks[i-1][0]))
            
            ratio = (merge_chunks[neighbour_set[-1]][1] - merge_chunks[neighbour_set[0]][0])/(ref_chunks[i-1][1]-ref_chunks[i-1][0])
            if ratio < max_ratio:
                merge_chunks[neighbour_set[0]][1] = merge_chunks[neighbour_set[-1]][1]
                
                for elem in neighbour_set:
                    if(elem != neighbour_set[0]):
                        to_be_removed.append(elem)
     
    print("removed: ", to_be_removed)
    for elem in to_be_removed:
        merge_chunks.remove(merge_chunks[elem])
    
#    #1 Handle case where one chunk is fully contained by another
    
#    
#    i=0
#    for chunk in merge_neighbours:
#        
#        #chunk has only one neighbour
#        if(len(chunk)==1):
#            #chunk begins and ends within bounds of neighbour
#            if(ref_chunks[chunk[0]][0]<merge_chunks[i][0] and ref_chunks[chunk[0]][1]>merge_chunks[i][1]):
#                print("Chunk ",chunk[0]," of ref signal contains chunk ",i)
#                print("check if chunk ",(chunk[0]-1)," is in ",ref_neighbours[chunk[0]])
#                #if previous chunk shares neighbour, merge with it (e.g. replace previous chunk ending with current chunk ending)
#                if((chunk[0]-1) in ref_neighbours[chunk[0]]):
#                    print("Chunk ",chunk[0]," of ref signal contains chunk ",i-1)
#                    merge_chunks[chunk[0]-1][1]= merge_chunks[i][1]
#
#        i+=1
    
    if(len(chunks_1) > len(chunks_2)):
        return merge_chunks, ref_chunks
    else:
        return ref_chunks, merge_chunks 
    
###############################################################################
    
def computeSAD(d1, d2):
    print("test")
    return 0#list(np.asarray(data1) - np.asarray(data2))
        
#------------------------------------------------------------------------------
#DEFUNCT
#----------------------------------------------------------------------
###############################################################################
def listOnsets(filename, fs = 44100, method = "specdiff", min_onset_delay = 0.25, threshold = 5000):   
    #INITIAL ATTEMPT, NOT REALLY USEFUL
    
    #AUBIO parameters
    win_s = 512 #fft size
    hop_s = win_s//2 #hop size
    
    #open the audio file
    s = source(filename, fs, hop_s)
    
    #extract parameters from audio
    samplerate = s.samplerate
    o = onset(method, win_s, hop_s, samplerate)
    
    # list of onsets, in samples
    onsets = []
    total_frames = 0
    while True:
        samples, read = s()
        if o(samples):
            if(len(onsets) < 1):
                print("%f" % o.get_last_s())
                onsets.append(o.get_last())
            elif (o.get_last_s() - (onsets[len(onsets)-1])/fs > min_onset_delay):
                print(onsets[len(onsets)-2]/fs)
                print("%f " % o.get_last_s(), end = " ")
                print("")
                onsets.append(o.get_last())
        total_frames += read
        if read < hop_s: break    
    return onsets
        
###############################################################################
def applyNoiseGate(signal, threshold, method = "absolute", segments = 16):
    
    #sets samples below a certain threshold to zero
    #two methods: absolute compares to an absolute value threshold
    if method == "absolute":
        signal[signal<threshold] = 0 
        
    #relative: applies the gate based on a percentage of peak value
    #within evenly spaced segments (as given by segments argument)
    
    #<TO BE IMPLEMENTED>
    return signal