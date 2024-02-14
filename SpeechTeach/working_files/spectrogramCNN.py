# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 23:08:04 2020

@author: Ben
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import parselmouth

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D
from IPython.display import SVG
from keras.utils import model_to_dot
from keras.utils import plot_model

def renamefiles():
    os.chdir('./newfiles')
    for file in os.listdir():
        print(file[-5])
        if file[-5] == '5':
            ext = '13.wav'
            name = file[:-5]
            print(name)
            os.rename(file, name + ext) 
    return

def spectrogram(wavfile, num):
    snd = parselmouth.Sound(wavfile)
    print(wavfile)
    spectrogram = snd.to_spectrogram()
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    plt.xlim([snd.xmin, 6])
    ax = plt.axes()
    ax.set_facecolor("black")
    plt.show
    plt.savefig('../trainingdata_'+ num +'/' + os.path.splitext(wavfile)[0] + '.png')
    
def spectrogram_normalized(wavfile, num):
    snd = parselmouth.Sound(wavfile)
    intensity = snd.to_intensity()
    spectrogram = snd.to_spectrogram()
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_intensity(intensity)
    plt.xlim([snd.xmin, snd.xmax])
    plt.show
    plt.savefig('../trainingdata_'+ num +'/' + os.path.splitext(wavfile)[0] + '.png')

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")
    
def draw_spectrogram(spectrogram, dynamic_range = 70):
    time, freq = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(time, freq, sg_db, vmin=sg_db.max() - dynamic_range, cmap = 'afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def organizeData(num):
    os.chdir('./Sentence_' + num);
    for wavfile in os.listdir():
        if wavfile != 'desktop.ini': #error case
            spectrogram_normalized(wavfile, num)
    os.chdir('..')
    return

def getimagesize():
    os.chdir('./trainingdata_3/test')
    for file in os.listdir():
        img = PIL.Image.open(file)
        width, height = img.size
    os.chdir('../../')
    return width, height
    
def trainModel(sentence_num):
    img_width, img_height = getimagesize()
    train_dir = 'trainingdata_3/train'
    validation_dir = 'trainingdata_3/valid'
    
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(train_dir, target_size = (img_width,img_height),batch_size = 2,class_mode = 'categorical')
    validation_gen = datagen.flow_from_directory(validation_dir, target_size = (img_width,img_height),batch_size =2,class_mode = 'categorical')
    
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(64)) #Dense layer of dimension 64
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Dense(2, activation='sigmoid')) #Must have an output with size of dimension 2
    model.summary()
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics = ['accuracy'])
    epochs = 30
    
    if sentence_num == '3':
        
        train_samples = 21
        validation_samples = 7
        
    if sentence_num == '10':
        train_samples = 21
        validation_samples = 12
        
    if sentence_num == '13':
        train_samples = 22
        validation_samples = 10
                          
    model.fit_generator(train_gen, samples_per_epoch = train_samples, epochs = epochs, validation_data = validation_gen, nb_val_samples= validation_samples)
    model.save('model/model_sentence_'+ sentence_num +'.h5')
    model.evaluate_generator(validation_gen, validation_samples)

    SVG(model_to_dot(model).create(prog='dot', format='svg'))
    plot_model(model, to_file='model.png')
    
    print('Model Trained and Saved!')

    return

def main():
    os.chdir('Sentence_3')
    spectrogram_normalized  ('024_3.wav','3')  
main()  