# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:13:37 2020

@author: Ben
"""

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import parselmouth
import PIL
import ntpath

import numpy as np
import os

def spectrogram(file_path):
        
    snd = parselmouth.Sound(file_path)
    
    spectrogram = snd.to_spectrogram()
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    plt.xlim([snd.xmin, snd.xmax])
    filename = ntpath.basename(file_path)
    imgname = os.path.splitext(filename)[0] + '.png'
    plt.savefig(os.path.dirname(__file__) + '\\userspectrograms\\' + imgname)

    return imgname
    
def draw_spectrogram(spectrogram, dynamic_range = 70):
    time, freq = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(time, freq, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    
def getimagesize(file):
    img = PIL.Image.open(file)
    width, height = img.size
    return width, height

def classify(modelnum, file_path):
    
    file_path = file_path.replace("/","\\")
    
    print("path is:  " + os.path.dirname(__file__) + '\model\model_sentence_' + str(modelnum) + '.h5')
    model = load_model(os.path.dirname(__file__) + '\model\model_sentence_' + str(modelnum) + '.h5')
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    
    print("LOL  "+ file_path)
    imgname = spectrogram(file_path) #spectrogram
    imgpath = os.getcwd() + '\\userspectrograms\\' + imgname
    img_width, img_height = getimagesize(imgpath)
    img = image.load_img(imgpath, target_size = (img_width, img_height))#load the img
        
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = img.reshape(-1, img_width, img_height, 3)
    result = model.predict_classes(img)
        
    print(result)
    return result
