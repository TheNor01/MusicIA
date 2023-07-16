import csv
from os import path
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
from torch import nn
import pickle
#Preprocessing
import os
from bin.code.models import LeNetColor,MiniAlexNet
from torch.utils.data import DataLoader
from PIL import ImageChops,Image
from bin.code.loaders import ImagesDataset
from bin.code.metrics_eval import train_classifier,test_classifier
from sklearn.metrics import accuracy_score
import splitfolders
import calendar
from sklearn.metrics import confusion_matrix
import time
import librosa



n_fft = 2048
hop_length = 512
n_mels=128



def trim(im):
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)
        return trim(im.convert('RGB'))


#rimuovere bordo bianco
if __name__ == '__main__':


    sounds_original = "./resources/archive/Data/genres_original"
    image_dataset_new = "./resources/archive/Data/images_newset/"

    if not os.path.exists(image_dataset_new):
                os.makedirs(image_dataset_new)

    size=64

    
    for root, dirs, files in os.walk(sounds_original):
        path = root.split(os.sep)
        print(path[0])

        print(dirs)

        category = os.path.basename(root)
        print((len(path) - 1) * '---',category )

        if not os.path.exists(image_dataset_new+"/"+category):
            os.makedirs(image_dataset_new+"/"+category)

        for file in files:
            print(len(path) * '---', file)

            fullPath = path[0]+"/"+category+"/"+file
            print("librosa load")
            audio_recording=fullPath
            #plt.figure(figsize=(12,4))
            #librosa.display.waveshow(data,color="#2B4F72")

            try:
                data,sr=librosa.load(audio_recording)
                S = librosa.feature.melspectrogram(y=data, sr=sr)
                S_DB = librosa.amplitude_to_db(S, ref=np.max)
                fig, ax = plt.subplots(1, figsize=(12,8))
                #librosa.display.specshow(DB, sr = sr, hop_length = hop_length, x_axis = 'time', y_axis = 'log')
                librosa.display.specshow(S_DB, sr = sr, hop_length = hop_length, x_axis = 'time', y_axis = 'log')

                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_frame_on(False)
                ax.set_xlabel(None)
                ax.set_ylabel(None)
                #plt.show()
                plt.savefig(image_dataset_new+category+"/"+file.replace(".wav",".png"),bbox_inches='tight', pad_inches=0)
                plt.cla()
                plt.clf()
                plt.close('all')
                #print("saved:"+ image_dataset_new+category+"/"+file.replace(".wav",".png"))
            except:
                print("SKIPPED: "+image_dataset_new+category+"/"+file.replace(".wav",".png"))



    

