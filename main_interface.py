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
import librosa
import librosa.display



#image analysis
torch.set_printoptions(linewidth=200)
np.set_printoptions(threshold=2**31-1)





#rimuovere bordo bianco
if __name__ == '__main__':
   

    print("LOAD your .wav")
    audio_recording="./resources/archive/Data/genres_original/classical/classical.00000.wav"
    data,sr=librosa.load(audio_recording)
    plt.figure(figsize=(12,4))
    librosa.display.waveshow(data,color="#2B4F72")

    plt.show()

    print(type(data),type(sr))
    stft=librosa.stft(data)
    stft_db=librosa.amplitude_to_db(abs(stft),ref=np.max)
    plt.figure(figsize=(14,6))
    librosa.display.specshow(stft_db,sr=sr,x_axis='time',y_axis='hz')
    plt.colorbar()

    plt.show()