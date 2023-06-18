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
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk, Image

screen = tk.Tk()

globalPath = ""

if __name__ == '__main__':
   
    def select_file():
        file_path = filedialog.askopenfilename(defaultextension=".wav",initialdir="./resources/archive/Data/genres_original")
        print("File selezionato:", file_path)
        

        if file_path:

            globalPath = file_path
            path_entry.insert(0,globalPath)

            print("librosa load")
            audio_recording=file_path
            data,sr=librosa.load(audio_recording)
            #plt.figure(figsize=(12,4))
            #librosa.display.waveshow(data,color="#2B4F72")
            
            #print(type(data),type(sr))
            stft=librosa.stft(data)
            stft_db=librosa.amplitude_to_db(abs(stft),ref=np.max)
            plt.figure(figsize=(14,6))
            librosa.display.specshow(stft_db,sr=sr,x_axis='time',y_axis='hz')
            plt.colorbar()
            #plt.show()

            plt.savefig("./resources/interface/"+"audio_spectrum.png")

            imageToLoad = Image.open("./resources/interface/"+"audio_spectrum.png")
            imageToLoad = imageToLoad.resize([300,300])
            img = ImageTk.PhotoImage(imageToLoad)
            imagebox.config(image=img)
            imagebox.image = img


            #trovare la ground 
            #file_path.



            true_label.insert(0,"True label")
            
   

    screen.minsize(400,400)
    screen.title("Classify genre")

    select_button = tk.Button(screen, text="Load your .wav audio", command=select_file)
    select_button.pack()

    path_entry = tk.Entry(screen,textvariable = globalPath, width = "100")
    path_entry.pack(side="top")

    true_entry = tk.Text(screen,textvariable = "True Label", width = "20")
    true_entry.pack(side="right")

    true_label = tk.Entry(screen,textvariable = "", width = "20")
    true_label.pack(side="right")

    filename_audio= tk.StringVar()

    # label to show the image
    imagebox = tk.Label(screen)
    imagebox.pack(side="left")
    

    screen.mainloop()

    


    """

   
    """


    