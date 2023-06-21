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
from bin.code.models import LeNetColor,MiniAlexNet
from PIL import ImageChops,Image

screen = tk.Tk()

globalPath = ""
n_fft = 2048
hop_length = 512
n_mels=128
globalImage = None

if __name__ == '__main__':


    #load model
    #model = LeNetColor(outChannels=16).to("cpu")
    model = MiniAlexNet(outChannels=16).to("cpu")
    model.load_state_dict(torch.load("./resources/archive/stored/models/miniAlex.pth"))
    
    print(model)
    
    model.eval()

    def trim(im):
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)
        return trim(im.convert('RGB'))
   
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


            D = np.abs(librosa.stft(data, n_fft=n_fft,  hop_length=hop_length))
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
            DB = librosa.amplitude_to_db(D, ref=np.max)
            #plt.colorbar(format='%+2.0f dB')


            #S = librosa.feature.melspectrogram(data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            #S_DB = librosa.power_to_db(S, ref=np.max)
            #librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
            #plt.colorbar(format='%+2.0f dB')

            #plt.colorbar();
            
            #print(type(data),type(sr))
            #stft=librosa.stft(data)
            #stft_db=librosa.amplitude_to_db(abs(stft),ref=np.max)
            #plt.figure(figsize=(14,6))
            #librosa.display.specshow(stft_db,sr=sr,x_axis='time',y_axis='hz')
            
            #plt.colorbar()
            #plt.show()
            #plt.savefig('out.png', bbox_inches='tight', pad_inches=0)
            fig, ax = plt.subplots(1, figsize=(12,8))
            imgSpec=librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')

            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            plt.savefig("./resources/interface/"+"audio_spectrum.png",bbox_inches='tight', pad_inches=0)

            imageToLoad = Image.open("./resources/interface/"+"audio_spectrum.png")
            im_trimmed = trim(imageToLoad)

            im_trimmed.save("./resources/interface/"+"audio_spectrum.png")
            im_trimmed = im_trimmed.resize([200,200])
            img = ImageTk.PhotoImage(im_trimmed)
            imagebox.config(image=img)
            imagebox.image = img
            #globalImage = im_trimmed

            print("Gp:"+globalPath)

            indexSlash = globalPath.rfind('/')
            categoryPath = globalPath[indexSlash:].split(".")[0][1:]
            print(categoryPath)


            #trovare la ground  .split(".")(0)
            #file_path.
            true_label.insert(0,categoryPath)
            
    def classify():

    
        print("classify: "+ path_entry.get())
        imageToLoad = Image.open("./resources/interface/"+"audio_spectrum.png")

        tf=transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor()
        ])


        imageTransformed = tf(imageToLoad)

        print(imageTransformed.shape)

        output = model(imageTransformed)
        preds = output.to('cpu').max(1)[1].numpy()

        print(preds)

        return "x"

    screen.minsize(400,400)
    screen.title("Classify genre")

    select_button = tk.Button(screen, text="Load your .wav audio", command=select_file)
    select_button.grid(row=1)

    path_entry = tk.Entry(screen,textvariable = globalPath, width = "100")
    path_entry.grid(column=0,row=0)

    true_entry = tk.Label(screen, text = "true entry", width = "20")
    true_entry.grid(column = 0, row = 5)

    true_label = tk.Entry(screen,textvariable = "", width = "20")
    true_label.grid(column=0,row=7)

    filename_audio= tk.StringVar()

    # label to show the image
    imagebox = tk.Label(screen)
    imagebox.grid(column=0,row=350)
    

    classify_button = tk.Button(screen, text="Classify", command=classify)
    classify_button.grid(column=0,row=400)

    screen.mainloop()

    


    """

   
    """


    