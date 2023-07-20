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

genre_dict = {
"blues" : 0,
"classical": 1,
"country": 2,
"disco": 3,
"hiphop": 4,
"jazz": 5,
"metal": 6,
"pop": 7,
"reggae": 8,
"rock" : 9
}


listbox = tk.Listbox(screen, selectmode=tk.SINGLE)
classifierNames = ['alexNet', 'lenetColor']
for classifier in classifierNames:
    listbox.insert(0, classifier)

reversedDict = dict((v, k) for k, v in genre_dict.items())

#alert if audio is not a .wav

if __name__ == '__main__':


    #load model
    modelLenet = LeNetColor(outChannels=16).to("cpu")
    modelLenet.load_state_dict(torch.load("./resources/archive/stored/models/leNet.pth"))
    modelLenet.eval()
    
    
    modelAlex = MiniAlexNet(outChannels=16).to("cpu")
    modelAlex.load_state_dict(torch.load("./resources/archive/stored/models/miniAlex.pth"))
    modelAlex.eval()

    
    model = None

    def trim(im):
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)
        return trim(im.convert('RGB'))
   
    def select_file():

        true_label.delete(0,'end')
        file_path = filedialog.askopenfilename(defaultextension=".wav",initialdir="./resources/archive/Data/genres_original")
        if(not file_path.endswith(".wav")):
            tk.messagebox.showerror(title="ERROR", message= "Not a valid file")
            return


        print("File selezionato:", file_path)
        

        if file_path:

            globalPath = file_path
            path_entry.insert(0,globalPath)

            print("librosa load")
            audio_recording=file_path
            data,sr=librosa.load(audio_recording)
            data, _ = librosa.effects.trim(data)
            #plt.figure(figsize=(12,4))
            #librosa.display.waveshow(data,color="#2B4F72")

            S = librosa.feature.melspectrogram(y=data, sr=sr)
            #D = np.abs(librosa.stft(data, n_fft=n_fft,  hop_length=hop_length))
            #librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
            #DB = librosa.amplitude_to_db(D, ref=np.max)
            S_DB = librosa.amplitude_to_db(S, ref=np.max)
            #plt.colorbar(format='%+2.0f dB')


           
            fig, ax = plt.subplots(1, figsize=(16,6))
            #librosa.display.specshow(DB, sr = sr, hop_length = hop_length, x_axis = 'time', y_axis = 'log')
            librosa.display.specshow(S_DB, sr = sr, hop_length = hop_length, x_axis = 'time', y_axis = 'log')

            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            plt.savefig("./resources/interface/"+"audio_spectrum.png",bbox_inches='tight', pad_inches=0)

            imageToLoad = Image.open("./resources/interface/"+"audio_spectrum.png")
            imageToLoad = trim(imageToLoad)

            print(imageToLoad)

            imageToLoad.save("./resources/interface/"+"audio_spectrum.png")
            imageToLoad = imageToLoad.resize([64,64])
            img = ImageTk.PhotoImage(imageToLoad)
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

        choosenModel=None

        valueModel = [listbox.get(idx) for idx in listbox.curselection()]
        if(len(valueModel)==0):
             model = None
        else:
            choosenModel = valueModel[0]

        if(choosenModel=="alexNet"):
            model = modelAlex
        elif(choosenModel=="lenetColor"):
            model = modelLenet
        else:
            model = None


        if(model is None):
            tk.messagebox.showerror(title="ERROR", message="CHOOSE ONE MODEL")
            return

        predicted_label.delete(0,'end')
        if not path_entry.get():
            tk.messagebox.showerror(title="ERROR", message="PATH EMPTY")
            return

        print("classify: "+ path_entry.get())
        imageToLoad = Image.open("./resources/interface/"+"audio_spectrum.png")
        new_image = imageToLoad.resize((64, 64))


        #Medie [0.36204   0.14410875 0.30086741]
        #Dev.Std. [0.29978628 0.14587127 0.18268844]

        m = np.array([0.36204 , 0.14410875, 0.30086741])
        s = np.array([0.29978628 , 0.14587127, 0.18268844])

        tf=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(m,s)
            ])


        imageTransformed = tf(new_image).unsqueeze(0)

        print(imageTransformed.shape)
        #model.eval()
        with torch.no_grad():
            output = model(imageTransformed.to("cpu"))


        
        _, predicted_idx = torch.max(output, 1)
        label = reversedDict[predicted_idx.item()]

        predicted_label.insert(0,label)
        print("Classified label:"+label)

    def clear():
        true_label.delete(0,'end')
        path_entry.delete(0,'end')
        predicted_label.delete(0,'end')
        imagebox.config(image='')

    screen.minsize(400,400)
    screen.title("Classify genre")

    select_button = tk.Button(screen, text="Load your .wav audio", command=select_file)
    select_button.grid(row=1)

    path_entry = tk.Entry(screen,textvariable = globalPath, width = "100")
    path_entry.grid(column=0,row=0)

    true_entry = tk.Label(screen, text = "true entry", width = "20")
    true_entry.grid(column = 0, row = 5)

    true_label = tk.Entry(screen,textvariable = "", width = "20",justify='center')
    true_label.grid(column=0,row=7)


    predicted_entry = tk.Label(screen, text = "label classified", width = "40")
    predicted_entry.grid(column = 0, row = 10)

    predicted_label = tk.Entry(screen,textvariable = "", width = "40",justify='center')
    predicted_label.grid(column=0,row=20)

    filename_audio= tk.StringVar()

    # label to show the image
    imagebox = tk.Label(screen)
    imagebox.grid(column=0,row=350)
    

    classify_button = tk.Button(screen, text="Classify", command=classify)
    classify_button.grid(column=0,row=400)

    clear_button = tk.Button(screen, text="Clear", command=clear)
    clear_button.grid(column=5)

    listbox.grid(row=400,column=5)



    screen.mainloop()



    