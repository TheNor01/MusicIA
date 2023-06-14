import csv
from os import path
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image,ImageChops
import numpy as np
import pickle
#Preprocessing
from torch.utils.data.dataset import Dataset


#image analysis
torch.set_printoptions(linewidth=200)
np.set_printoptions(threshold=2**31-1)




#How large is an image?
#Print 1 sample for fol


#rimuovere bordo bianco

doPreprocess = 0


storedPath= "./resources/archive/stored"
songs_train_test = "./resources/archive/Data/songs_train_test/"

size=100

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return trim(im.convert('RGB'))




if(doPreprocess==1):


    rootPathData = "./resources/archive/Data/images_original/"
    transformedData = "./resources/archive/Data/images_transformed_"+str(size)

    #if os.path.exists(songs_train_test+"train.txt"):
    #    os.remove(songs_train_test+"train.txt")
        


    train = open(songs_train_test+"train.txt", "a")  # append mode


    #Resize according to size 

    fileLabels = []
    tensorsLabels = []
    categoryLabels = []

    transform = transforms.ToTensor()
    if not os.path.exists(transformedData):
            os.makedirs(transformedData)

    for root, dirs, files in os.walk(rootPathData):
        path = root.split(os.sep)
        print(path[0])

        category = os.path.basename(root)
        print((len(path) - 1) * '---',category )

        

        for file in files:
            print(len(path) * '---', file)
            #image = plt.imread(path[0]+"/"+file)

            img = Image.open(path[0]+"/"+file)
            #print(img.shape) #(288, 432, 4)  , 4 canali, 288x432
            print(img.size)
            #plt.imshow(image)

            #img.show()

            #plt.imshow(image.squeeze(),cmap='gray')
            #plt.title(category)
            #plt.show()
            #print(convert_tensor(img))
            #tensor = transform(img)
            #print(im_ts)

            im_trimmed = trim(img)
            new_image = im_trimmed.resize((50, 50))
            new_image.save(transformedData+"/"+file)

            im_ts = torch.from_numpy(np.array(new_image))
            #print(np.array(new_image)[0])
        
            #print(im_ts.shape)

            fileLabels.append(file)
            tensorsLabels.append(im_ts)
            categoryLabels.append(category)

            train.write(file+","+category+"\n")

    train.close()
    dfSongs = pd.DataFrame(list(zip(fileLabels, tensorsLabels,categoryLabels)),
                columns =['file', 'tensor','label'])
    
    with open(storedPath+"/"+"dfSongs"+str(size)+".pickle", "wb") as outfile:
        pickle.dump(dfSongs, outfile)

    print(dfSongs)


class ScenesDataset(Dataset):
    

    def __init__(self,base_path,txt_list,transform=None):
     
    #conserviamo il path alla cartella contenente le immagini
        self.base_path=base_path
    #carichiamo la lista dei file
    #sarà una matrice con n righe (numero di immagini) e 2 colonne (path, etichetta)
        self.images = np.loadtxt(txt_list,dtype=str,delimiter=',')

        print(self.images.size)
    #conserviamo il riferimento alla trasformazione da applicare
        self.transform = transform

    def __getitem__(self, index):
    #recuperiamo il path dell'immagine di indice index e la relativa etichetta
        f,c = self.images[index]
    
    #carichiamo l'immagine utilizzando PIL
        im = Image.open(self.base_path+"/"+f)
    
    #se la trasfromazione è definita, applichiamola all'immagine
        if self.transform is not None:
            im = self.transform(im)
 
 #convertiamo l'etichetta in un intero
        label = str(c)
        #restituiamo un dizionario contenente immagine etichetta
        return {'image' : im, 'label':label}
        #restituisce il numero di campioni: la lunghezza della lista "images"

    def __len__(self):
        return len(self.images)



dataset = ScenesDataset('./resources/archive/Data/images_transformed_'+str(size),'./resources/archive/Data/songs_train_test/train.txt',transform=transforms.ToTensor())


sample = dataset[0]
print(sample['image'].shape)
print(sample['label'])



"""
Per poter allenare un classificatore su questi dati, abbiamo bisogno di normalizzarli in modo che essi abbiano media nulla e deviazione standard unitaria. Calcoliamo media e varianza dei
pixel contenuti in tutte le immagini del training set:
"""


#print(dfSong_restored.iloc[0]['tensor'].T.shape)


m = np.zeros(3)
print(m)
for sample in dataset:
    m+=sample['image'].sum(1).sum(1).numpy() #accumuliamo la somma dei pixel canale per canale

#dividiamo per il numero di immagini moltiplicato per il numero di pixel
m=m/(len(dataset)*256*256)
 
#procedura simile per calcolare la deviazione standard
s = np.zeros(3)
for sample in dataset:
    s+=((sample['image']-torch.Tensor(m).view(3,1,1))**2).sum(1).sum(1).numpy()
 
s=np.sqrt(s/(len(dataset)*256*256))


print("Medie",m)
print("Dev.Std.",s)



