
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image


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

class ScenesDataset(Dataset):
    
    def __init__(self,base_path,txt_list,transform=None):
     
        #conserviamo il path alla cartella contenente le immagini
        self.base_path=base_path
        #carichiamo la lista dei file
        #sarà una matrice con n righe (numero di immagini) e 2 colonne (path, etichetta)
        self.images = np.loadtxt(txt_list,dtype=str,delimiter=',')

        print("Images loader :"+ str(self.images.size))
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
        label = genre_dict[c]
        #restituiamo un dizionario contenente immagine etichetta
        return {'image' : im, 'label':label}
        #restituisce il numero di campioni: la lunghezza della lista "images"

    def __len__(self):
        return len(self.images)