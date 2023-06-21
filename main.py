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


#image analysis
torch.set_printoptions(linewidth=200)
np.set_printoptions(threshold=2**31-1)




#How large is an image?
#Print 1 sample for fol


#rimuovere bordo bianco
if __name__ == '__main__':
    doPreprocess = 0


    storedPath= "./resources/archive/stored"
    songs_train_test = "./resources/archive/Data/songs_train_test/"

    size=64

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
            
        fullDs = open(songs_train_test+"fullDs.txt", "a")  # append mode


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

            print(dirs)

            category = os.path.basename(root)
            print((len(path) - 1) * '---',category )

            if not os.path.exists(transformedData+"/"+category):
                os.makedirs(transformedData+"/"+category)

        
            for file in files:
                print(len(path) * '---', file)
                #image = plt.imread(path[0]+"/"+file)

                img = Image.open(path[0]+"/"+file)
                #print(img.shape) #(288, 432, 4)  , 4 canali, 288x432
                print(img.size)


                im_trimmed = trim(img)
                new_image = im_trimmed.resize((size, size))
                new_image.save(transformedData+"/"+category+"/"+file)

            
                im_ts = torch.from_numpy(np.array(new_image))
            

                fileLabels.append(file)
                tensorsLabels.append(im_ts)
                categoryLabels.append(category)

                fullDs.write(category+"/"+file+","+category+"\n")

        fullDs.close()
        dfSongs = pd.DataFrame(list(zip(fileLabels, tensorsLabels,categoryLabels)),
                    columns =['file', 'tensor','label'])
        
        with open(storedPath+"/"+"dfSongs"+str(size)+".pickle", "wb") as outfile:
            pickle.dump(dfSongs, outfile)

        print(dfSongs)

        splitfolders.ratio("./resources/archive/Data/images_transformed_"+str(size), output="./resources/archive/Data/",seed=1337, ratio=(.7, .2, .1), group_prefix=None, move=False) # default values


    dataset = ImagesDataset('./resources/archive/Data/images_transformed_'+str(size),'./resources/archive/Data/songs_train_test/fullDs.txt',transform=transforms.ToTensor())

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
        m+=sample['image'].sum(1).sum(1).numpy() 

    m=m/(len(dataset)*size*size)
    
    s = np.zeros(3)
    for sample in dataset:
        s+=((sample['image']-torch.Tensor(m).view(3,1,1))**2).sum(1).sum(1).numpy()
    
    s=np.sqrt(s/(len(dataset)*size*size))


    print("Medie",m)
    print("Dev.Std.",s)



    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(m,s),
                                    ])


    trainTxt = open('./resources/archive/Data/songs_train_test/train_loader.txt', "w+")
    valTxt = open('./resources/archive/Data/songs_train_test/val_loader.txt', "w+")

    
    for (dir_path, dir_names, file_names) in os.walk('./resources/archive/Data/train'):
        category = os.path.basename(dir_path)
        for file in file_names:
            print(category+"/"+file)
            trainTxt.write(category+"/"+file+","+category+"\n")


    for (dir_path, dir_names, file_names) in os.walk('./resources/archive/Data/val'):
        category = os.path.basename(dir_path)
        for file in file_names:
            print(category+"/"+file)
            valTxt.write(category+"/"+file+","+category+"\n")


    trainTxt.close()
    valTxt.close()

    dataset_train = ImagesDataset('./resources/archive/Data/train','./resources/archive/Data/songs_train_test/train_loader.txt',transform=transform)
    dataset_test = ImagesDataset('./resources/archive/Data/val','./resources/archive/Data/songs_train_test/val_loader.txt',transform=transform)


    print(len(dataset_train))
    print(len(dataset_test))


    print(dataset_train[0]['image'].shape)
    print(dataset_train[0]['label'])

    print(dataset_test[0]['image'].shape)
    print(dataset_test[0]['label'])


    train_dataset = DataLoader(dataset_train,batch_size=32,num_workers=0,shuffle=True)
    test_dataset = DataLoader(dataset_test,batch_size=32,num_workers=0,shuffle=2)

    current_GMT = time.gmtime()

    # ts stores timestamp
    ts = calendar.timegm(current_GMT)
    print("Current timestamp:", ts)

    """
    lenetModel = LeNetColor(sizeInput=size,outChannels=16)


    for i_batch, sample_batched in enumerate(train_dataset):
        print(i_batch, sample_batched['image'].size())


    lenet_mnist = train_classifier(lenetModel, train_dataset,test_dataset, exp_name=str(ts)+"_color", epochs = 400,lr=0.01,momentum=0.9)


    torch.save(lenet_mnist.state_dict(), "./resources/archive/stored/models/"+"leNet.pth")

    lenet_train_predictions, cifar100_labels_train = test_classifier(lenet_mnist,train_dataset)
    lenet_test_predictions, cifar100_labels_test = test_classifier(lenet_mnist,test_dataset)

    print("Accuracy train LeNetColor: %0.2f" % accuracy_score(cifar100_labels_train,lenet_train_predictions))
    print("Accuracy test LeNetColor: %0.2f" % accuracy_score(cifar100_labels_test,lenet_test_predictions))

    """
    
    #improve cnn
    miniAlex = MiniAlexNet(outChannels=16)
    alex_mnist = train_classifier(miniAlex, train_dataset,test_dataset, exp_name=str(ts)+"_alex", epochs = 400,lr=0.001)


    torch.save(alex_mnist.state_dict(), "./resources/archive/stored/models/"+"miniAlex.pth")


    alex_train_predictions, alex_labels_train = test_classifier(alex_mnist,train_dataset)
    alex_test_predictions, alex_labels_test = test_classifier(alex_mnist,test_dataset)


    print("Accuracy train Alex: %0.2f" % accuracy_score(alex_labels_train,alex_train_predictions))
    print("Accuracy train Alex: %0.2f" % accuracy_score(alex_labels_test,alex_test_predictions))

    confusion_matrix(alex_labels_test, alex_test_predictions)
    #what if we increase samples training? overfittin is here
    

