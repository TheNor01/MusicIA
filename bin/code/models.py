from torch import nn



# [(W−K+2P)/S]+1.

class LeNetColor(nn.Module):


    def calculateOutputImage(W,K,P,S):
        return ((W-K+2*P)/S)+1


    def __init__(self,sizeInput,outChannels):
        super(LeNetColor, self).__init__() 
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,int(outChannels),kernel_size=3,stride=1,padding=1), # input  3 x 100  x 100 --> outChannels x calculateOutputImage x calculateOutputImage
            nn.MaxPool2d(2), #outChannels x calculateOutputImage x calculateOutputImage -> outChannels x  calculateOutputImage/2 x calculateOutputImage/2
            nn.ReLU(),
            #nn.Conv2d(int(outChannels), int(outChannels+10), 5),
            #nn.MaxPool2d(2),
            #nn.ReLU()
        )
 
        self.classifier = nn.Sequential(
            nn.Linear(outChannels * 32 * 32, 10), #Input: 28 * 5 * 5
            #nn.ReLU(),
            #nn.Linear(int(4732/2), int(4732/4)),
            #nn.ReLU(),
            #nn.Linear(int(4732/4), 10)
        )
        
 
    def forward(self,x):
        #Applichiamo le diverse trasformazioni in cascata
        x = self.feature_extractor(x)
        x = self.classifier(x.view(x.shape[0],-1))
        return x


#size 100
class MiniAlexNet(nn.Module):
    def __init__(self, input_channels=3,outChannels=10, out_classes=10):
        super(MiniAlexNet, self).__init__() 
        #ridefiniamo il modello utilizzando i moduli sequential.
        #ne definiamo due: un "feature extractor", che estrae le feature maps
        #e un "classificatore" che implementa i livelly FC
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels,int(outChannels),kernel_size=2,stride=1,padding=0), 
            nn.MaxPool2d(2),
            nn.ReLU(),

            #----

            nn.Conv2d(int(outChannels),int(outChannels)*2,kernel_size=2,stride=1,padding=0), 
            nn.MaxPool2d(2),
            nn.ReLU(),

            # ---

            nn.Conv2d(int(outChannels)*2,int(outChannels) * 2 * 2,kernel_size=2,stride=1,padding=0), 
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
 
        self.classifier = nn.Sequential(

            nn.Dropout(),
            nn.Linear(3136, 1568),
            nn.ReLU(),

            nn.Dropout(),
            nn.Linear(1568, 784),
            nn.ReLU(),
            nn.Linear(784, out_classes)
        )
        
 
    def forward(self,x):
        #Applichiamo le diverse trasformazioni in cascata
        x = self.feature_extractor(x)
        x = self.classifier(x.view(x.shape[0],-1))
        return x