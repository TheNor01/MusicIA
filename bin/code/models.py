from torch import nn



# [(W−K+2P)/S]+1.

class LeNetColor(nn.Module):


    def calculateOutputImage(W,K,P,S):
        return ((W-K+2*P)/S)+1


    def __init__(self,outChannels):
        super(LeNetColor, self).__init__() 
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,int(outChannels),kernel_size=3,stride=1,padding=1),
            nn.AvgPool2d(2), 
            nn.ReLU(),
            
        )
 
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.50),
            nn.BatchNorm1d(outChannels * 32 * 32),
            nn.Linear(outChannels * 32 * 32, outChannels * 16 * 16),
            nn.ReLU(),
            nn.Linear(outChannels * 16 * 16, 10),
        
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
            nn.Conv2d(input_channels,int(outChannels),kernel_size=2,stride=1,padding=1), 
            nn.MaxPool2d(2),
            nn.ReLU(),

            #----
            nn.BatchNorm2d(int(outChannels)),
            nn.Conv2d(int(outChannels),int(outChannels)*2,kernel_size=2,stride=1,padding=1), 
            nn.MaxPool2d(2),
            nn.ReLU(),

            # ---
            nn.BatchNorm2d(int(outChannels)*2),
            nn.Conv2d(int(outChannels)*2,int(outChannels) * 2 * 2,kernel_size=2,stride=1,padding=1), 
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
 
        self.classifier = nn.Sequential(
            #16 = 4096
            nn.Dropout(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, int(4096/2)),
            nn.ReLU(),

            nn.Dropout(),
            nn.BatchNorm1d(int(4096/2)),
            nn.Linear(int(4096/2), int(4096/4)),
            nn.ReLU(),

            nn.Dropout(),
            nn.BatchNorm1d(int(4096/4)),
            nn.Linear(int(4096/4), int(4096/8)),
            nn.ReLU(),


            nn.Linear(int(4096/8), out_classes)
        )
        
 
    def forward(self,x):
        #Applichiamo le diverse trasformazioni in cascata
        x = self.feature_extractor(x)
        x = self.classifier(x.view(x.shape[0],-1))
        return x