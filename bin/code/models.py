from torch import nn

class LeNetColor(nn.Module):
    def __init__(self):
        super(LeNetColor, self).__init__() 
        #ridefiniamo il modello utilizzando i moduli sequential.
        #ne definiamo due: un "feature extractor", che estrae le feature maps
        #e un "classificatore" che implementa i livelly FC
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 18, 5), #Input: 3 x 32 x 32. Ouput: 18 x 28 x 28
            nn.MaxPool2d(2), #Input: 18 x 28 x 28. Output: 18 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(18, 28, 5), #Input 18 x 14 x 14. Output: 28 x 10 x 10
            nn.MaxPool2d(2), #Input 28 x 10 x 10. Output: 28 x 5 x 5
            nn.ReLU()
        )
 
        self.classifier = nn.Sequential(
            nn.Linear(2268, 1200), #Input: 28 * 5 * 5
            nn.ReLU(),
            nn.Linear(1200, 600),
            nn.ReLU(),
            nn.Linear(600, 100)
        )
        
 
    def forward(self,x):
        #Applichiamo le diverse trasformazioni in cascata
        x = self.feature_extractor(x)
        x = self.classifier(x.view(x.shape[0],-1))
        return x
        
