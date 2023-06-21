from bin.code.models import LeNetColor,MiniAlexNet
import torch

lenetModel = LeNetColor(sizeInput=64,outChannels=16)
lenetModel.load_state_dict(torch.load("./resources/archive/stored/models/leNet.pth"))
lenetModel.eval()