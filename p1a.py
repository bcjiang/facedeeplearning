import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.Module):

    def __init__(self):
        super(SiameseNet, self).__init__()
        self.nn1 = nn.Sequential(
            nn.Conv2d(3,64,5,padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(64,128,5,padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(256,512,3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

        self.nn2 = nn.Sequential(
            nn.Linear(131072,1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
        )

    def net_forward(self,x):
        temp = self.nn1(x)
        temp = temp.view(temp.size()[0], -1)
        output = self.nn2(temp)
        return output

    def forward(self,x1,x2):
        output1 = self.net_forward(x1)
        output2 = self.net_forward(x2)
        return output1, output2