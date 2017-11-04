import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

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

        self.nn3 = nn.Sequential(
            nn.Linear(2048, 1)
            nn.Sigmoid()
        )

    def net_forward(self,x):
        temp = self.nn1(x)
        temp = temp.view(temp.size()[0], -1)
        output = self.nn2(temp)
        return output

    def forward(self,x1,x2):
        output1 = self.net_forward(x1)
        output2 = self.net_forward(x2)
        output12 = torch.cat((output1,output2))
        output = self.nn3(output12)
        return output

# Define BCE loss
class BCEloss(nn.Module):
    def __init__(self):
        super(BCEloss, self).__init__()

    def forward(self,output,label):
        loss = F.binary_cross_entropy_with_logits(output,label)
        return loss

# Training the net
net = SiameseNet()
optimizer = optim.Adam(net.parameters(), lr = 0.0005)
loss_function = BCEloss()
total_epoch_num = 100
for epoch in range(total_epoch_num):
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        x = Variable(imgs)
        y = Variable(labels)
        optimizer.zero_grad()
        y_pred = net(img_input1,img_input2)
        bce_loss = loss_function(y_pred,y)
        bce_loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print "Epoch %d, Batch %d Loss %f" % (epoch, batch_idx, bce_loss.data[0])
    
# Test the net