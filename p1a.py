import torch
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt 
import numpy as np
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader


# Define dataset class
class FaceDateSet(Dataset):
    """lfw face data set."""

    def __init__(self, root_dir, split_file, transform = None):
        self.root_dir = root_dir
        self.split_file = split_file
        self.transform = transform
        self.img_paths = self.parse_files()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Get items from path here
        img1_path = os.path.join(self.root_dir, self.img_paths[idx][0])
        img2_path = os.path.join(self.root_dir, self.img_paths[idx][1])
        img_label = float(self.img_paths[idx][2])
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        sample = {'img1': img1, 'img2': img2, 'label': img_label}
        return sample

    def parse_files(self):
        img_paths = []
        with open(self.split_file) as f:
            img_paths = f.readlines()
        img_paths = [x.split() for x in img_paths]
        return img_paths

# Define deep neural network
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
            nn.Linear(2048, 1),
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
        output12 = torch.cat((output1,output2),1)
        output = self.nn3(output12)
        return output

# Training process setup
data_trans = transforms.Compose([transforms.ToPILImage(),transforms.Scale((128,128)),transforms.ToTensor()])
face_train = FaceDateSet(root_dir='lfw', split_file='train.txt', transform = data_trans)
train_loader = DataLoader(face_train, batch_size=4, shuffle=True, num_workers=4)

# Training the net
net = SiameseNet().cuda()
optimizer = optim.Adam(net.parameters(), lr = 1e-6)
loss_fn = nn.BCELoss()
total_epoch = 1000
for epoch in range(total_epoch):
    for batch_idx, batch_sample in enumerate(train_loader):
        img1 = batch_sample['img1']
        img2 = batch_sample['img2']
        label = batch_sample['label'].float()
        label = label.view(label.numel(),-1)
        img1, img2, y = Variable(img1).cuda(), Variable(img2).cuda(), Variable(label).cuda()
        optimizer.zero_grad()
        y_pred = net(img1, img2)
        bce_loss = loss_fn(y_pred, y)
        bce_loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print "Epoch %d, Batch %d Loss %f" % (epoch, batch_idx, bce_loss.data[0])
    
# Save the trained network
model.save_state_dict(net.state_dict(),'./model_best.pth.tar')

