import torch
import os
import cv2
import argparse
import torch.nn as nn
import random
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
import pickle
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Parsing arguments
parser = argparse.ArgumentParser(description='Face recognition using BCELoss.')
parser.add_argument('--load', type = str, help = 'Using trained parameter to test on both train and test sets.')
parser.add_argument('--save', type = str, help = 'Train the model using splitting file provided.')
args = parser.parse_args()

# Setting up configuration
configs = {"batch_train": 16, \
            "batch_test": 4, \
            "epochs": 40, \
            "num_workers": 4, \
            "learning_rate": 1e-6, \
            "data_augment": True, \
            "loss_margin": 1.0, \
            "decision_thresh": 1.0}

# Define dataset class
class FaceDateSet(Dataset):
    """lfw face data set."""

    def __init__(self, root_dir, split_file, transform = None, augment = False):
        self.root_dir = root_dir
        self.split_file = split_file
        self.transform = transform
        self.img_paths = self.parse_files()
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Get items from path here
        img1_path = os.path.join(self.root_dir, self.img_paths[idx][0])
        img2_path = os.path.join(self.root_dir, self.img_paths[idx][1])
        img_label = map(float,self.img_paths[idx][2])
        img_label = torch.from_numpy(np.array(img_label)).float()
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')

        isaugment = (random.random() <= 0.7)
        if self.augment == True and isaugment == True:
            isflip = (random.random() <= 0.7)
            if isflip == True:
                img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            isscale = (random.random() <= 0.7)
            if isscale == True:
                ratio = 0.6*random.random() + 0.7
                if ratio > 1:
                    old_size = img1.size
                    new_size = tuple([int(i*ratio) for i in img1.size])
                    img1 = img1.resize(new_size, Image.ANTIALIAS)
                    left = abs((old_size[0] - new_size[0])/2)
                    top = abs((old_size[1] - new_size[1])/2)
                    right = abs((old_size[0] + new_size[0])/2)
                    bottom = abs((old_size[1] + new_size[1])/2)
                    img1 = img1.crop((left,top,right,bottom))
                else:
                    old_size = img1.size
                    new_size = tuple([int(i*ratio) for i in img1.size])
                    img1 = img1.resize(new_size, Image.ANTIALIAS)
                    left = 0
                    top = 0
                    right = old_size[0]
                    bottom = old_size[1]
                    img1 = img1.crop((left,top,right,bottom))
            istrans = (random.random() <= 0.7)
            if istrans == True:
                translate_x = int(10 - random.random()*20)
                translate_y = int(10 - random.random()*20)
                img1 = img1.transform(img1.size, Image.AFFINE, (1,0,translate_x,0,1,translate_y))
            isrotate = (random.random() <= 0.7)
            if isrotate == True:
                angle = 30 - 60*random.random()
                img1.rotate(angle)

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
            nn.Linear(1024,5)
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

class ContrastiveLoss(nn.Module):

    def __init__(self, margin = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        d = F.pairwise_distance(output1, output2)
        loss = torch.mean((label) * torch.pow(d,2) +(1-label) * torch.pow(torch.clamp(self.margin - d, min=0.0), 2))

# Switch to training
if args.save != None: 
    weights_dir = args.save

    # Training process setup
    data_trans = transforms.Compose([transforms.Scale((128,128)),transforms.ToTensor()])
    face_train = FaceDateSet(root_dir='lfw', split_file='train.txt', transform=data_trans, augment=configs['data_augment'])
    train_loader = DataLoader(face_train, batch_size=configs['batch_train'], shuffle=True, num_workers=configs['num_workers'])

    # Training the net
    net = SiameseNet().cuda()
    optimizer = optim.Adam(net.parameters(), lr = configs['learning_rate'])
    loss_fn = ContrastiveLoss(margin = configs['loss_margin'])
    total_epoch = configs['epochs']
    counter = []
    loss_history = []
    iteration = 0 
    for epoch in range(total_epoch):
        for batch_idx, batch_sample in enumerate(train_loader):
            img1 = batch_sample['img1']
            img2 = batch_sample['img2']
            label = batch_sample['label']
            img1, img2, y = Variable(img1).cuda(), Variable(img2).cuda(), Variable(label).cuda()
            optimizer.zero_grad()
            # y_pred = net(img1, img2)
            feature1, feature2 = net(img1, img2)
            contr_loss = loss_fn(feature1, feature2, y)
            contr_loss.backward()
            optimizer.step()

            if batch_idx % (len(face_train)/configs['batch_train']/5) == 0:
                print "Epoch %d, Batch %d Loss %f" % (epoch, batch_idx, contr_loss.data[0])
                iteration += 20
                counter.append(iteration)
                loss_history.append(contr_loss.data[0])
        
    # Save the trained network
    torch.save(net.state_dict(), weights_dir)
    total_hist = [counter, loss_history]
    with open("training_history.txt", "wb") as fp:
        pickle.dump(total_hist, fp)


# Switching to testing
elif args.load != None: 
    if os.path.isfile(args.load):
        weights_dir = args.load
        net = SiameseNet().cuda()
        net.load_state_dict(torch.load(weights_dir))
        net.eval()
        loss_fn = nn.BCELoss()

        # Testing on the training data
        data_trans1 = transforms.Compose([transforms.Scale((128,128)),transforms.ToTensor()])
        face_test1 = FaceDateSet(root_dir='lfw', split_file='train.txt', transform = data_trans1)
        test1_loader = DataLoader(face_test1, batch_size=configs['batch_test'], shuffle=False)
        total_loss = 0.0
        total_correct = 0

        for batch_idx, batch_sample in enumerate(test1_loader):
            img1 = batch_sample['img1']
            img2 = batch_sample['img2']
            label = batch_sample['label']
            label = label.view(label.numel(),-1)
            img1, img2, y = Variable(img1, volatile=True).cuda(), \
                            Variable(img2, volatile=True).cuda(), \
                            Variable(label,volatile=True).cuda()
            feature1, feature2 = net(img1, img2)
            distance = F.pairwise_distance(feature1, feature2)
            y_pred_round = int(distance < configs['decision_thresh'])
            if batch_idx % int(len(face_test1)/configs['batch_test']/5) == 0:
                print "Batch %d feature distance %f" % (batch_idx, distance)
            total_correct += (y_pred_round.view(-1) == y.view(-1)).sum().float()

        mean_correct = total_correct / float(len(face_test1))
        print "Prediction accuracy on training set is: ", mean_correct

        # Testing on the testing data
        data_trans2 = transforms.Compose([transforms.Scale((128,128)),transforms.ToTensor()])
        face_test2 = FaceDateSet(root_dir='lfw', split_file='test.txt', transform = data_trans2)
        test2_loader = DataLoader(face_test2, batch_size=configs['batch_test'], shuffle=False)
        total_loss = 0.0
        total_correct = 0

        for batch_idx, batch_sample in enumerate(test2_loader):
            img1 = batch_sample['img1']
            img2 = batch_sample['img2']
            label = batch_sample['label']
            label = label.view(label.numel(),-1)
            img1, img2, y = Variable(img1, volatile=True).cuda(), \
                            Variable(img2, volatile=True).cuda(), \
                            Variable(label,volatile=True).cuda()
            feature1, feature2 = net(img1, img2)
            distance = F.pairwise_distance(feature1, feature2)
            y_pred_round = int(distance < configs['decision_thresh'])
            if batch_idx % int(len(face_test2)/configs['batch_test']/5) == 0:
                print "Batch %d feature distance %f" % (batch_idx, distance)
            total_correct += (y_pred_round.view(-1) == y.view(-1)).sum().float()

        mean_correct = total_correct / float(len(face_test2))
        print "Prediction accuracy on test set is: ", mean_correct

    else:
        print "Parameter file does not exist!"

else:
    print "Please use [-h] for help on the usage."
