import torch
import pandas
from utils import *
from models import *

# basic cifar workflow:
#   1. for each batch, make 3 copies of every image at each rotation
#   2. train

# at test time:
#   1. plop three MLP layers on the end and train with SGD (conv layers frozen)
model = irevnet()
dataframe = pd.DataFrame(columns=['epoch','loss','top1','top5'])

def rotate(input):
    rotated_imgs = []
    angles = [90, 180, 270]
    for angle in angles:
        rotated_imgs.append(transforms.RandomRotation([angle,angle]))

    rotated_imgs = torch.tensor(rotated_imgs)
    rotation_labels = torch.LongTensor([0,1,2,3])
    return rotated_imgs, rotation_labels


def train_test(train=True):
    if train:
        model.train()
        dataloader = trainloader
    else:
        model.eval()
        dataloader = valloader

     for i, (input, target) in enumerate(dataloader):
         # rotate inputs
         input, target = rotate(input)
         input, target = input.to(device), target.to(device)

         output = model(input)
         loss = criterion(output, target)

         if train:
             # compute gradient and do SGD step
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
         else:
            append loss to error history


def main():
