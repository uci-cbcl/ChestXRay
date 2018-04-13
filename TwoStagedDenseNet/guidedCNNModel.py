import numpy as np
import os
import _pickle as pickle
import pandas as pdg
import torch
from torch.utils.data import Dataset
from skimage.color import gray2rgb
from torchvision import transforms, utils
from skimage import io, transform
from torch.utils.data import DataLoader
import torchvision.models as models
import time
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
import pandas as pd

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
from torch import nn,optim
import argparse
import math

import shutil
import sys
from torch.nn import DataParallel
import random

from collections import OrderedDict

class GuidedCNN(nn.Module):
    def __init__(self,stage1,stage2):
        # stage1 is the vanilla densenet121
        super().__init__()
        self.stage1 = stage1
        
        # build stage2
        stage2_compress_ftrs = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1024,3,1)),
            ('norm1',nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True)),
            ('relu1', nn.ReLU())
        ]))
        
        stage2_guided_input=nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(6,3,1)),
            ('norm2', nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True)),
            ('relu2', nn.ReLU())
        ]))
        
        #stage2_densenet = models.densenet121(pretrained =True)
        stage2_densenet = stage2
        #stage2_densenet.classifier = nn.Linear(1024,14)
        
        self.stage2 = nn.Sequential(OrderedDict([
            ('compress_ftrs',stage2_compress_ftrs),
            ('guided_input',stage2_guided_input),
            ('densenet',stage2_densenet)
        ]))
        self.combine = nn.Linear(2048,14)
        
    def forward(self,img):
        y = self.stage1(img)
        out1 = F.relu(y)
        out1 = F.avg_pool2d(out1, kernel_size=7, stride=1).view(out1.size(0), -1)

        y = self.stage2.compress_ftrs(y)
        y = self.upsample_ftrs_map(y)
        
        y = self.stage2.guided_input(torch.cat((img,y),dim=1))
        y = self.stage2.densenet(y)
        out2 = F.relu(y)
        out2 = F.avg_pool2d(out2, kernel_size=7, stride=1).view(out2.size(0), -1)

        #combine
        y= self.combine(torch.cat((out1,out2),dim=1))


        return y
    def upsample_ftrs_map(self,feature_map):
        bn,cn,_,_ = feature_map.shape
        enlarged_map = Variable(torch.ones(bn,cn,224,224)).cuda()
        input_enlarged_map = Variable(torch.ones(bn,cn,224,224)).cuda()
        for i in range(7):
            for j in range(7):
                input_enlarged_map[:,:,i*7:(i+1)*7,j*7:(j+1)*7]= enlarged_map[:,:,i*7:(i+1)*7,j*7:(j+1)*7]*feature_map[:,:,i:i+1,j:j+1]
        return input_enlarged_map
        
        
        