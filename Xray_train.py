#This script is in python of version 3.
#This script uses the original loss function.
#see readme



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


##### Set path
#image_dir = '/media/hdd10tb/deyingk/xRay/images/'  # This is on CNN
#weight_dir ='/media/hdd10tb/deyingk/xRay/weights/split2/upsample2_origLossFun_Xtran5_Rotate5/' #This is on CNN 

#image_dir = '/home/deyingk/Documents/LabProjects/Xray/data/images/'  # This is on home PC
#weight_dir ='/home/deyingk/Documents/LabProjects/Xray/data/weights/weights3/' #This is on home PC 


label_dir = './'
image_dir = '/home/deyingk/Documents/LabProjects/Xray/images/'
weight_dir = '/home/deyingk/Documents/LabProjects/Xray/data/weights/weights_ResizeAndCrop/'




class MultiLabelDataset(Dataset):
    def __init__(self, csv_path, img_path, transform=None):
        tmp_df = pd.read_csv(csv_path)
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.transform = transform

        self.X = tmp_df['Image Index']

         
        a = self.mlb.fit_transform(tmp_df['Finding Labels'].str.split('|')).astype(np.float32)
        self.y =a
        NoFindingIndex=list(self.mlb.classes_).index('No Finding')
        self.y =np.delete(a,NoFindingIndex,1)         #delete the classification for "No Finding"
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.X[index]))
        img = img.resize((256,256))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.y[index])

        #print ('line 79',img.size())
        return img, label


    def __len__(self):
        return len(self.X.index)

normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
normTransform = transforms.Normalize(normMean, normStd)

class XTranslation(object):
    # do translation on the Tensor
    # if padding=None, fill the blank of the translated image with 0s
    # if padding = "cyclic", the blank is filled by the part of image translated out
    def __init__(self,tran_max_size,padding=None):
        self.tran_max_size = tran_max_size
        self.padding =padding
    def __call__(self,img):
        tran_size = random.randint(- self.tran_max_size,self.tran_max_size)

        new_img = torch.Tensor(*img.size()).zero_()
        if self.padding ==None:
            if tran_size>0:
                #translate to the right
                new_img[:,:,tran_size:] = img[:,:,:-tran_size]
            elif tran_size<0:
                #translate to the left
                new_img[:,:,:tran_size] = img[:,:,-tran_size:]
            else:
                new_img = img
        elif self.padding =='cyclic':
            if tran_size>0:
                #translate to the right
                new_img[:,:,tran_size:] = img[:,:,:-tran_size]
                new_img[:,:,:tran_size] = img[:,:,-tran_size:]
            elif tran_size<0:
                #translate to the left
                new_img[:,:,:tran_size] = img[:,:,-tran_size:]
                new_img[:,:,tran_size:] = img[:,:,:-tran_size]

            else:
                new_img = img
        else:
            raise Exception('Wrong using Translation!')        

        return new_img








trainTransform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(5),
    transforms.ToTensor(),
    #XTranslation(5,'cyclic'),
    normTransform,
])

valTransform = transforms.Compose([
    #transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    normTransform
])







#trainF#

##### The following block sets up the weights in the loss function,the weights inside each patho class
#frq =torch.FloatTensor([16057.,3906,7177,3433,18974,3586,2211,284,25366,8269,8409,5172,2092,7134])/112120
#frqMat = Variable(torch.diag(frq).cuda(),requires_grad=False)
#ratio = (1-frq)/frq
#coe1 =ratio/(1+ratio)
#coe2 =1-coe1
#coe1_Mat = Variable(torch.diag(coe1).cuda(),requires_grad=False)
#coe2_Mat = Variable(torch.diag(coe2).cuda(),requires_grad=False)

def get_loss_function(output, target):
    possiblility_vec = 1/(1+(-output).exp())
    #return np.sum(-target*np.log(possiblility_vec)-(1-target)*np.log(1-possiblility_vec))
    # loss = -target*possiblility_vec.log()-(1-target)*(1-possiblility_vec).log()
    loss = -target*(possiblility_vec+1e-10).log()-(1-target)*(1-possiblility_vec+1e-10).log()
    
    #The following line is for the weighted loss func.
    #loss = (-target*torch.mm((possiblility_vec+1e-10).log(),coe1_Mat) -(1-target)*torch.mm((1-possiblility_vec+1e-10).log(),coe2_Mat))*2
    # we multiply the loss by a factor of 2, just to make it comparable to previous version of loss fun. Since in prev version, either weight is 1, summing up to 2.
    return loss.mean()

#-------------Training------------------#
def train_model(model, optimizer, num_epochs=50):
    since = time.time()
    
    # nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for epoch in range(16,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(trainLoader):
            data, target = Variable(data.cuda()),Variable(target.cuda())
#             print target
            optimizer.zero_grad()
            output = model(data)
            loss = get_loss_function(output, target)
            loss.backward()
            optimizer.step()
            # nProcessed += len(data)
            running_loss += loss.data[0]

        epoch_loss = running_loss / len(data_train)

        print('{} Loss: {:.4f}'.format('train', epoch_loss))

            # deep copy the model
        with open(weight_dir+'densenet_epoch_'+str(epoch)+'.pkl','wb') as f:
        	pickle.dump(model,f)

if __name__ == "__main__":

    data_train = MultiLabelDataset(label_dir+'train.csv',image_dir,trainTransform)
    #data_val = MultiLabelDataset('val.csv',image_dir,valTransform)



    trainLoader = DataLoader(
        data_train, batch_size=64, shuffle=True,num_workers=6)
    #valLoader = DataLoader(
    #    data_val, batch_size=16, shuffle=False,num_workers=6)
    dataset_train_len=len(data_train)
    #dataset_val_len=len(data_val)
    #densenet = models.densenet121(num_classes=14)
    densenet = models.densenet121(pretrained=True)
    densenet.classifier = nn.Linear(1024,14)



    # densenet = pickle.load(open('../../../../media/data/yangliu/xrays/our_trained_densenet_epoch_14.pkl', 'rb'))
    densenet = densenet.cuda()
    densenet = DataParallel(densenet)
    
    with open(weight_dir+'densenet_epoch_15.pkl','rb') as f:
        densenet = pickle.load(f)
    
    parameter=0
    for param in densenet.parameters():
        parameter+=param.data.nelement()
    print ('Total trainable parameters are {}'.format(parameter))

    optimizer=optim.Adam(densenet.parameters(),lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    model_ft = train_model(densenet, optimizer,num_epochs=100)

