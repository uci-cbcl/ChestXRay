import numpy as np
import os
import _pickle as pickle
import torch
from torch.utils.data import Dataset
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
import math

import shutil
import sys
from torch.nn import DataParallel
import random


from guidedCNNModel import GuidedCNN


WHERE_IS_THE_CODE="home"

if WHERE_IS_THE_CODE=='home': 
	#homeServer Path
	label_dir = './'
	image_dir = '/home/deyingk/Documents/LabProjects/Xray/images/'
	weight_dir = '/home/deyingk/Documents/LabProjects/Xray/data/weights/newNewModelWeight_combing/'
elif WHERE_IS_THE_CODE=='gru':
	#gru path
	label_dir = './'
	image_dir = '/mnt/data/deyingk/xRay/data/images/'
	weight_dir = '/mnt/data/deyingk/xRay/data/newModelWeight/weight_1/'


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
def train_model(model, optimizer,num_epochs=50):
    
    nTrain = len(trainLoader.dataset)


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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

            #break

        epoch_loss = running_loss / len(data_train)

        print('{} Loss: {:.4f}'.format('train', epoch_loss))

        torch.save(model,weight_dir+'2_stages_epoch_'+str(epoch)+'.pth')




if __name__ == "__main__":


    data_train = MultiLabelDataset(label_dir+'train.csv',image_dir,trainTransform)
    #data_val = MultiLabelDataset('val.csv',image_dir,valTransform)



    trainLoader = DataLoader(
        data_train, batch_size=32, shuffle=True,num_workers=6)
    #valLoader = DataLoader(
    #    data_val, batch_size=16, shuffle=False,num_workers=6)
    dataset_train_len=len(data_train)
    #dataset_val_len=len(data_val)
    #densenet = models.densenet121(num_classes=14)
    # densenet = models.densenet121(pretrained=True)
    # densenet.classifier = nn.Linear(1024,14)



    # densenet = pickle.load(open('../../../../media/data/yangliu/xrays/our_trained_densenet_epoch_14.pkl', 'rb'))
    # densenet = densenet.cuda()
    # densenet = DataParallel(densenet)
    
    #with open(weight_dir+'densenet_epoch_15.pkl','rb') as f:
    #    densenet = pickle.load(f)
    

    stage1 = torch.load('vanillaBestDenseNet.pth')
    if type(stage1)==DataParallel:
        print("VanillaDensenet type is DataParallel")
        stage1 = stage1.module
    stage1 = nn.Sequential(*list(stage1.children())[:-1][0])

    stage2 = torch.load('vanillaBestDenseNet.pth')
    if type(stage2)==DataParallel:
        print("VanillaDensenet type is DataParallel")
        stage2 = stage2.module
    stage2 = nn.Sequential(*list(stage2.children())[:-1][0])

    guided_cnn = GuidedCNN(stage1,stage2).cuda()


    parameter=0
    for param in guided_cnn.parameters():
        parameter+=param.data.nelement()
    print ('Total trainable parameters are {}'.format(parameter))


    #print('4-------------------',attention)

    optimizer=optim.Adam(guided_cnn.parameters(),lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    model_ft = train_model(guided_cnn, optimizer, num_epochs=100)
















