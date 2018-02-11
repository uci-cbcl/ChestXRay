import numpy as np
import os
import _pickle as pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from PIL import Image

import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn,optim
from torch.nn import DataParallel
import math
import shutil
import sys
import random






from Xray_train import MultiLabelDataset
from Xray_train import normTransform, label_dir,image_dir,weight_dir
from Xray_train import get_loss_function
auroc_save_dir = './'

def prediction(val_loader,net):
    #net.eval()
    predict_container =  np.zeros((0, 14))
    target_container = np.zeros((0, 14))
    for i, (data, target) in enumerate(val_loader):
        data = Variable(data.float().cuda(), volatile = True)
        target = Variable(target.float().cuda(), volatile = True)
        output = net(data)
        pred_temp = 1/(1+(-output).exp())
        #preds = (pred_temp > 0.5)
        #print preds.data.cpu().numpy()
        predict_container = np.concatenate((predict_container,pred_temp.data.cpu().numpy()),axis=0)
        target_container = np.concatenate((target_container,target.data.cpu().numpy()),axis=0)
        
    return predict_container, target_container


def makeValLoader():

	valTransform = transforms.Compose([
	    transforms.RandomCrop(224),
	    transforms.ToTensor(),
	    normTransform
	])

	data_val = MultiLabelDataset(label_dir+'val.csv',image_dir,valTransform)

	valLoader = DataLoader(
	    data_val, batch_size=64, shuffle=False,num_workers=6)
	dataset_val_len=len(data_val)
	print('Valadation date set length is ',dataset_val_len)
	return valLoader

def main():
	valLoader = makeValLoader()

	auroc_dict ={}
	for epoch_i in range(100):
	    densenet = pickle.load(open(weight_dir+'densenet_epoch_'+str(epoch_i)+'.pkl', 'rb'))
	    densenet.eval()
	    y_score, y_test = prediction(valLoader,densenet)
	    # Compute ROC curve and ROC area for each class
	    fpr = dict()
	    tpr = dict()
	    roc_auc = dict()
	    for i in range(14):
	        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
	        roc_auc[i] = auc(fpr[i], tpr[i])
	    auroc_dict['epoch_'+str(epoch_i)]=roc_auc
	    print(epoch_i,auroc_dict['epoch_'+str(epoch_i)])
	    with open(auroc_save_dir+'auroc_dict.pkl','wb') as f:
	        pickle.dump(auroc_dict,f)

if __name__ =='__main__':
	main()


























