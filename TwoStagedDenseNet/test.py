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






from newNewModelTrain import MultiLabelDataset
from newNewModelTrain import normTransform, label_dir,image_dir,weight_dir
from newNewModelTrain import get_loss_function
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


def makeTestLoader():

	testTransform = transforms.Compose([
	    transforms.RandomCrop(224),
	    transforms.ToTensor(),
	    normTransform
	])

	data_test = MultiLabelDataset(label_dir+'test.csv',image_dir,testTransform)

	testLoader = DataLoader(
	    data_test, batch_size=64, shuffle=False,num_workers=6)
	dataset_test_len=len(data_test)
	print('Test date set length is ',dataset_test_len)
	return testLoader

def main():
	testLoader = makeTestLoader()

	all_test_roc_auc=[]
	for epoch in range(11,12):

		auroc_dict ={}
		guided_cnn = torch.load(weight_dir+'2_stages_epoch_'+str(epoch)+'.pth')
		guided_cnn.eval()
		y_score, y_test = prediction(testLoader,guided_cnn)
		# Compute ROC curve and ROC area for each class
		fpr = dict()
		tpr = dict()
		test_roc_auc = dict()
		for i in range(14):
		    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
		    test_roc_auc[i] = auc(fpr[i], tpr[i])

		print(epoch,"-th epoch",test_roc_auc)
		all_test_roc_auc.append(test_roc_auc)
	#with open(auroc_save_dir+'all_test_auroc.pkl','wb') as f:
	    #pickle.dump(all_test_roc_auc,f)

if __name__ =='__main__':
	BEST_EPOCH = 17
	main()


























