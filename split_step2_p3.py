# import random
# import os
# import csv
# import pandas as pd
# import numpy as np

# def resample(df,labels):
#     target_num_samples = len(df[df['Finding Labels'].str.contains('Infiltration')])#print out the length for a class
#     num_samples = len(df[df['Finding Labels'].str.contains(labels)])
#     print (target_num_samples, num_samples)
#     if target_num_samples<num_samples:
#         return df[df['Finding Labels'].str.contains(labels)]
#     # print df['Finding Labels'].value_counts()
#     mul = target_num_samples // num_samples #multiplication
#     rem = target_num_samples % num_samples #reminder
#     ls = []

#     for i in range(mul):
#         ls.append(df[df['Finding Labels'].str.contains(labels)])

#     l = df[df['Finding Labels'].str.contains(labels)].index
#     rnd_idx = random.sample(list(l), rem)
#     ls.append(df.iloc[rnd_idx,:])

#     return pd.concat(ls)


# y_test_all = pd.read_csv('train.csv')
# ls = []
# for i in ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule','Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis','Pleural', 'Thickening', 'Hernia','No Finding']:
#     ls.append(resample(y_test_all, i))

# resampled_df = pd.concat(ls)

# resampled_df.to_csv('resample_train.csv',index=False)




'''
upsample the Hermia by a factor of 2.
upsample the Nodule by a factor of 2.
upsample the Emphysema by a factor of 2.
upsample the Mass by a factor of 1.5.
'''
import numpy as np
import random
import pandas as pd
import csv



resample_dict = {'Atelectasis':1.0, 
                'Cardiomegaly':1.0, 
                'Effusion':1.0, 
                'Infiltration':1.0, 
                'Mass':1.5, 
                'Nodule':2.0,
                'Pneumonia':1.0, 
                'Pneumothorax':1.0, 
                'Consolidation':1.0, 
                'Edema':1.0, 
                'Emphysema':2.0, 
                'Fibrosis':1.0,
                'Pleural':1.0, 
                'Thickening':1.0, 
                'Hernia':2.0,
                'No Finding':1.0}

def resample(df,label,resample_ratio):
    if resample_ratio==1:
        #copy it
        return df[df['Finding Labels'].str.contains(label)]
    elif resample_ratio<1:
        print("Some sample_ratio is less than 1, please redo it.")
    else:
        #upsampling
        ls =[]
        for i in np.arange(int(resample_ratio)):
            ls.append(df[df['Finding Labels'].str.contains(label)])

        num_samples = len(df[df['Finding Labels'].str.contains(label)])

        l = df[df['Finding Labels'].str.contains(label)].index
        rnd_idx = random.sample(list(l),int(num_samples*(resample_ratio-int(resample_ratio))))
        ls.append(df.iloc[rnd_idx,:])
        return pd.concat(ls)



train_df = pd.read_csv('train.csv')
ls=[]
for label,resample_ratio in resample_dict.items():
    print(label,resample_ratio)
    ls.append(resample(train_df, label,resample_ratio))
resampled_df = pd.concat(ls)
resampled_df.to_csv('resample_train.csv',index=False)
























