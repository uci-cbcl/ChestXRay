# ChestXRay

1. Dataset splitting.

   Image titles and labels are initially saved in the csv file, Data_Entry_2017.csv.
   
   split_step1_p3.py is used to split the data in Data_Entry_2017.csv to 3 datasets, generating train.csv (70%), val.csv (10%) and test.csv (20%).
   
   split_step2_p3.py is only used for upsampling the data in train.csv, resulting in resample_train.csv.
2. Network training, validation and test.

   Use Xray_train.py to train the dense121 network.
   
   Run validation.py and pick the best trained model according to the AUROC.
   
   Run test.py to get the AUROC on the test dataset.
   
3. Heatmap.

   cam.ipynb is the primitive code for showing the prediction result. Would be refined later.
   
4. Results

Class    |  Pathology | Standford |Our implement
------------ | ------------- |--------------|------------
0|Atelectasis |   0.8209 |0.8104 
1|Cardiomegaly |   0.9048 |**0.9085**
2|Consolidation |  0.7939 |0.7932
3|Edema|   0.8932 |**0.8894**
4|Effusion |   0.8831 |**0.8863**
5|Emphysema |   0.9260 |0.9041
6|Fibrosis |   0.8044 |**0.8288**
7|Hernia |   0.9387 |**0.9659**
8|Infiltration |   0.7204 |0.7024
9|Mass |  0.8618 |0.8611
10|Nodule |  0.7766 |0.7687
11|Pleural Thickening |   0.8138 |0.8010
12|Pneumonia |  0.7632 |0.7597
13|Pneumothorax |   0.8932 |0.8787

5. Reference

   The images and Data_Entry_2017.csv could be downloaded at https://nihcc.app.box.com/v/ChestXray-NIHCC .
   
