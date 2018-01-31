# ChestXRay

1. Dataset splitting.
   Image titles and labels are initially saved in the csv file, Data_Entry_2017.csv.
   split_step1_p3.py is used to split the data in Data_Entry_2017.csv to 3 datasets, generating train.csv, val.csv and            test.csv.
   split_step2_p3.py is used for upsampling the data in train.csv, resulting in resample_train.csv.
2. Network training.
   Use Xray_train_p3.py to train the dense121 network.
3. Network prediction.
4. Reference
   The images and Data_Entry_2017.csv could be downloaded at https://nihcc.app.box.com/v/ChestXray-NIHCC .
   
