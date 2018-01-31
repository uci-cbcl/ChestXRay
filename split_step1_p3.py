#This code would split the data set to training set(85%), validation set(5%), test set(10%).

import random
import os
import csv

reader = csv.reader(open('Data_Entry_2017.csv'))
#headers = reader.next()
headers = next(reader)
train_path = 'train.csv'
val_path = 'val.csv'
test_path = 'test.csv'
train_path_writer = csv.writer(open(train_path, 'w'))
train_path_writer.writerow(headers)

val_path_writer = csv.writer(open(val_path, 'w'))
val_path_writer.writerow(headers)

test_path_writer = csv.writer(open(test_path, 'w'))
test_path_writer.writerow(headers)

for i, row in enumerate(reader):
	rnd_idx = random.random()

	if rnd_idx < 0.05:
		val_path_writer.writerow(row)
	elif rnd_idx>=0.05 and rnd_idx<0.15:
		test_path_writer.writerow(row)
	else:
		train_path_writer.writerow(row)
