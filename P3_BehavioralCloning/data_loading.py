import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import csv
from sklearn.utils import shuffle

def load_data(path='./sample_data/'):
    image_paths=[]
    steering_measurements=[]

    with open(path+'driving_log.csv','rt') as csvfile:
        reader=csv.reader(csvfile)
        next(reader)
        for row in reader:
            for col in range(3):
                img_path=path+'IMG/'+row[col].strip().replace('\\','/').split('/')[-1]
                image_paths.append(img_path)
            steering_measurements.append(float(row[3]))
            steering_measurements.append(float(row[3])+0.25)
            steering_measurements.append(float(row[3])-0.25)

    return np.array(image_paths),np.array(steering_measurements)

data_path='./sample_data/'
image_paths1,steering_measurements1=load_data(path=data_path)
print('data size (sample data): images - ',image_paths1.shape,' steering measurements - ',steering_measurements1.shape)
##data_path='./collected_data/'
##image_paths2,steering_measurements2=load_data(path=data_path)
##print('data size (collected data): images - ',image_paths2.shape,' steering measurements - ',steering_measurements2.shape)
image_paths2=[]
steering_measurements2=[]

image_paths=np.concatenate((image_paths1,image_paths2))
steering_measurements=np.concatenate((steering_measurements1,steering_measurements2))
print('data size (sample+collected): images - ',image_paths.shape,' steering measurements - ',steering_measurements.shape)
image_paths,steering_measurements=shuffle(image_paths,steering_measurements)

with open('./image_paths.p','wb') as f:
    pickle.dump(image_paths,f,protocol=pickle.HIGHEST_PROTOCOL)
with open('./steering_measurements.p','wb') as f:
    pickle.dump(steering_measurements,f,protocol=pickle.HIGHEST_PROTOCOL)
