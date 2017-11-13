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
            steering_measurements.append(float(row[3])+0.25) # adjusting steering angle for left camera image by adding 0.25
            steering_measurements.append(float(row[3])-0.25) # adjusting steering angle for left camera image by subtracting 0.25

    return np.array(image_paths),np.array(steering_measurements)

data_path='./sample_data/'
image_paths,steering_measurements=load_data(path=data_path)
print('data size (sample data): images - ',image_paths.shape,' steering measurements - ',steering_measurements.shape)

def visualize(image_path=[],steering_measurements=[]):
    for i in range(0,len(image_paths),3):
        image=cv2.imread(image_paths[i])
        shape=image.shape
        image=image[math.ceil(shape[0]/3):shape[0]-25,:,:]
        new_y=shape[0]-25-math.ceil(shape[0]/3)
        angle=steering_measurements[i]
        cv2.line(image,(shape[1]/2,new_y),(int(shape[1]/2+20*math.sin(angle)),int(new_y-20*math.cos(angle))),(0,255,0),3)
        cv2.putText(image,str(angle),(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),2,cv2.LINE_AA)
	cv2.imshow('',image)
        cv2.waitKey(10)

##Uncomment to visualize data
#visualize(image_paths,steering_measurements)

image_paths,steering_measurements=shuffle(image_paths,steering_measurements)

train_image_paths,validation_image_paths,train_steering_measurements,validation_steering_measurements=train_test_split(image_paths,steering_measurements,test_size=0.15)

print('data size (train): images - '+str(train_image_paths.shape)+' steering measurements - '+str(train_steering_measurements.shape))
print('data size (validation): images - '+str(validation_image_paths.shape)+' steering measurements - '+str(validation_steering_measurements.shape))

with open('./train_image_paths.p','wb') as f:
    pickle.dump(train_image_paths,f,protocol=pickle.HIGHEST_PROTOCOL)
with open('./train_steering_measurements.p','wb') as f:
    pickle.dump(train_steering_measurements,f,protocol=pickle.HIGHEST_PROTOCOL)

with open('./validation_image_paths.p','wb') as f:
    pickle.dump(validation_image_paths,f,protocol=pickle.HIGHEST_PROTOCOL)
with open('./validation_steering_measurements.p','wb') as f:
    pickle.dump(validation_steering_measurements,f,protocol=pickle.HIGHEST_PROTOCOL)

