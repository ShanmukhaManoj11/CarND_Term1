import cv2
import os
import csv
import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

with open('./image_paths.p','rb') as f:
    image_paths=pickle.load(f)
with open('./steering_measurements.p','rb') as f:
    steering_measurements=pickle.load(f)

########################################################################
# Initial analysis on data
print('image_paths shape: ',image_paths.shape)
print('steering_measurements shape: ',steering_measurements.shape)

angle_max=np.max(steering_measurements)
angle_min=np.min(steering_measurements)
print('maximum steering angle recorded: ',angle_max)
print('minimum steering angle recorded: ',angle_min)

n_bins=25
average_measurements_per_bin=math.ceil(len(steering_measurements)/n_bins)
print('average measurements per bin: ',average_measurements_per_bin)
y,bins=np.histogram(steering_measurements,n_bins)
width=0.75*(bins[1]-bins[0])/2.0
x=(bins[:-1]+bins[1:])/2.0
plt.bar(x,y,align='center',width=width)
plt.plot((angle_min,angle_max),(average_measurements_per_bin,average_measurements_per_bin),'-g')
plt.xticks(x)
plt.show()
########################################################################

ids_in_bin={}
for i in range(n_bins):
    ids_in_bin[i]=[]
    for j in range(len(steering_measurements)):
        if steering_measurements[j]>=bins[i] and steering_measurements[j]<bins[i+1]:
            ids_in_bin[i].append(j)

remove_list=np.array([])
for i in range(n_bins):
    count=len(ids_in_bin[i])
    if count>average_measurements_per_bin:
        remove_list=np.append(remove_list,np.random.permutation(ids_in_bin[i])[:count-average_measurements_per_bin])
print('removelist shape: ',remove_list.shape)
image_paths=np.delete(image_paths,remove_list)
steering_measurements=np.delete(steering_measurements,remove_list)

##ids_in_bin={}
##for i in range(n_bins):
##    ids_in_bin[i]=[]
##    for j in range(len(steering_measurements)):
##        if steering_measurements[j]>=bins[i] and steering_measurements[j]<bins[i+1]:
##            ids_in_bin[i].append(j)
##
##add_list=np.array([])
##for i in range(n_bins):
##    count=len(ids_in_bin[i])
##    if count<average_measurements_per_bin and count>0:
##        add_list=np.append(add_list,np.random.choice(ids_in_bin[i],average_measurements_per_bin-count))
##print('add list shape: ',add_list.shape)
##image_paths=np.append(image_paths,image_paths[np.array(add_list,dtype=np.int32)])
##steering_measurements=np.append(steering_measurements,steering_measurements[np.array(add_list,dtype=np.int32)])

y,bins=np.histogram(steering_measurements,n_bins)
width=0.75*(bins[1]-bins[0])/2.0
x=(bins[:-1]+bins[1:])/2.0
plt.bar(x,y,align='center',width=width)
plt.plot((angle_min,angle_max),(average_measurements_per_bin,average_measurements_per_bin),'-g')
plt.xticks(x)
plt.show()

print('image_paths shape: ',image_paths.shape)
print('steering_measurements shape: ',steering_measurements.shape)

train_image_paths,validation_image_paths,train_steering_measurements,validation_steering_measurements=train_test_split(image_paths,steering_measurements,test_size=0.15,random_state=111)

with open('./train_image_paths_adjusted.p','wb') as f:
    pickle.dump(train_image_paths,f,protocol=pickle.HIGHEST_PROTOCOL)
with open('./train_steering_measurements_adjusted.p','wb') as f:
    pickle.dump(train_steering_measurements,f,protocol=pickle.HIGHEST_PROTOCOL)

with open('./validation_image_paths_adjusted.p','wb') as f:
    pickle.dump(validation_image_paths,f,protocol=pickle.HIGHEST_PROTOCOL)
with open('./validation_steering_measurements_adjusted.p','wb') as f:
    pickle.dump(validation_steering_measurements,f,protocol=pickle.HIGHEST_PROTOCOL)
