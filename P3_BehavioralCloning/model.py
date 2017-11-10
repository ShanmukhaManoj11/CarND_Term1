import cv2
import os
import csv
import numpy as np
import math
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense,Flatten,Activation,Dropout,Lambda
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras import regularizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt

def random_brightness_adjustment(image):
    img=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    img=np.array(img,dtype=np.float64)
    random_brightness=0.5+np.random.uniform()
    img[:,:,2]=img[:,:,2]*random_brightness
    img[:,:,2][img[:,:,2]>255]=255
    img=np.array(img,dtype=np.uint8)
    img=cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

def random_translation(image,angle):
    rows,cols,ch=image.shape
    tr_x=100*(np.random.uniform()-0.5)
    tr_y=40*(np.random.uniform()-0.5)
    tr_M=np.float32([[1,0,tr_x],[0,1,tr_y]])
    image=cv2.warpAffine(image,tr_M,(cols,rows))
    angle=angle+(tr_x*0.004)
    return image,angle

def random_shadow_addition(image):
    top_x=0
    bottom_x=160
    top_y=320*np.random.uniform()
    bottom_y=320*np.random.uniform()
    image_hls=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask=0*image_hls[:,:,1]
    X_m=np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m=np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bottom_y-top_y)-(bottom_x-top_x)*(Y_m-top_y)>=0)]=1
    if np.random.randint(2)==1:
        cond1=(shadow_mask==1)
        cond0=(shadow_mask==0)
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1]=image_hls[:,:,1][cond1]*0.5
        else:
            image_hls[:,:,1][cond0]=image_hls[:,:,1][cond0]*0.5    
    image=cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def preprocess(image,angle):
    shape=image.shape
    image=random_brightness_adjustment(image)
    image=random_shadow_addition(image)
    image=image[math.floor(shape[0]/3):shape[0]-25,:,:]
    image,angle=random_translation(image,angle)
    if np.random.uniform()<0.5:
        image=cv2.flip(image,1)
        angle=-1.0*angle
    image=cv2.resize(image,(200,66),interpolation=cv2.INTER_AREA) #for NVIDIA architecture
##    image=cv2.resize(image,(64,64),interpolation=cv2.INTER_AREA) #for CONV architecture   
    return image,angle

def data_generator(image_paths,steering_measurements,batch_size=128):
    n_samples=len(image_paths)
    i=0
    batch_images=[]
    batch_measurements=[]
    epoch=0
    while True:
        epoch=epoch+1
        image_paths,steering_measurements=shuffle(image_paths,steering_measurements)
        for j in range(n_samples):
            batch_image_path=image_paths[j]
            batch_measurement=steering_measurements[j]
            if abs(batch_measurement)<0.1 and np.random.uniform()<threshold:
                continue
            else:
                batch_image=cv2.imread(batch_image_path)
                batch_image=cv2.cvtColor(batch_image,cv2.COLOR_BGR2RGB)
                batch_image,batch_measurement=preprocess(batch_image,batch_measurement)
                batch_images.append(batch_image)
                batch_measurements.append(batch_measurement)
                i=i+1
            if i==batch_size:
                yield np.array(batch_images),np.array(batch_measurements)
                i=0
                batch_images=[]
                batch_measurements=[]

##def data_generator(image_paths,steering_measurements,batch_size=128):
##    n_samples=len(image_paths)
##    i=0
##    batch_images=[]
##    batch_measurements=[]
##    epoch=0
##    max_low_steering_per_batch=math.ceil(batch_size/2)
##    n_low_steering_per_batch=0
##    while True:
##        epoch=epoch+1
##        image_paths,steering_measurements=shuffle(image_paths,steering_measurements)
##        for j in range(n_samples):
##            batch_image_path=image_paths[j]
##            batch_measurement=steering_measurements[j]
##            if abs(batch_measurement)<0.1 and n_low_steering_per_batch>max_low_steering_per_batch:
##                continue
##            else:
##                if abs(batch_measurement)<0.1:
##                    n_low_steering_per_batch=n_low_steering_per_batch+1
##                batch_image=cv2.imread(batch_image_path)
##                batch_image=cv2.cvtColor(batch_image,cv2.COLOR_BGR2RGB)
##                batch_image,batch_measurement=preprocess(batch_image,batch_measurement)
##                batch_images.append(batch_image)
##                batch_measurements.append(batch_measurement)
##                i=i+1
##            if i==batch_size:
##                yield np.array(batch_images),np.array(batch_measurements)
##                n_low_steering_per_batch=0
##                i=0
##                batch_images=[]
##                batch_measurements=[]
                
with open('./image_paths.p','rb') as f:
    train_image_paths=pickle.load(f)
with open('./steering_measurements.p','rb') as f:
    train_steering_measurements=pickle.load(f)

print('Train: ',train_image_paths.shape,train_steering_measurements.shape)

#############################################################################################
##batch_size=6
##threshold=0.5
##train_data_generator=data_generator(train_image_paths,train_steering_measurements,batch_size=batch_size)
##
##batch_images,batch_measurements=next(train_data_generator)
##for i in range(batch_size):
##    plt.subplot(2,3,i+1)
##    plt.imshow(batch_images[i])
##    plt.title('steering angle: '+str(batch_measurements[i]))
##plt.show()

##count=0
##for measurement in train_steering_measurements:
##    if abs(measurement)<0.1:
##        count=count+1
##print('no. of measurements in [-0.1,0.1]: '+str(count))
##print('ratio of low measurements= '+str(count/len(train_steering_measurements)))
#############################################################################################

batch_size=128
train_data_generator=data_generator(train_image_paths,train_steering_measurements,batch_size=batch_size)

def create_NVIDIA_architecture(load_trained_model=False,trained_model=''):
    if load_trained_model:
        model=load_model(trained_model)
    else:
        model=Sequential()
        model.add(Lambda(lambda x:(x-127.5)/127.5,input_shape=(66,200,3)))
        model.add(Conv2D(24,(5,5),strides=(2,2),activation='elu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(36,(5,5),strides=(2,2),activation='elu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(48,(5,5),strides=(2,2),activation='elu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(64,(3,3),activation='elu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(64,(3,3),activation='elu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100,activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(50,activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(10,activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(optimizer=Adam(lr=0.0001),loss='mse')
    return model

def create_CONV_architecture(load_trained_model=False,trained_model=''):
    if load_trained_model:
        model=load_model(trained_model)
    else:
        model=Sequential()
        model.add(Lambda(lambda x:(x-127.5)/127.5,input_shape=(64,64,3)))
        model.add(Conv2D(3,(1,1),strides=(1,1),padding='same',activation='elu'))
        model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='elu'))
        model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='elu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='elu'))
        model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='elu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='elu'))
        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='elu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(512,activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(64,activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(16,activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(optimizer=Adam(lr=0.0001),loss='mse')
    return model

####################################################################################
#Training NVIDIA architecture
model=create_NVIDIA_architecture()
for i in range(8):
    threshold=1/(i+1)
    history=model.fit_generator(train_data_generator,steps_per_epoch=10000,\
                                epochs=1,verbose=1)
print(model.summary())
model.save('./model_NVIDIA.h5')
####################################################################################

####################################################################################
###Training CONV architecture
##model=create_CONV_architecture()
##min_loss=1.0
##for i in range(10):
##    threshold=1/(i+1)
##    history=model.fit_generator(train_data_generator,steps_per_epoch=2000,\
##                                epochs=1,verbose=1)
##    if history.history['loss'][0]<min_loss:
##        min_loss=history.history['loss'][0]
##        model.save('./model_CONV.h5')
##        print('model saved')
##print(model.summary())
####################################################################################
