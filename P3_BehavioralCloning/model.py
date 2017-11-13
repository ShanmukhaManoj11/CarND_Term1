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
    # adds random brightness to the image
    img=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    img=np.array(img,dtype=np.float64)
    random_brightness=0.5+np.random.uniform()
    img[:,:,2]=img[:,:,2]*random_brightness
    img[:,:,2][img[:,:,2]>255]=255
    img=np.array(img,dtype=np.uint8)
    img=cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

def random_translation(image,angle):
    # translates the image randomly in horizontal and vertical directions
    # for every pixel translation in x direction, angle is adjusted by adding (or) subtracting 0.004 to the angle depending on direction of translation
    rows,cols,ch=image.shape
    tr_x=100*(np.random.uniform()-0.5)
    tr_y=40*(np.random.uniform()-0.5)
    tr_M=np.float32([[1,0,tr_x],[0,1,tr_y]])
    image=cv2.warpAffine(image,tr_M,(cols,rows))
    angle=angle+(tr_x*0.004)
    return image,angle

def random_shadow_addition(image):
    # adds random shadow spanning top to bottom of the image
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
    # preprocessing the image
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

def data_generator(images,steering_measurements,batch_size=128,training_data_flag=True):
    # data generator generating the batches for training
    n_samples=len(images)
    i=0
    batch_images=[]
    batch_measurements=[]
    while True:
        images,steering_measurements=shuffle(images,steering_measurements)
        for j in range(n_samples):
            batch_image=images[j]
            batch_measurement=steering_measurements[j]
            if training_data_flag:
                if abs(batch_measurement)<0.1 and np.random.uniform()<threshold:
                    continue
                else:
                    batch_image=cv2.cvtColor(batch_image,cv2.COLOR_BGR2RGB)
                    batch_image,batch_measurement=preprocess(batch_image,batch_measurement)
                    batch_images.append(batch_image)
                    batch_measurements.append(batch_measurement)
                    i=i+1
            else:
                batch_image=cv2.cvtColor(batch_image,cv2.COLOR_BGR2RGB)
                shape=batch_image.shape
                batch_image=batch_image[int(math.floor(shape[0]/3)):shape[0]-25,:,:]
                batch_image=cv2.resize(batch_image,(200,66),interpolation=cv2.INTER_AREA)
                batch_images.append(batch_image)
                batch_measurements.append(batch_measurement)
                i=i+1
            if i==batch_size:
                yield np.array(batch_images),np.array(batch_measurements)
                i=0
                batch_images=[]
                batch_measurements=[]

with open('./train_image_paths.p','rb') as f:
    train_image_paths=pickle.load(f)
train_images=[]
for image_path in train_image_paths:
    image=cv2.imread(image_path)
    train_images.append(image)
train_images=np.array(train_images)
with open('./train_steering_measurements.p','rb') as f:
    train_steering_measurements=pickle.load(f)

with open('./validation_image_paths.p','rb') as f:
    validation_image_paths=pickle.load(f)
validation_images=[]
for image_path in validation_image_paths:
    image=cv2.imread(image_path)
    validation_images.append(image)
validation_images=np.array(validation_images)
with open('./validation_steering_measurements.p','rb') as f:
    validation_steering_measurements=pickle.load(f)

print('Train: '+str(train_images.shape)+str(train_steering_measurements.shape))
print('Validation: '+str(validation_images.shape)+str(validation_steering_measurements.shape))

#############################################################################################
# initial data analysis
plt.subplots_adjust(hspace=1.0)
k=1
for i in range(5):
    train_image=cv2.cvtColor(train_images[i],cv2.COLOR_BGR2RGB)
    plt.subplot(5,2,k)
    plt.imshow(train_image)
    plt.title('steering angle: '+str(train_steering_measurements[i]))
    processed_image,processed_angle=preprocess(train_image,train_steering_measurements[i])
    plt.subplot(5,2,k+1)
    plt.imshow(processed_image)
    plt.title('steering angle: '+str(processed_angle))
    k=k+2
plt.show()
    
##batch_size=6
##threshold=0.5
##train_data_generator=data_generator(train_images,train_steering_measurements,batch_size=batch_size,training_data_flag=True)
##
##batch_images,batch_measurements=next(train_data_generator)
##for i in range(batch_size):
##    plt.subplot(2,3,i+1)
##    plt.imshow(batch_images[i])
##    plt.title('steering angle: '+str(batch_measurements[i]))
##plt.show()
#############################################################################################

##batch_size=128
##train_data_generator=data_generator(train_images,train_steering_measurements,batch_size=batch_size,training_data_flag=True)
##validation_data_generator=data_generator(validation_images,validation_steering_measurements,batch_size=batch_size,training_data_flag=False)
##
##def create_NVIDIA_architecture(load_trained_model=False,trained_model=''):
##    if load_trained_model:
##        model=load_model(trained_model)
##    else:
##        model=Sequential()
##        model.add(Lambda(lambda x:(x-127.5)/127.5,input_shape=(66,200,3)))
##        model.add(Conv2D(24,(5,5),strides=(2,2),activation='elu'))
##        model.add(Dropout(0.5))
##        model.add(Conv2D(36,(5,5),strides=(2,2),activation='elu'))
##        model.add(Dropout(0.5))
##        model.add(Conv2D(48,(5,5),strides=(2,2),activation='elu'))
##        model.add(Dropout(0.5))
##        model.add(Conv2D(64,(3,3),activation='elu'))
##        model.add(Dropout(0.5))
##        model.add(Conv2D(64,(3,3),activation='elu'))
##        model.add(Flatten())
##        model.add(Dropout(0.5))
##        model.add(Dense(100,activation='elu'))
##        model.add(Dropout(0.5))
##        model.add(Dense(50,activation='elu'))
##        model.add(Dropout(0.5))
##        model.add(Dense(10,activation='elu'))
##        model.add(Dropout(0.5))
##        model.add(Dense(1))
##        model.compile(optimizer=Adam(lr=0.0001),loss='mse')
##    return model
##
##def create_CONV_architecture(load_trained_model=False,trained_model=''):
##    if load_trained_model:
##        model=load_model(trained_model)
##    else:
##        model=Sequential()
##        model.add(Lambda(lambda x:(x-127.5)/127.5,input_shape=(64,64,3)))
##        model.add(Conv2D(3,(1,1),strides=(1,1),padding='same',activation='elu'))
##        model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='elu'))
##        model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='elu'))
##        model.add(MaxPooling2D(pool_size=(2,2)))
##        model.add(Dropout(0.5))
##        model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='elu'))
##        model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='elu'))
##        model.add(MaxPooling2D(pool_size=(2,2)))
##        model.add(Dropout(0.5))
##        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='elu'))
##        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='elu'))
##        model.add(MaxPooling2D(pool_size=(2,2)))
##        model.add(Dropout(0.5))
##        model.add(Flatten())
##        model.add(Dense(512,activation='elu'))
##        model.add(Dropout(0.5))
##        model.add(Dense(64,activation='elu'))
##        model.add(Dropout(0.5))
##        model.add(Dense(16,activation='elu'))
##        model.add(Dropout(0.5))
##        model.add(Dense(1))
##        model.compile(optimizer=Adam(lr=0.0001),loss='mse')
##    return model
##
######################################################################################
###Training NVIDIA architecture
##plt.ion()
##fig=plt.figure()
##model=create_NVIDIA_architecture()
##min_val_loss=1.0
##losses=[]
##val_losses=[]
##x=[]
##for i in range(10):
##    threshold=1/(i+1)
##    history=model.fit_generator(train_data_generator,steps_per_epoch=2000,epochs=1,verbose=1,validation_data=validation_data_generator,validation_steps=int(math.ceil(len(validation_images)/batch_size)))
##    loss=history.history['loss'][0]
##    val_loss=history.history['val_loss'][0]
##    losses.append(loss)
##    val_losses.append(val_loss)
##    x.append(i)
##    plt.plot(x,losses,'b.-',label='loss')
##    plt.plot(x,val_losses,'g.-',label='val_loss')
##    plt.draw()
##    if i==0:	
##        plt.legend()
##    plt.pause(0.05)
##    if val_loss<min_val_loss:
##        min_val_loss=val_loss
##	model.save('./model_NVIDIA.h5')
##	print('model saved')
##
##with open('./losses_NVIDIA.p','wb') as f:
##    pickle.dump(losses,f,protocol=pickle.HIGHEST_PROTOCOL)
##with open('./val_losses_NVIDIA.p','wb') as f:
##    pickle.dump(val_losses,f,protocol=pickle.HIGHEST_PROTOCOL)
##
##print(model.summary())
##while plt.fignum_exists(fig.number):
##	plt.pause(0.001)
######################################################################################
