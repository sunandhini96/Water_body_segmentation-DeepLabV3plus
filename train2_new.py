#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing all packages

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
#from model import Deeplabv3
from deeplabv3Ex import Deeplabv3
from metrics import dice_coef, iou
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

from os import listdir
from PIL import Image as PImage
import imageio


# In[3]:


physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)

""" Global parameters """
H = 100
W = 100

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# defining shuffling function

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

# defining the load Images function to load images function

def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        #img = imageio.imread(path + image)
        img = cv2.imread(path+'//'+image, cv2.IMREAD_COLOR)
        img = img.astype(np.float32)
        #x = np.expand_dims(x, axis=-1)
        loadedImages.append(img)

    return loadedImages

# defining the load masks function to load the masks

def loadMasks(path):
    # return array of images

    masksList = listdir(path)
    loadedMasks = []
    for image in masksList:
        #img = imageio.imread(path + image)
        x = cv2.imread(path+'//'+image, cv2.IMREAD_GRAYSCALE)
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=-1)
        loadedMasks.append(x)

    return loadedMasks




# defining the paths for training and validation images and masks

train_images_path = "/home/vv_sajithvariyar/sunandini_dir/oct_30/Train/image/"
val_images_path= "/home/vv_sajithvariyar/sunandini_dir/oct_30/Val/image/"
train_masks_path="/home/vv_sajithvariyar/sunandini_dir/oct_30/Train/mask/"
val_masks_path="/home/vv_sajithvariyar/sunandini_dir/oct_30/Val/mask/"



train_x = loadImages(train_images_path)
train_y=loadMasks(train_masks_path)
val_x=loadImages(val_images_path)
val_y=loadMasks(val_masks_path)



#y = loadImages(path)

#train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2,random_state=44)


# In[6]:


# In[10]:

# our images in an array
train_x=np.array(train_x)
print(train_x.shape)
train_y=np.array(train_y)
print(train_y.shape)
val_x=np.array(val_x)
print(val_x.shape)
val_y=np.array(val_y)
print(val_y.shape)



# In[6]:


# In[11]:

# load data function for loading the test data

def load_data(path):
    X = sorted(glob(os.path.join(path, "Images", "*png")))
    Y = sorted(glob(os.path.join(path, "masks", "*png")))
    return X, Y





if __name__ == "__main__":
    """ Seeding """
    np.random.seed(55)
    tf.random.set_seed(55)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 4
    lr = 0.0001
    num_epochs = 50

    """ Dataset """
 
    train_x, train_y = shuffling(train_x, train_y)
   
    train_x = np.array(train_x)/255.0
    train_y = np.array(train_y)/255.0
    val_x = np.array(val_x)/255.0
    val_y = np.array(val_y)/255.0


    

    print("train x images",len(train_x))
    print("val x  images", len(val_x))
    print("train y masks", len(train_y))
    print("val y masks", len(val_y))
    print("train x shape", np.shape(train_x))
    print("train y shape", np.shape(train_y))
    print("train x type", type(train_x))
    print("train y type", type(train_y))
    print("val x type", type(val_x))
    print("val y type", type(val_y))

    
    


# In[12]:


# In[7]:


print(num_epochs)


# In[8]:




#""" Model """

model = Deeplabv3((H, W, 3))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics=['acc', Recall(), Precision()])
checkpoint_path = "training_50/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# summary of the model

model.summary()
cp_callback = tf.keras.callbacks.ModelCheckpoint(
filepath=checkpoint_path, 
verbose=1, 
save_weights_only=True,
save_freq='epoch',
period=10)

model.save_weights(checkpoint_path.format(epoch=0))
with tf.device('/GPU:0'):
    history = model.fit(train_x,train_y, batch_size = batch_size, epochs=num_epochs,validation_data=(val_x, val_y),callbacks=[cp_callback])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training loss', 'Validation loss'], loc='upper right')
plt.show()

plt.savefig('loss_plot.png')
plt.clf()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Acc', 'Validation Acc'], loc='upper right')
plt.show() 

plt.savefig('Acc_plot.png')
   


# In[ ]:




