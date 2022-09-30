#############################################################################################
# University             : Hochschule Anhalt
# Group                  : 3    
# Authors                : Elkin Fernandez
#                          Vishnudev Kurumbaparambil
# Degree Course          : Electrical and Computer Engineering (M. Eng.)
# Subject                : Machine Learning and AI
# File Name              : Task8_1_Kitchenware_Classifier_Model.py
# Date                   : 30-09-2022
# Description            : This script creates model for kitchenware classification
#############################################################################################


#############################    I M P O R T    S E C T I O N   ######################################
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

#############################    M A I N    S E C T I O N   ######################################

####
# Global values
####

## Desired image resolution
img_height = 256
img_width = 256
batch_size = 32
epochs_train = 15

dataset_foldername = 'data'

"""From Google Drive"""

## Directory to dataset in drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

cd_path = '/content/gdrive/MyDrive/MachineLearning/Kitchenware_Classifier/'

## Train and test dataset paths
data_path_train = cd_path + '/' + dataset_foldername + '/'  + 'train'
data_path_test = cd_path + '/' +  dataset_foldername + '/'  + 'test'

####
# Prepare training and testing data
####
ds_train = tf.keras.utils.image_dataset_from_directory(
    data_path_train,
    color_mode = 'rgb',
    image_size=(img_height,img_width), # reshape
    shuffle = True,
    batch_size = batch_size,
    validation_split=0.2,
    subset="training",
    seed=100
)

ds_val = tf.keras.utils.image_dataset_from_directory(
    data_path_train,
    color_mode = 'rgb',
    image_size=(img_height,img_width), # reshape
    shuffle = True,
    batch_size = batch_size,
    validation_split=0.2,
    subset="validation",
    seed=100
)
  
ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    data_path_test,
    color_mode = 'rgb',
    image_size=(img_height,img_width), # reshape
    shuffle = True,
    batch_size = batch_size,
    seed=123
)

## Read the class list from train or test set
ds_class_names = ds_train.class_names


####
#  Building CNN model
####
input_shape = (img_height, img_width, 3)

## Preprocessing layer
resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.Resizing(img_height, img_width),
  tf.keras.layers.Rescaling(1./255, input_shape=input_shape)
])

## Create the architecture of the model
model = tf.keras.Sequential([
    resize_and_rescale,
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1500, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])              
              
              
## Using GPU provided by Google for faster training
device_name = tf.test.gpu_device_name()
if len(device_name) > 0:
    print("Found GPU at: {}".format(device_name))
else:
    device_name = "/device:CPU:0"
    print("No GPU, using {}.".format(device_name))

with tf.device(device_name):

    ## Training...
    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs_train)

    
## Evluate the accuracy using test set
test_loss, test_acc = model.evaluate(ds_test, verbose=2)
print('\nThe test accuracy : ', test_acc)

## Plot the accuracy and loss information during the training
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs_train)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

## Summary of the model
model.summary()
## Save the model for future use
model.save(cd_path + 'TrainedModel/kitchenware_trainedmodel.h5')