#############################################################################################
# University             : Hochschule Anhalt
# Group                  : 3    
# Authors                : Elkin Fernandez
#                          Vishnudev Kurumbaparambil
# Degree Course          : Electrical and Computer Engineering (M. Eng.)
# Subject                : Machine Learning and AI
# File Name              : Task7_Train_test_classify.py
# Date                   : 27-06-2022
# Description            : This script creates model for dishware classification
#############################################################################################


########################### I M P O R T     S E C T I O N ###################################
import tensorflow as tf
import numpy as np
import os
import shutil
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers 

#############################################################################################




data_path_train = 'data_new/train'
data_path_test = 'data_new/test'

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(data_path_train, batch_size = 20, class_mode = 'binary', target_size = (224, 224))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory( data_path_test,  batch_size = 20, class_mode = 'binary', target_size = (224, 224))

base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False
    
    
# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])


vgghist = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 10, epochs = 10)