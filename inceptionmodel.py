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
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers 

#############################################################################################




data_path_train = 'data_new/train'
data_path_test = 'data_new/test'

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator( rescale = 1.0/255. )


train_generator = train_datagen.flow_from_directory(data_path_train, batch_size = 10, class_mode = 'binary', target_size = (150, 150))
validation_generator = test_datagen.flow_from_directory(data_path_test, batch_size = 10, class_mode = 'binary', target_size = (150, 150))


base_model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False
    

x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])

inc_history = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 10, epochs = 10)