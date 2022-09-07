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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
#############################################################################################


######################### F U N C T I O N     S E C T I O N #################################

#############################################################################################
# Name          : Plot_PredImg
# Description   : This API plots the images predicted by the Keras APIs
# Input         : Predicted values, image, expected output
# Output        : None
#############################################################################################

def Plot_PredImg( predictions_array, img, expected_label):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  #plt.imshow(img, cmap=plt.cm.binary)
  plt.imshow(img)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == expected_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(ds_class_names[predicted_label],
                                100*np.max(predictions_array),
                                ds_class_names[expected_label]),
                                color=color)


#############################################################################################
# Name          : Plot_PredVal
# Description   : This API plots the predicted values by the Keras APIs
# Input         : Predicted values, expected output
# Output        : None
#############################################################################################

def Plot_PredVal(predictions_array, expected_label):
  plt.grid(False)
  plt.xticks(range(num_classes))
  plt.yticks([])
  thisplot = plt.bar(range(num_classes), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[expected_label].set_color('blue')

 
#############################################################################################
# Name        : Store_NewImagesInDir
# Description : To store the newly generated images
# Input       : Image, directory path, file name
# Output      : None
#############################################################################################
def Store_NewImagesInDir(image_temp, store_dir_path, file_name):
    
    # Create directory if not existing
    if(os.path.exists(store_dir_path) == False):
        os.makedirs(store_dir_path)
    
    file_path= store_dir_path + '/' + file_name
    print(file_path)
    cv2.imwrite(file_path,image_temp)
    
    
#############################################################################################
# Name        : CannyEdge_Auto
# Description : To store the newly generated images
# Input       : Image, directory path, file name
# Output      : None
#############################################################################################
def CannyEdge_Auto(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
    
    
#end of function

############################# M A I N    S E C T I O N ######################################

## Global values
# Number of classes
num_classes = 3
datasets_types = ['train','test']
data_classes = ['cups','dishes','plates']
# Desired image resolution
img_height = 256
img_width = 256
batch_size = 1
epochs_train = 15
# Path of train and test sets
dataset_subfolder =  'edgedetected'

cd_path = os.path.dirname(os.path.abspath(__file__)) #Current path
dataset_folders = ['data','data_grp']
datasetNew_foldername = 'data_new'

'''

# Delete folder
if(os.path.exists(cd_path + '/' + datasetNew_foldername) == True):
    shutil.rmtree(cd_path + '/' + datasetNew_foldername)
        
###
# Create new dataset after preprocessing
###

# Store existing images

## Loop between training and testing set
for dataset_type in datasets_types:        
    ## Loop between different classes
    for data_class in data_classes:    
        for dataset_foldername in dataset_folders:    
            # Iterate directory
            current_dataset_path = dataset_foldername + '/' + dataset_type + '/' + data_class
            for path in os.listdir(cd_path + '/' + current_dataset_path):
                # Parse the current image after resizing
                current_image_path = current_dataset_path + '/' + path
                img = cv2.imread(current_image_path,cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (img_width,img_height))
                Store_NewImagesInDir(img, cd_path + '/' + datasetNew_foldername  + '/' + dataset_type + '/' + data_class, dataset_foldername + '_' + path)
           

      
# Do rotational 

#####
######## ROTATION ###########
#####
imgProcTechnq_name = 'rotation'

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=15,
        height_shift_range=15,
        zoom_range=[0.5,1.1],
        horizontal_flip=True,
        brightness_range=[0.3,1.3],
        fill_mode='nearest')
        
## Loop between training and testing set
for dataset_type in datasets_types:        
    ## Loop between different classes
    for data_class in data_classes:    
        for dataset_foldername in dataset_folders:
            # Iterate directory
            current_dataset_path = dataset_foldername + '/' + dataset_type + '/' + data_class
            for path in os.listdir(cd_path + '/' + current_dataset_path):
                # Parse the current image after resizing
                current_image_path = current_dataset_path + '/' + path
                save_path=cd_path + '/' + datasetNew_foldername  + '/' + dataset_type + '/' + data_class
                # img = cv2.imread(current_image_path,cv2.IMREAD_UNCHANGED)
                # img = cv2.resize(img, (img_width,img_height))
                img = tf.keras.preprocessing.image.load_img(current_image_path)  # this is a PIL image
                x = tf.keras.preprocessing.image.img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
                x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)


                file_path= save_path + '/' + dataset_foldername
                print(file_path)
    
                # the .flow() command below generates batches of randomly transformed images
                # and saves the results to the `preview/` directory
                i = 0
                for batch in datagen.flow(x, batch_size=1,
                                          save_to_dir=save_path, save_prefix=dataset_foldername, save_format='jpeg'):
                    i += 1
                    if i > 15:
                        break  # otherwise the generator would loop indefinitely
                
                
                

# Feed that to edge detection

#####
######## EDGE DETECTION ###########
#####
imgProcTechnq_name = 'edgedetected'

## Loop between training and testing set
for dataset_type in datasets_types:        
    ## Loop between different classes
    for data_class in data_classes:    
            # Iterate directory
            current_dataset_path = datasetNew_foldername + '/' + dataset_type + '/' + data_class
            for path in os.listdir(cd_path + '/' + current_dataset_path):
                # Parse the current image after resizing
                current_image_path = current_dataset_path + '/' + path
                img = cv2.imread(current_image_path,cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (img_width,img_height))
     
                # The initial processing of the image
                image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)             
                
                blur = cv2.GaussianBlur(image_bw,(3,3),0)
                
                # Threshold the images
                # ret, thresh3 = cv2.threshold(blur,220,255,cv2.THRESH_TRUNC)
                
                # Edge Detection of the image
                # edges_normal = cv2.Canny(thresh3,25,120)
                edges_normal = CannyEdge_Auto(blur)

                Store_NewImagesInDir(edges_normal, cd_path + '/' + datasetNew_foldername + '/' + imgProcTechnq_name + '/' + dataset_type + '/' + data_class, 'claheEdge_' + path)

'''

# Use that as the dataset


data_path_train = datasetNew_foldername + '/' + dataset_subfolder + '/train'
data_path_test = datasetNew_foldername + '/' + dataset_subfolder + '/test'

####
# Prepare training and testing data
####
ds_train = tf.keras.utils.image_dataset_from_directory(
    data_path_train,
    color_mode = 'grayscale',
    image_size=(img_height,img_width), # reshape
    shuffle = True,
    batch_size = batch_size,
    validation_split=0.2,
    subset="training",
    seed=100
)

ds_val = tf.keras.utils.image_dataset_from_directory(
    data_path_train,
    color_mode = 'grayscale',
    image_size=(img_height,img_width), # reshape
    shuffle = True,
    batch_size = batch_size,
    validation_split=0.2,
    subset="validation",
    seed=100
)
  
ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    data_path_test,
    color_mode = 'grayscale',
    image_size=(img_height,img_width), # reshape
    shuffle = False,
    batch_size = batch_size,
    seed=123
)


test_data = np.zeros((len(ds_test.file_paths), img_height, img_width))
test_label = np.zeros(len(ds_test.file_paths))
i = 0
for images, labels in ds_test.take(-1):
        temp = images.numpy().astype("uint8")
        temp = np.squeeze(temp)
        temp = temp / 255.0
        test_data[i] = temp
        test_label[i] = labels
        i = i + 1

# Read the class list from train or test set
ds_class_names = ds_train.class_names



####
#  Building NN model
####
input_shape = (img_height, img_width, 1)

 

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=input_shape),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(3, name="outputs")
])


# Compile the built model with the training dataset
# model.compile(optimizer='adam',
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # metrics=['accuracy'])
              
# model.compile(loss=tf.keras.losses.categorical_crossentropy,
              # optimizer=tf.keras.optimizers.Adam(),
              # metrics=['accuracy'])
              
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])              
              
test_label = tf.keras.utils.to_categorical(test_label, 3)


history = model.fit(ds_train, validation_data=ds_val, epochs=epochs_train)


# Accuracy evaluation
test_loss, test_acc = model.evaluate(ds_test, verbose=2)
print('\nThe test accuracy : ', test_acc)


## Plotting the training
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


# Making predictions
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_data)


# Plot the first X test images, their predicted labels, and the true labels.
# Correct predictions are in blue and incorrect predictions are in red.
num_rows = 4
num_cols = 5
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  Plot_PredImg(predictions[i], test_data[i], int(test_label[i]))
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  Plot_PredVal( predictions[i],  int(test_label[i]))
plt.tight_layout()
plt.show()