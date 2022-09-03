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
    
#end of function

############################# M A I N    S E C T I O N ######################################

## Global values
# Number of classes
num_classes = 3
datasets_types = ['train','test']
data_classes = ['cups','dishes','plates']
# Desired image resolution
img_height = 128
img_width = 128
batch_size = 1
# Path of train and test sets
dataset_subfolder =  'edgedetected'

cd_path = os.path.dirname(os.path.abspath(__file__)) #Current path
dataset_folders = ['data','data_grp']
datasetNew_foldername = 'data_new'



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
                
                # rotate the image
                (h, w) = img.shape[:2]
                (cX, cY) = (w // 2, h // 2)    
                # rotate our image by x degrees clockwise around the center of the image
                M = cv2.getRotationMatrix2D((cX, cY), -20, 1.0)
                rotated_p45 = cv2.warpAffine(img, M, (w, h))
                # rotate our image by x degrees anticlockwise around the center of the image
                M = cv2.getRotationMatrix2D((cX, cY), 20, 1.0)
                rotated_m45 = cv2.warpAffine(img, M, (w, h))

                Store_NewImagesInDir(rotated_p45, cd_path + '/' + datasetNew_foldername +  '/' + dataset_type + '/' + data_class, dataset_foldername + '_rcw_' + path)
                Store_NewImagesInDir(rotated_m45, cd_path + '/' + datasetNew_foldername +  '/' + dataset_type + '/' + data_class, dataset_foldername + '_racw_' + path)
                
                
                # rotate our image by x degrees clockwise around the center of the image
                M = cv2.getRotationMatrix2D((cX, cY), -35, 1.0)
                rotated_p45 = cv2.warpAffine(img, M, (w, h))
                # rotate our image by x degrees anticlockwise around the center of the image
                M = cv2.getRotationMatrix2D((cX, cY), 35, 1.0)
                rotated_m45 = cv2.warpAffine(img, M, (w, h))

                Store_NewImagesInDir(rotated_p45, cd_path + '/' + datasetNew_foldername +  '/' + dataset_type + '/' + data_class, dataset_foldername + '_rcw1_' + path)
                Store_NewImagesInDir(rotated_m45, cd_path + '/' + datasetNew_foldername +  '/' + dataset_type + '/' + data_class, dataset_foldername + '_racw1_' + path)
            

#####
######## TRANSLATION ###########
#####
imgProcTechnq_name = 'translation'

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
                
                # translation to right
                M = np.float32([[1, 0, 10], [0, 1, 0]])
                shifted_right = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                # translation to left
                M = np.float32([[1, 0, -10], [0, 1, 0]])
                shifted_left = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                
                Store_NewImagesInDir(shifted_right, cd_path + '/' + datasetNew_foldername  + '/' + dataset_type + '/' + data_class, dataset_foldername + '_tr_' + path)
                Store_NewImagesInDir(shifted_left, cd_path + '/' + datasetNew_foldername  + '/' + dataset_type + '/' + data_class, dataset_foldername + '_tl_' + path)
           
                # translation to right
                M = np.float32([[1, 0, 20], [0, 1, 0]])
                shifted_right = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                # translation to left
                M = np.float32([[1, 0, -20], [0, 1, 0]])
                shifted_left = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                
                Store_NewImagesInDir(shifted_right, cd_path + '/' + datasetNew_foldername  + '/' + dataset_type + '/' + data_class, dataset_foldername + '_tr1_' + path)
                Store_NewImagesInDir(shifted_left, cd_path + '/' + datasetNew_foldername  + '/' + dataset_type + '/' + data_class, dataset_foldername + '_tl1_' + path)
            


#####
######## FLIPPING ###########
#####
imgProcTechnq_name = 'flipping'

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

                # Flip the images
                flipVertical = cv2.flip(img, 0)
                flipHorizontal = cv2.flip(img, 1)
                flipBoth = cv2.flip(img, -1)
                
                Store_NewImagesInDir(flipHorizontal, cd_path + '/' + datasetNew_foldername  + '/' + dataset_type + '/' + data_class, dataset_foldername + '_fh_' + path)
       




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
                
                blur = cv2.GaussianBlur(image_bw,(5,5),0)
                
                # Threshold the images
                ret, thresh3 = cv2.threshold(blur,220,255,cv2.THRESH_TRUNC)
                
                # Edge Detection of the image
                edges_normal = cv2.Canny(thresh3,25,120)

                Store_NewImagesInDir(edges_normal, cd_path + '/' + datasetNew_foldername + '/' + imgProcTechnq_name + '/' + dataset_type + '/' + data_class, 'claheEdge_' + path)



# Use that as the dataset


data_path_train = datasetNew_foldername + '/' + dataset_subfolder + '/train'
data_path_test = datasetNew_foldername + '/' + dataset_subfolder + '/test'

####
# Prepare training and testing data
####
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    data_path_train,
    color_mode = 'grayscale',
    image_size=(img_height,img_width), # reshape
    shuffle = False,
    batch_size = batch_size,
    seed=123
)
ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    data_path_test,
    color_mode = 'grayscale',
    image_size=(img_height,img_width), # reshape
    shuffle = False,
    batch_size = batch_size,
    seed=123
)

# Prescaling and extraction of the images
train_data = np.zeros((len(ds_train.file_paths), img_height, img_width))
train_label = np.zeros(len(ds_train.file_paths))
i = 0
for images, labels in ds_train.take(-1):
        temp = images.numpy().astype("uint8")
        temp = np.squeeze(temp)
        temp = temp / 255.0
        train_data[i] = temp
        train_label[i] = labels
        i = i + 1

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
# cnn1 = Sequential()
# cnn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# cnn1.add(MaxPooling2D(pool_size=(2, 2)))
# cnn1.add(Dropout(0.2))

# cnn1.add(Flatten())

# cnn1.add(Dense(128, activation='relu'))
# cnn1.add(Dense(10, activation='softmax'))

# cnn1.compile(loss=tf.keras.losses.categorical_crossentropy,
              # optimizer=tf.keras.optimizers.Adam(),
              # metrics=['accuracy'])
    
# history1 = cnn1.fit(train_data, train_label,
          # batch_size=10,
          # epochs=10,
          # verbose=1)   

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3)
])


#Compile the built model with the training dataset
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x = train_data, y = train_label,  epochs=15)




# Accuracy evaluation
test_loss, test_acc = model.evaluate(x = test_data, y = test_label, verbose=2)
print('\nThe test accuracy : ', test_acc)





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
