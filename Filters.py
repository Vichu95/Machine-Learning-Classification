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
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
#############################################################################################


######################### F U N C T I O N     S E C T I O N #################################

#############################################################################################
# Name        : Convert_BGR_to_RGB
# Description : The OpenCV library stores the Red Green Blue components in the order B G R, which was the old format used. 
#               This is converted in to R G B for other functions.
# Input       : Image
# Output      : Image after converting to RGB
#############################################################################################
def Convert_BGR_to_RGB(image_temp):
    b,g,r = cv2.split(image_temp)       # get b,g,r
    image_temp = cv2.merge([r,g,b]) 
    return image_temp


############################# M A I N    S E C T I O N ######################################

imgCup = cv2.imread('data/1.1.jpeg')
imgPlate = cv2.imread('data/PXL_20220909_190648851.jpg')
imgDish = cv2.imread('data/3.6.jpeg')



'''
##Median Filter
imgCupFlt = cv2.medianBlur(imgCup,7)
imgPlateFlt = cv2.medianBlur(imgPlate,7)
imgDishFlt = cv2.medianBlur(imgDish,7)


#plot
fig = plt.figure()
fig.suptitle('Median Filter', fontsize=15)

plt.subplot(2,3,1)
plt.imshow(Convert_BGR_to_RGB(imgCup))
plt.title('Original Cup')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(Convert_BGR_to_RGB(imgPlate))
plt.title('Original Plate')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(Convert_BGR_to_RGB(imgDish))
plt.title('Original Dish')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,4)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt))
plt.title('Filtered Cup')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,5)
plt.imshow(Convert_BGR_to_RGB(imgPlateFlt))
plt.title('Filtered Plate')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,6)
plt.imshow(Convert_BGR_to_RGB(imgDishFlt))
plt.title('Filtered Dish')
plt.xticks([]), plt.yticks([])


plt.show()
cv2.waitKey(0)





imgCupFlt3 = cv2.medianBlur(imgCup,3)
imgCupFlt5 = cv2.medianBlur(imgCup,5)
imgCupFlt7 = cv2.medianBlur(imgCup,7)

fig = plt.figure()
fig.suptitle('Median Filter with different kernel size', fontsize=15)

plt.subplot(1,4,1)
plt.imshow(Convert_BGR_to_RGB(imgCup))
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,2)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt3))
plt.title('Kernel Size 3')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,3)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt5))
plt.title('Kernel Size 5')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,4)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt7))
plt.title('Kernel Size 7')
plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)

'''

'''
##Averaging Filter
imgCupFlt = cv2.blur(imgCup,(5,5))
imgPlateFlt = cv2.blur(imgPlate,(5,5))
imgDishFlt = cv2.blur(imgDish,(5,5))


#plot
fig = plt.figure()
fig.suptitle('Averaging Filter', fontsize=15)

plt.subplot(2,3,1)
plt.imshow(Convert_BGR_to_RGB(imgCup))
plt.title('Original Cup')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(Convert_BGR_to_RGB(imgPlate))
plt.title('Original Plate')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(Convert_BGR_to_RGB(imgDish))
plt.title('Original Dish')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,4)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt))
plt.title('Filtered Cup')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,5)
plt.imshow(Convert_BGR_to_RGB(imgPlateFlt))
plt.title('Filtered Plate')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,6)
plt.imshow(Convert_BGR_to_RGB(imgDishFlt))
plt.title('Filtered Dish')
plt.xticks([]), plt.yticks([])


plt.show()
cv2.waitKey(0)



imgCupFlt3 = cv2.blur(imgCup,(3,3))
imgCupFlt5 = cv2.blur(imgCup,(5,5))
imgCupFlt7 = cv2.blur(imgCup,(7,7))

fig = plt.figure()
fig.suptitle('Averaging Filter with different kernel size', fontsize=15)

plt.subplot(1,4,1)
plt.imshow(Convert_BGR_to_RGB(imgCup))
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,2)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt3))
plt.title('Kernel Size 3')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,3)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt5))
plt.title('Kernel Size 5')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,4)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt7))
plt.title('Kernel Size 7')
plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)

'''


##Gaussian Filter
imgCupFlt = cv2.GaussianBlur(imgCup,(5,5),cv2.BORDER_DEFAULT)
imgPlateFlt = cv2.GaussianBlur(imgPlate,(5,5),cv2.BORDER_DEFAULT)
imgDishFlt = cv2.GaussianBlur(imgDish,(5,5),cv2.BORDER_DEFAULT)


#plot
fig = plt.figure()
fig.suptitle('Gaussian Filter', fontsize=15)

plt.subplot(2,3,1)
plt.imshow(Convert_BGR_to_RGB(imgCup))
plt.title('Original Cup')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(Convert_BGR_to_RGB(imgPlate))
plt.title('Original Plate')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(Convert_BGR_to_RGB(imgDish))
plt.title('Original Dish')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,4)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt))
plt.title('Filtered Cup')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,5)
plt.imshow(Convert_BGR_to_RGB(imgPlateFlt))
plt.title('Filtered Plate')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,6)
plt.imshow(Convert_BGR_to_RGB(imgDishFlt))
plt.title('Filtered Dish')
plt.xticks([]), plt.yticks([])


plt.show()
cv2.waitKey(0)



imgCupFlt3 = cv2.GaussianBlur(imgCup,(3,3),cv2.BORDER_DEFAULT)
imgCupFlt5 = cv2.GaussianBlur(imgCup,(5,5),cv2.BORDER_DEFAULT)
imgCupFlt7 = cv2.GaussianBlur(imgCup,(7,7),cv2.BORDER_DEFAULT)

fig = plt.figure()
fig.suptitle('Gaussian Filter with different kernel size', fontsize=15)

plt.subplot(1,4,1)
plt.imshow(Convert_BGR_to_RGB(imgCup))
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,2)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt3))
plt.title('Kernel Size 3')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,3)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt5))
plt.title('Kernel Size 5')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,4)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt7))
plt.title('Kernel Size 7')
plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)


imgCupFlt3 = cv2.GaussianBlur(imgCup,(5,5),cv2.BORDER_CONSTANT)
imgCupFlt5 = cv2.GaussianBlur(imgCup,(5,5),cv2.BORDER_REFLECT)
imgCupFlt7 = cv2.GaussianBlur(imgCup,(5,5),cv2.BORDER_TRANSPARENT)

fig = plt.figure()
fig.suptitle('Gaussian Filter with different Border parameters', fontsize=15)

plt.subplot(1,4,1)
plt.imshow(Convert_BGR_to_RGB(imgCup))
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,2)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt3))
plt.title('BORDER_CONSTANT')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,3)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt5))
plt.title('BORDER_REFLECT')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,4)
plt.imshow(Convert_BGR_to_RGB(imgCupFlt7))
plt.title('BORDER_TRANSPARENT')
plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)