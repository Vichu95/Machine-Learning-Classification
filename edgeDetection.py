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

def Canny_AutoThres(image, sigma=0.33):
	# Compute the median
	v = np.median(image)
	# Calculate the thresholds
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

############################# M A I N    S E C T I O N ######################################

imgCup = cv2.imread('data/1.1.jpeg', cv2.COLOR_BGR2GRAY)
imgPlate = cv2.imread('data/PXL_20220909_190648851.jpg', cv2.COLOR_BGR2GRAY)
imgDish = cv2.imread('data/3.6.jpeg', cv2.COLOR_BGR2GRAY)



'''


# remove noise
imgCup = cv2.GaussianBlur(imgCup,(3,3),0)
imgPlate = cv2.GaussianBlur(imgPlate,(3,3),0)
imgDish = cv2.GaussianBlur(imgDish,(3,3),0)

#Sobel Edge Detection
imgCupSobelx = cv2.Sobel(imgCup, cv2.CV_64F, 1, 0, 7);
imgCupSobely = cv2.Sobel(imgCup, cv2.CV_64F, 0, 1, 7);
imgCupSobelxy = cv2.Sobel(imgCup, cv2.CV_64F, 1, 1, 7);

imgPlateSobelx = cv2.Sobel(imgPlate, cv2.CV_64F, 1, 0, 7);
imgPlateSobely = cv2.Sobel(imgPlate, cv2.CV_64F, 0, 1, 7);
imgPlateSobelxy = cv2.Sobel(imgPlate, cv2.CV_64F, 1, 1, 7);

imgDishSobelx = cv2.Sobel(imgDish, cv2.CV_64F, 1, 0, 7);
imgDishSobely = cv2.Sobel(imgDish, cv2.CV_64F, 0, 1, 7);
imgDishSobelxy = cv2.Sobel(imgDish, cv2.CV_64F, 1, 1, 7);


#plot
fig = plt.figure()
fig.suptitle('Sobel Edge Detection', fontsize=15)

plt.subplot(3,4,1)
plt.imshow(Convert_BGR_to_RGB(imgCup))
plt.title('Original Cup')
plt.xticks([]), plt.yticks([])

plt.subplot(3,4,2)
plt.imshow(Convert_BGR_to_RGB(imgCupSobelx))
plt.title('Sobel x')
plt.xticks([]), plt.yticks([])

plt.subplot(3,4,3)
plt.imshow(Convert_BGR_to_RGB(imgCupSobely))
plt.title('Sobel y')
plt.xticks([]), plt.yticks([])

plt.subplot(3,4,4)
plt.imshow(Convert_BGR_to_RGB(imgCupSobelxy))
plt.title('Sobel xy')
plt.xticks([]), plt.yticks([])



plt.subplot(3,4,5)
plt.imshow(Convert_BGR_to_RGB(imgPlate))
plt.title('Original Plate')
plt.xticks([]), plt.yticks([])

plt.subplot(3,4,6)
plt.imshow(Convert_BGR_to_RGB(imgPlateSobelx))
plt.title('Sobel x')
plt.xticks([]), plt.yticks([])

plt.subplot(3,4,7)
plt.imshow(Convert_BGR_to_RGB(imgPlateSobely))
plt.title('Sobel y')
plt.xticks([]), plt.yticks([])

plt.subplot(3,4,8)
plt.imshow(Convert_BGR_to_RGB(imgPlateSobelxy))
plt.title('Sobel xy')
plt.xticks([]), plt.yticks([])



plt.subplot(3,4,9)
plt.imshow(Convert_BGR_to_RGB(imgDish))
plt.title('Original Dish')
plt.xticks([]), plt.yticks([])

plt.subplot(3,4,10)
plt.imshow(Convert_BGR_to_RGB(imgDishSobelx))
plt.title('Sobel x')
plt.xticks([]), plt.yticks([])

plt.subplot(3,4,11)
plt.imshow(Convert_BGR_to_RGB(imgDishSobely))
plt.title('Sobel y')
plt.xticks([]), plt.yticks([])

plt.subplot(3,4,12)
plt.imshow(Convert_BGR_to_RGB(imgDishSobelxy))
plt.title('Sobel xy')
plt.xticks([]), plt.yticks([])


plt.show()
cv2.waitKey(0)




imgCupFlt3 = cv2.Sobel(imgCup, cv2.CV_64F, 1, 1, 3);
imgCupFlt5 = cv2.Sobel(imgCup, cv2.CV_64F, 1, 1, 5);
imgCupFlt7 = cv2.Sobel(imgCup, cv2.CV_64F, 1, 1, 7);

fig = plt.figure()
fig.suptitle('Sobel Edge Detection with different kernel size', fontsize=15)

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


# remove noise
imgCup = cv2.GaussianBlur(imgCup,(3,3),0)
imgPlate = cv2.GaussianBlur(imgPlate,(3,3),0)
imgDish = cv2.GaussianBlur(imgDish,(3,3),0)


#Laplacian Edge Detection
imgCupLaplacian = cv2.Laplacian(imgCup, cv2.CV_64F,  7);
imgPlateLaplacian = cv2.Laplacian(imgPlate, cv2.CV_64F,  7);
imgDishLaplacian = cv2.Laplacian(imgDish, cv2.CV_64F, 7);


#plot
fig = plt.figure()
fig.suptitle('Laplacian Edge Detection', fontsize=15)


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
plt.imshow(Convert_BGR_to_RGB(imgCupLaplacian))
plt.title('Edge Detected Cup')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,5)
plt.imshow(Convert_BGR_to_RGB(imgPlateLaplacian))
plt.title('Edge Detected Plate')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,6)
plt.imshow(Convert_BGR_to_RGB(imgDishLaplacian))
plt.title('Edge Detected Dish')
plt.xticks([]), plt.yticks([])


plt.show()
cv2.waitKey(0)



imgCupLaplacian3 = cv2.Laplacian(imgCup, cv2.CV_64F, 3);
imgCupLaplacian5 = cv2.Laplacian(imgCup, cv2.CV_64F, 5);
imgCupLaplacian7 = cv2.Laplacian(imgCup, cv2.CV_64F, 7);

fig = plt.figure()
fig.suptitle('Laplacian Edge Detection with different kernel size', fontsize=15)

plt.subplot(1,4,1)
plt.imshow(Convert_BGR_to_RGB(imgCup))
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,2)
plt.imshow(Convert_BGR_to_RGB(imgCupLaplacian3))
plt.title('Kernel Size 3')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,3)
plt.imshow(Convert_BGR_to_RGB(imgCupLaplacian5))
plt.title('Kernel Size 5')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,4)
plt.imshow(Convert_BGR_to_RGB(imgCupLaplacian7))
plt.title('Kernel Size 7')
plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)

'''


#Canny Edge Detection
imgCupCanny = cv2.Canny(imgCup, 50,  100);
imgPlateCanny = cv2.Canny(imgPlate,50,  100);
imgDishCanny = cv2.Canny(imgDish, 50, 100);


#plot
fig = plt.figure()
fig.suptitle('Canny Edge Detection', fontsize=15)


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
plt.imshow((imgCupCanny), cmap = 'gray')
plt.title('Edge Detected Cup')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,5)
plt.imshow((imgPlateCanny), cmap = 'gray')
plt.title('Edge Detected Plate')
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,6)
plt.imshow((imgDishCanny), cmap = 'gray')
plt.title('Edge Detected Dish')
plt.xticks([]), plt.yticks([])


plt.show()
cv2.waitKey(0)



imgCupCanny50_100 = cv2.Canny(imgCup, 50,  100);
imgCupCanny100_150 = cv2.Canny(imgCup, 100,  150);
imgCupcannyAuto = auto_canny(imgCup);

fig = plt.figure()
fig.suptitle('Canny Edge Detection with different threshold values', fontsize=15)

plt.subplot(1,4,1)
plt.imshow(Convert_BGR_to_RGB(imgCup))
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,2)
plt.imshow((imgCupCanny50_100), cmap = 'gray')
plt.title('Threshold = (50,100)')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,3)
plt.imshow((imgCupCanny100_150), cmap = 'gray')
plt.title('Threshold = (100,150)')
plt.xticks([]), plt.yticks([])
plt.subplot(1,4,4)
plt.imshow((imgCupcannyAuto), cmap = 'gray')
plt.title('Threshold based on Median')
plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)

