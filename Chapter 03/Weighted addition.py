import cv2
import numpy as np

img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Add the two images with different weights
result = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()