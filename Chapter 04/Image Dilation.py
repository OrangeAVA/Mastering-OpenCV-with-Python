import cv2
import numpy as np

img = cv2.imread("test2.jpg")

# apply dilation once to remove noise
kernel = np.ones((5,5),np.uint8)
img1 = cv2.dilate(img, kernel, iterations=1)

# apply dilation multiple times to show bad result
img2 = cv2.dilate(img, kernel, iterations=5)

cv2.imshow('Original Image', img)
cv2.imshow('Dilation', img1)
cv2.imshow('Over Dilation', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()