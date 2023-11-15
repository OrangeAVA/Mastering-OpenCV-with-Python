import cv2
import numpy as np

img = cv2.imread("image.jpg")

# apply erosion once to remove noise
kernel = np.ones((5,5),np.uint8)
img1 = cv2.erode(img, kernel, iterations=1)

# apply erosion multiple times to show bad result
img2 = cv2.erode(img, kernel, iterations=10)

cv2.imshow('Original Image', img)
cv2.imshow('Erosion', img1)
cv2.imshow('Over Erosion', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()