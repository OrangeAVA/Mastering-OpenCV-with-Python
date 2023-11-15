import cv2
import numpy as np

image = cv2.imread('objects.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image
contour_image = image.copy()

# Draw contours on the copy of the original image
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

cv2.imshow('Original Image', image)
cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

