import cv2
import numpy as np

# Read input image in grayscale
img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)

# Define a rectangular structuring element for the top hat operation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Perform the bottom hat operation 
bottomhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow("Original Image", img)
cv2.imshow("Bottom Hat Result", bottomhat)
cv2.waitKey(0)
cv2.destroyAllWindows()


