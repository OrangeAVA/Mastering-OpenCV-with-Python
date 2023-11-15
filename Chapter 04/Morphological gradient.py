import cv2
import numpy as np

# Read the input image
img = cv2.imread('img.jpg', 0)

# Define the kernel
kernel = np.ones((3,3), np.uint8)

# Apply morphological gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# Display the result
cv2.imshow("Image", img)
cv2.imshow('Morphological Gradient', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()