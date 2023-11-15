import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('23.jpg', 0) 

# Create a binary mask
mask = np.zeros_like(image, dtype=np.uint8)
mask[100:300, 100:300] = 255 

# Apply the mask to the image
masked_image = cv2.bitwise_and(image, mask)

# Calculate and plot the histogram
histogram = cv2.calcHist([image], [0], masked_image, [256], [0, 256])
plt.plot(histogram)

cv2.imshow('Original Image', image)
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()