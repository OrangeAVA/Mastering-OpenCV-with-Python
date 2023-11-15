import cv2
import numpy as np

img = cv2.imread('cat.jpg')

# Define the new size
new_size = (400, 400)

# Compute the scaling factors for x and y axis
sx = new_size[0]/img.shape[1]
sy = new_size[1]/img.shape[0]

# Define the transformation matrix
M = np.float32([[sx, 0, 0], [0, sy, 0]])

# Apply the affine transformation
resized_img = cv2.warpAffine(img, M, new_size)

cv2.imshow('Original Image', img)
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()