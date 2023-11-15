import cv2
import numpy as np

img = cv2.imread('111.png')
rows, cols = img.shape[:2]

# Get rotation matrices.
M1 = cv2.getRotationMatrix2D((100,100), 30, 1)

M2 = cv2.getRotationMatrix2D((cols/2,rows/2), 45, 2)

M3 = cv2.getRotationMatrix2D((cols/2,rows/2), -90, 1)

#new values for the output
scale = 2
new_cols = int(cols * scale)
new_rows = int(rows * scale)

# Create output image with the new size
rotated2 = cv2.warpAffine(img, M2, (new_cols, new_rows))

# Perform rotation
rotated1 = cv2.warpAffine(img, M1, (cols, rows))
rotated2 = cv2.warpAffine(img, M2, (cols, rows))
rotated3 = cv2.warpAffine(img, M3, (cols, rows))

cv2.imshow('Original Image', img)
cv2.imshow('Rotated Image 1', rotated1)
cv2.imshow('Rotated Image 2', rotated2)
cv2.imshow('Rotated Image 3', rotated3)
cv2.waitKey(0)
cv2.destroyAllWindows()