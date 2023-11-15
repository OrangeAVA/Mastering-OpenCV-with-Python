import cv2
import numpy as np

img = cv2.imread("input.jpg")

# Define the translation matrix
tx = 50   # x-direction
ty = 100  # y-direction
M = np.float32([[1, 0, tx], [0, 1, ty]])

# Apply the translation to the image
rows, cols, _ = img.shape
translated_img = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow("Original Image", img)
cv2.imshow("Translated Image", translated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()