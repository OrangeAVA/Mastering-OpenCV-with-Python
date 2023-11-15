import cv2
import numpy as np

img = cv2.imread("image.jpg")

# shearing parameters
shear_factor_x = 0.2
shear_factor_y = 0.3

# Obtain shearing matrices
M_x = np.array([[1, shear_factor_x, 0],
                [0, 1, 0]])
M_y = np.array([[1, 0, 0],
                [shear_factor_y, 1, 0]])

# Apply shearing transformations
rows, cols = img.shape[:2]
sheared_img_x = cv2.warpAffine(img, M_x, (cols + int(rows * shear_factor_x), rows))
sheared_img_xy = cv2.warpAffine(sheared_img_x, M_y, (cols + int(rows * shear_factor_x), rows + int(cols * shear_factor_y)))

cv2.imshow("Original Image", img)
cv2.imshow("Sheared Image (X axis)", sheared_img_x)
cv2.imshow("Sheared Image (X and Y axis)", sheared_img_xy)
cv2.waitKey(0)
cv2.destroyAllWindows()