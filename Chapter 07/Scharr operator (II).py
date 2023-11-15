import cv2
import numpy as np

image = cv2.imread('objects.jpg', cv2.IMREAD_GRAYSCALE)

# Define Scharr-like kernels
scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)

# Compute Scharr-like gradients
gradient_x = cv2.filter2D(image, cv2.CV_32F, scharr_x)
gradient_y = cv2.filter2D(image, cv2.CV_32F, scharr_y)
gradient = np.sqrt(gradient_x**2 + gradient_y**2)

cv2.imshow("Gradient X", gradient_x)
cv2.imshow("Gradient Y", gradient_y)
cv2.imshow("Gradient ", gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

