import numpy as np
import cv2

# Initialize two sample 3x3 images
img1 = np.array([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]], dtype=np.uint8)
img2 = np.array([[100, 200, 150],
                 [50, 250, 100],
                 [150, 200, 50]], dtype=np.uint8)

# Add the images
cv2_add = cv2.add(img1, img2)
print("cv2.add() result:\n", cv2_add)

# Add the images using numpy addition
numpy_add = img1 + img2
print("Numpy addition result:\n", numpy_add)