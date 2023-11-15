import numpy as np
import cv2

# Create a black image
img = np.zeros((600, 500, 3), dtype=np.uint8)

# Draw the figure using rectangles and lines
# Face
cv2.rectangle(img, (150, 150), (350, 400), (242, 199, 155), thickness=-1)  

# Cap
cv2.rectangle(img, (100, 50), (400, 150), (198, 131, 56), thickness=-1) 

# Mouth
cv2.rectangle(img, (200, 310), (300, 330), (128, 0, 128), thickness=2) 

# Draw the eyes on the face as X shapes
cv2.line(img, (195, 200), (212, 228), (0, 0, 0), thickness=2)
cv2.line(img, (212, 200), (195, 228), (0, 0, 0), thickness=2)
cv2.line(img, (288, 200), (305, 228), (0, 0, 0), thickness=2)
cv2.line(img, (305, 200), (288, 228), (0, 0, 0), thickness=2)

# Display the image
cv2.imshow(“Robo”, img)
cv2.waitKey(0)
cv2.destroyAllWindows()