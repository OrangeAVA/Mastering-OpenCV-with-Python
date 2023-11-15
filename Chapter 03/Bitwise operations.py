import cv2
import numpy as np

# Create two black and white images
img1 = np.zeros((400, 400), dtype=np.uint8)
img2 = np.zeros((400, 400), dtype=np.uint8)

# Draw a rectangle on img1
cv2.rectangle(img1, (50, 50), (350, 350), (255, 255, 255), -1)

# Draw a circle on img2
cv2.circle(img2, (200, 200), 150, (255, 255, 255), -1)

# Perform bitwise AND
bitwise_and = cv2.bitwise_and(img1, img2)

# Perform bitwise OR
bitwise_or = cv2.bitwise_or(img1, img2)

# Perform bitwise XOR
bitwise_xor = cv2.bitwise_xor(img1, img2)

# Perform bitwise NOT on img1
bitwise_not = cv2.bitwise_not(img1)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('AND', bitwise_and)
cv2.imshow('OR', bitwise_or)
cv2.imshow('XOR', bitwise_xor)
cv2.imshow('NOT of img1', bitwise_not)
cv2.waitKey(0)
cv2.destroyAllWindows()