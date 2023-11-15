import cv2
import numpy as np

# Generate a 300x300 image with a black background
img = np.zeros((200, 450), np.uint8)

# Draw the text "OPENING" on the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "OPENING", (15, 125), font, 3, (255, 255, 255), 5)

# Add noise to the image
noise = np.zeros((200, 450), np.uint8)
cv2.randn(noise, 0, 50)
noisy= cv2.add(img, noise)

# Define a 5x5 kernel for the erosion and dilation operations
kernel = np.ones((5, 5), np.uint8)

# Perform erosion
erosion = cv2.erode(img, kernel, iterations=1)

# Perform dilation on eroded image
opening = cv2.dilate(erosion, kernel, iterations=1)

cv2.imshow("Original Image", img)
cv2.imshow("Noisy Image", noisy)
cv2.imshow("erosion", erosion)
cv2.imshow("Opening Result", opening)
cv2.waitKey(0)
cv2.destroyAllWindows()