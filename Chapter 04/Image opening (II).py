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
img2 = cv2.add(img, noise)

# Define a 5x5 kernel for the opening operation
kernel = np.ones((5, 5), np.uint8)

# Perform the opening operation on the image
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

cv2.imshow("Noisy Image", noise))
cv2.imshow("Opening Result", opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

