import cv2
import numpy as np

img = np.zeros((200, 450), np.uint8)

# Draw the text "CLOSING" on the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "CLOSING", (15, 125), font, 3, (255, 255, 255), 5)

noisy_img = img.copy()

# Create noise using difference of two images
noise = np.zeros_like(img)
for i in range(1000):
    x, y = np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])
    cv2.circle(noisy_img, (x, y), 1, (0, 0, 0), -1, lineType=cv2.LINE_AA)

# Define kernel for closing operation
kernel = np.ones((5, 5), np.uint8)

# Apply closing operation
closed_img = cv2.morphologyEx(noisy_img, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Noisy Image", noisy_img)
cv2.imshow("Closed", closed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()