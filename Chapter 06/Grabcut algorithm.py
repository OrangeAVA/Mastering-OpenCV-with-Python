import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('dog.jpg')

# Create a mask with same shape as the image, initialized with zeros
mask = np.zeros(image.shape[:2], np.uint8)

# Create the background and foreground model
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Define the region of interest (ROI) as a rectangle
rect = (140,232,300,560)

# Apply GrabCut
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

# Assign 0 and 2 to the background and possible background regions in the mask
mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')

# Apply the mask to the original image to extract the foreground
result = image * mask2[:, :, np.newaxis]

# Convert images from BGR to RGB to display images using matplotlib
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)

# Display the original image and the simple thresholded image side by side
plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(result, cmap='gray')
plt.title('Grabcut'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()