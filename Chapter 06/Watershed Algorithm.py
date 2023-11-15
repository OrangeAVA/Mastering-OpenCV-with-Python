import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
image = cv2.imread('coin.jpg')
original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
ret, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

# Perform morphological operations to remove noise and enhance regions
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
sure_bg = cv2.dilate(opening, kernel, iterations=5)

# Perform distance transform to identify markers
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

# Identify unknown regions
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Create markers for the watershed algorithm
ret, markers = cv2.connectedComponents(sure_fg)
markers += 1
markers[unknown == 255] = 0

# Apply the Watershed algorithm
cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]

# Convert images to RGB for display
gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Display the intermediate steps and final result
titles = ['Input Image', 'Binary Threshold', 'Morphological Operations','Distance Transform', 'Markers', 'Final Result']
images = [original_rgb,  binary, sure_bg,dist_transform, markers, image_rgb]

for i in range(len(titles)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()