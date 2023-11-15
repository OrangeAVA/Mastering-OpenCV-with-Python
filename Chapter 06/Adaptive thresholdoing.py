import cv2
import matplotlib.pyplot as plt

image = cv2.imread('flower.jpg', 0)

# Apply simple thresholding
_, thresh_simple = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Apply adaptive thresholding with method=cv2.ADAPTIVE_THRESH_MEAN_C
thresh_adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Create subplots for original image and thresholded images
plt.figure(figsize=(12, 4))

# Display the original image
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image', fontsize=10),plt.xticks([]),plt.yticks([])

# Display the simple thresholded image
plt.subplot(132)
plt.imshow(thresh_simple, cmap='gray')
plt.title('Simple Threshold', fontsize=10),plt.xticks([]),plt.yticks([])

# Display the adaptive thresholded image
plt.subplot(133)
plt.imshow(thresh_adaptive, cmap='gray')
plt.title('Adaptive Threshold', fontsize=10),plt.xticks([]),plt.yticks([])

# Show the plot
plt.tight_layout()
plt.show()