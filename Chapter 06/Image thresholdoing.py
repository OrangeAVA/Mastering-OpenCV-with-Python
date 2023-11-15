import cv2
import matplotlib.pyplot as plt

image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

# Apply binary threshold
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Apply binary inverse threshold
_, binary_inv = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Apply truncation threshold
_, trunc = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)

# Apply to zero threshold
_, to_zero = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)

# Apply to zero inverse threshold
_, to_zero_inv = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)

# Set figure size
plt.figure(figsize=(10, 8))

plt.subplot(231), plt.imshow(image, cmap='gray')
plt.title('Original Image', fontsize=10), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(binary, cmap='gray')
plt.title('Binary Threshold', fontsize=10), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(binary_inv, cmap='gray')
plt.title('Binary Inverse Threshold', fontsize=10), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(trunc, cmap='gray')
plt.title('Truncation Threshold', fontsize=10), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(to_zero, cmap='gray')
plt.title('Threshold To Zero', fontsize=10), plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(to_zero_inv, cmap='gray')
plt.title('Threshold To Zero Inverse', fontsize=10), plt.xticks([]), plt.yticks([])

plt.show()