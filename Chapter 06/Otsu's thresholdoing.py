import cv2
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', 0) 
# Apply simple thresholding with a threshold value of 127
_, thresholded_image_simple = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Apply Otsu's thresholding
retval, thresholded_image_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the original, simple thresholding, and Otsu's thresholding results
plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(thresholded_image_simple, cmap='gray')
plt.title('Simple Thresholding'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(thresholded_image_otsu, cmap='gray')
plt.title("Otsu's Thresholding"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()