import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img.jpg', 0)

# Calculate histogram
hist_input = cv2.calcHist([img],[0],None,[256],[0,256])

# Perform histogram equalization
equalized = cv2.equalizeHist(img)

# Calculate histogram of equalized image
hist_equalized = cv2.calcHist([equalized],[0],None,[256],[0,256])

# Save equalized image to file
cv2.imshow("Original Image", img)
cv2.imshow(“Equalized_image”, equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot histograms using matplotlib
plt.plot(hist_input, color='blue')
plt.fill_between(range(len(hist_input)), hist_input.flatten(), color='blue')
plt.xlim([0,255])
plt.show()

plt.plot(hist_equalized, color='blue')
plt.xlim([0,255])
plt.show()