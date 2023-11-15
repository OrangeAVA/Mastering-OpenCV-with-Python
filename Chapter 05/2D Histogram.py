import cv2
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg')

# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split the image channels
h, s, v = cv2.split(hsv_image)

# Define the number of bins for each channel
bins = [50, 50] 

# Compute the 2D histogram
histogram = cv2.calcHist([h, s], [0, 1], None, bins, [0, 180, 0, 256])

plt.imshow(histogram, interpolation='nearest', origin='upper', aspect='auto', cmap='jet')
plt.colorbar()
plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.title('2D Histogram')
plt.show()