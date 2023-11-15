import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

# Split the image into its three color channels
b, g, r = cv2.split(img)

# Set the histogram parameters
histSize = 256
histRange = (0, 256)

# Compute the histograms for each color channel
b_hist = cv2.calcHist([b], [0], None, [histSize], histRange)
g_hist = cv2.calcHist([g], [0], None, [histSize], histRange)
r_hist = cv2.calcHist([r], [0], None, [histSize], histRange)

plt.plot(b_hist, color='b')
plt.plot(g_hist, color='g')
plt.plot(r_hist, color='r')
plt.xlim([0, 256])
plt.show()