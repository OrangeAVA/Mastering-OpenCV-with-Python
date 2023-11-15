import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define number of bins for the histogram
num_bins = 8

# Define the range for each bin
bin_range = [0, 256]

# Calculate the histogram with 256 bins
hist = cv2.calcHist([gray_img], [0], None, [num_bins], bin_range)


# Calculate the histogram with 8 bins
hist2 = cv2.calcHist([gray_img], [0], None, [256], bin_range)

# Plot the histogram using matplotlib
plt.plot(hist)
plt.xlim([0, 255])
plt.show()

plt.plot(hist2)
plt.xlim([0, num_bins-1])
plt.show()