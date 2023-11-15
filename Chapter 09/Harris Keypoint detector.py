import cv2

image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Harris corner detector parameters
block_size = 2
ksize = 3
k = 0.04

# Harris corner detection
corners = cv2.cornerHarris(gray, block_size, ksize, k)

# Threshold and mark the detected corners
threshold = 0.01 * corners.max()
marked_image = image.copy()
marked_image[corners > threshold] = [0, 0, 255]

cv2.imwrite(output.jpg', marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()