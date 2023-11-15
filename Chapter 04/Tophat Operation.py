import cv2

# Read input image in grayscale
img = cv2.imread("galaxy.jpg", cv2.IMREAD_GRAYSCALE)

# Define a rectangular structuring element for the top hat operation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Perform the top hat operation
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

cv2.imshow("Original Image", img)
cv2.imshow("Top Hat Result", tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()