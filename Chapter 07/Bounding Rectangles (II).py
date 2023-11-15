import cv2
import numpy as np

image = cv2.imread('tects.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image
bounding_rect_image = image.copy()

# Loop through the contours
for contour in contours:
    # Get the bounding rectangle for the contour
    rect = cv2.minAreaRect(contour)

    # Draw a rectangle around the object
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Draw the minimum bounding rectangle on the image
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    

cv2.imshow('Original Image', gray)
cv2.imshow('Bounding Rectangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

