import cv2
import numpy as np

image = cv2.imread('rectangles.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image
bounding_rect_image = image.copy()

for contour in contours:
    # Get the bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Draw a rectangle around the object
    cv2.rectangle(bounding_rect_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Original Image', gray)
cv2.imshow('Bounding Rectangles', bounding_rect_image)
cv2.waitKey(0)
cv2.destroyAllWindows()