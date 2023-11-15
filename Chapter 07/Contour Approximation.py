import cv2
import numpy as np

image = cv2.imread('box.jpg')
image2 = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


for contour in contours:
    # Perform contour approximation
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Draw the contour and its approximation
    cv2.drawContours(image, [approx], 0, (0, 0, 0), 1)
    cv2.drawContours(image2, [contour], 0, (0, 0, 255), 1)

cv2.imshow('Contour Approximation', image)
cv2.imshow('Contour Approximation 2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()