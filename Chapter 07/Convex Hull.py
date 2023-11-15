import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('box.jpg')
image_copy = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

# Find contours in the binary image
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Apply convex hull on the contour
    hull = cv2.convexHull(contour)
    
    # Reshape the hull points for polylines
    hull_points = hull.reshape((-1, 1, 2))
    
    # Draw the convex hull lines on the image
    cv2.polylines(image, [hull_points], True, (0, 255, 0), 2)
    cv2.drawContours(image2, [contour], 0, (255, 0, 255), 2)

cv2.imwrite("Convex.jpg ", image)
cv2.imwrite("AllContours.jpg ", image_copy)

plt.title('Convex Hull')
plt.show()