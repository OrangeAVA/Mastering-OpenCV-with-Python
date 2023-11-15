import cv2
import numpy as np

image = cv2.imread('filter.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
filtered_contours = []
filtered_objects = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:  # Set minimum area threshold as needed
        filtered_contours.append(contour)
        x, y, w, h = cv2.boundingRect(contour)
        filtered_objects.append(image[y:y+h, x:x+w])

print(len(filtered_contours))

# Create a blank image of the same size as the original image
result = np.zeros_like(image)

# Draw the filtered contours on the result image
cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

cv2.imshow('Filtered Contours', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i, obj in enumerate(filtered_objects):
    cv2.imshow(f'Object {i+1}', obj)

cv2.waitKey(0)
cv2.destroyAllWindows()