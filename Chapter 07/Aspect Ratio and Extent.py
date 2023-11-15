import cv2
import numpy as np

image = cv2.imread('image.jpg')
image_copy = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Calculate aspect ratio using minimum area rectangle
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    aspect_ratio = int(width) / int(height)

    # Calculate extent
    area = cv2.contourArea(contour)
    bounding_area = width * height
    extent = area / bounding_area

    # Classify object based on aspect ratio
    if aspect_ratio >= 1.05 or aspect_ratio <=0.95:
        # Rectangle
        cv2.drawContours(image, [contour], 0, (255, 0, 255), 2)
        cv2.putText(image, 'Rectangle', (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    else:
        # Circle
        cv2.drawContours(image, [contour], 0, (0, 255, 255), 2)
        cv2.putText(image, 'Circle', (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),1)


    # Classify object based on extent
    if extent <= 1.05 and extent >= 0.95:
        # Rectangle
        print(1)
        cv2.drawContours(image_copy, [contour], 0, (255, 0, 255), 2)
        cv2.putText(image_copy, 'Rectangle', (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    else:
        # Circle
        print(2)
        cv2.drawContours(image_copy, [contour], 0, (0, 255, 255), 2)
        cv2.putText(image_copy, 'Circle', (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),1)

cv2.imshow('Aspect Ratio Classification', image)
cv2.imshow('Extent Classification', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()