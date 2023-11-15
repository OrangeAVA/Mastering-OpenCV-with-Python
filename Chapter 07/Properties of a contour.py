import cv2

image = cv2.imread('objects.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to obtain a binary image
_, binary = cv2.threshold(gray, 22, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over the contours
for contour in contours:
    # Calculate the area of the contour
    area = int(cv2.contourArea(contour))
    
    # Calculate the perimeter of the contour
    perimeter = int(cv2.arcLength(contour, True))
    
    # Calculate the centroid of the contour
    M = cv2.moments(contour)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])

    print("Area:", area)
    print("Perimeter:", perimeter)
    print("Centroid (x, y):", centroid_x, centroid_y)
    
    # Draw the contour, centroid, and text on the image
    cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
    cv2.circle(image, (centroid_x, centroid_y), 5, (0, 0, 0), -1)
    cv2.putText(image, f"Area: {area}", (centroid_x - 50, centroid_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(image, f"Perimeter: {perimeter}", (centroid_x - 80, centroid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow("Object Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()