import cv2

image = cv2.imread("12.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Laplacian with default ksize
laplacian_default = cv2.Laplacian(image, cv2.CV_64F)

# Apply Laplacian with higher ksize (e.g., 11)
laplacian_higher = cv2.Laplacian(image, cv2.CV_64F, ksize=7)

# Convert the results to unsigned 8-bit for visualization
laplacian_default = cv2.convertScaleAbs(laplacian_default)
laplacian_higher = cv2.convertScaleAbs(laplacian_higher)

# Display the original image and the Laplacian results
cv2.imshow("Original Image", image)
cv2.imshow("Laplacian (Default ksize)", laplacian_default)
cv2.imshow("Laplacian (Higher ksize)", laplacian_higher)
cv2.waitKey(0)
cv2.destroyAllWindows()