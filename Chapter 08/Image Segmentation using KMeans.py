import numpy as np
import cv2

image = cv2.imread('image.jpg')

# Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 3).astype(np.float32)

# Define the criteria for k-means clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set the k values
k_values = [2, 3, 7]

cv2.imshow('K-Means Segmentation', image)
cv2.waitKey(0)

# Perform k-means clustering for each k value and display the segmented images
for k in k_values:
    # Perform k-means clustering
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the centers to integers
    centers = np.uint8(centers)

    # Replace each pixel value with its corresponding cluster center value
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    cv2.imshow('K-Means Segmentation', segmented_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()