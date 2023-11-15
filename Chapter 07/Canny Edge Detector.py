import cv2
import numpy as np

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Compute the gradients using Sobel operators
gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Compute the magnitude and direction of the gradients
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y, gradient_x)
cv2.imwrite('gradient_magnitude.jpg', gradient_magnitude)
cv2.imwrite('gradient_direction.jpg', gradient_direction)

# Perform non-maximum suppression
suppressed = np.copy(gradient_magnitude)
for i in range(1, suppressed.shape[0] - 1):
    for j in range(1, suppressed.shape[1] - 1):
        direction = gradient_direction[i, j] * 180. / np.pi
        if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
            if suppressed[i, j] <= suppressed[i, j + 1] or suppressed[i, j] <= suppressed[i, j - 1]:
                suppressed[i, j] = 0
        elif (22.5 <= direction < 67.5):
            if suppressed[i, j] <= suppressed[i - 1, j + 1] or suppressed[i, j] <= suppressed[i + 1, j - 1]:
                suppressed[i, j] = 0
        elif (67.5 <= direction < 112.5):
            if suppressed[i, j] <= suppressed[i - 1, j] or suppressed[i, j] <= suppressed[i + 1, j]:
                suppressed[i, j] = 0
        else:
            if suppressed[i, j] <= suppressed[i - 1, j - 1] or suppressed[i, j] <= suppressed[i + 1, j + 1]:
                suppressed[i, j] = 0
cv2.imwrite('suppressed.jpg', suppressed)

# Perform thresholding to classify pixels as strong or weak edges
low_threshold = 30
high_threshold = 100
edges = np.zeros_like(suppressed)
edges[suppressed >= high_threshold] = 255
edges[suppressed <= low_threshold] = 0
weak_edges = np.logical_and(suppressed > low_threshold, suppressed < high_threshold)

# Perform edge tracking by connecting weak edges to strong edges
strong_edges_i, strong_edges_j = np.where(edges == 255)
for i, j in zip(strong_edges_i, strong_edges_j):
    if np.any(weak_edges[i - 1:i + 2, j - 1:j + 2]):
        edges[i - 1:i + 2, j - 1:j + 2] = 255
cv2.imwrite('edges.jpg', edges)

cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()