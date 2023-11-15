import cv2
import numpy as np

image1 = cv2.imread('2211.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('22.jpg', cv2.IMREAD_GRAYSCALE)

# Rotate image
angle = 45  
rows, cols = image1.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
rotated_image = cv2.warpAffine(image1, rotation_matrix, (cols, rows))
image1 = rotated_image.copy()

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Compute RootSIFT descriptors
epsilon = 1e-7  # Small constant to avoid division by zero
descriptors1 /= (descriptors1.sum(axis=1, keepdims=True) + epsilon)
descriptors1 = np.sqrt(descriptors1)

descriptors2 /= (descriptors2.sum(axis=1, keepdims=True) + epsilon)
descriptors2 = np.sqrt(descriptors2)

# Create a BFMatcher object
matcher = cv2.BFMatcher()

# Match descriptors
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# Filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

good_matches = good_matches[:15]

# Draw matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('root2.jpg', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()