import cv2
import numpy as np

image1 = cv2.imread('dog.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('dog2.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Create a BFMatcher object
matcher = cv2.BFMatcher()

# Match descriptors
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test to filter good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)


# Draw matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched image
cv2.imwrite(output.jpg', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()