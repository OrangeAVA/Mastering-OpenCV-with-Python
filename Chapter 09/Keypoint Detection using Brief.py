import cv2

image1 = cv2.imread('dog.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('dog2.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Set the ORB score type to BRIEF
orb.setScoreType(cv2.ORB_FAST_SCORE)

# Detect keypoints and compute descriptors using ORB
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Create a BFMatcher object
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = matcher.match(descriptors1, descriptors2)

# Sort matches by score
matches = sorted(matches, key=lambda x: x.distance)

# Draw top matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('brief.jpg', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()