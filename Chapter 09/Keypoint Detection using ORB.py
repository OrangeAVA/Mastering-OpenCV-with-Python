import cv2

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = orb.detectAndCompute(image, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=0)

cv2.imwrite('orb.jpg', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()