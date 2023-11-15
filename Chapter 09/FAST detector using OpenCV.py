import cv2

image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# FAST detector object
fast = cv2.FastFeatureDetector_create()

# Detect keypoints using FAST
keypoints = fast.detect(gray, None)

# Draw detected keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

cv2.imwrite(output.jpg', image_with_keypoints)