import cv2

image = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the SURF detector and descriptor
surf = cv2.SURF_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = surf.detectAndCompute(gray, None)

# Get the total number of keypoints
num_keypoints = len(keypoints)

# Print the number of keypoints
print("Number of keypoints:", num_keypoints)

# Print the size of the descriptors
descriptor_size = descriptors.shape[1]
print("Descriptor size:", descriptor_size)