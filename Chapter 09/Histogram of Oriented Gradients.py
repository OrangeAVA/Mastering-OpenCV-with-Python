import cv2
from skimage.feature import hog
from skimage import exposure

image = cv2.imread('dog.jpg', cv2.IMREAD_GRAYSCALE)

# Compute HOG features
hog_features, hog_image = hog(image, visualize=True)

# Display the HOG image
hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))
hog_image = hog_image.astype("uint8")
cv2.imshow("HOG Image", hog_image)
cv2.waitKey(0)
cv2.destroyAllWindows()