import cv2
image = cv2.imread('img.jpg')

# apply Gaussian blur with kernel size (5,5) and standard deviation of 0
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# display the original and blurred images side by side
cv2.imshow('Original Image', image)
cv2.imshow('Gaussian Blurred Image', gaussian_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()