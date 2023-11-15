import cv2

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Compute Scharr X gradient using the cv2.Scharr function
gradient_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)

# Compute Scharr-like Y gradient using the cv2.sobel function
gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=-1)

cv2.imshow("Gradient X", gradient_x)
cv2.imshow("Gradient Y", gradient_y)
cv2.waitKey(0)
cv2.destroyAllWindows()