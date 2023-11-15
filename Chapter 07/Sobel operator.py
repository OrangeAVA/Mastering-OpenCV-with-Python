import cv2

image = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)

# Compute the gradient along x and y directions
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

cv2.imshow("Gradient X",gradient_x)
cv2.imshow("Gradient Y",gradient_y)

cv2.waitKey(0)
cv2.destroyAllWindows()