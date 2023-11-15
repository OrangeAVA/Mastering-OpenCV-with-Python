import cv2

img_bgr = cv2.imread('img.jpg')

# BGR to grayscale
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

cv2.imshow("Original Image", img_bgr)
cv2.imshow('Grayscale Image', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()