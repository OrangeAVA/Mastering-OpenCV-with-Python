import cv2

img = cv2.imread('image.jpg')

# Apply median blur with kernel size 3x3
median_3 = cv2.medianBlur(img, 3)

# Apply median blur with kernel size 7x7
median_7 = cv2.medianBlur(img, 7)

# Apply median blur with kernel size 15x15
median_15 = cv2.medianBlur(img, 15)

cv2.imshow('Original', img)
cv2.imshow('Median Blur 3', median_3)
cv2.imshow('Median Blur 7', median_7)
cv2.imshow('Median Blur 15', median_15)
cv2.waitKey(0)
cv2.destroyAllWindows()