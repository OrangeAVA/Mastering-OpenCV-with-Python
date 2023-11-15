import cv2

img = cv2.imread('image.jpg')

# Apply bilateral filter with high sigma values
filtered_img_high = cv2.bilateralFilter(img, 15, 200, 200)

# Apply bilateral filter with low sigma values
filtered_img_low = cv2.bilateralFilter(img, 15, 50, 50)

cv2.imshow('Original', img)
cv2.imshow('Filtered (high)', filtered_img_high)
cv2.imshow('Filtered (low)', filtered_img_low)
cv2.waitKey(0)
cv2.destroyAllWindows()