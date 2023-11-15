import cv2

img = cv2.imread('image.jpg')

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)

# Perform histogram equalization on the value channel
v_eq = cv2.equalizeHist(v)

# Merge the equalized value channel back into the HSV image
img_hsv_eq = cv2.merge((h, s, v_eq))

# Convert the equalized HSV image back to the original color space
img_eq = cv2.cvtColor(img_hsv_eq, cv2.COLOR_HSV2BGR)

cv2.imshow('Original Image', img)
cv2.imshow('Equalized Image', img_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()