import cv2

img = cv2.imread('input_image.jpg')

# Resize the image to half its size
resized_img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

# Resize the image to a specific width and height
resized_img = cv2.resize(img, (640, 480))

cv2.imshow('Original Image', img)
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()