import cv2

img = cv2.imread('image.png')

# Flip the image horizontally
x_flip = cv2.flip(img, 1)

# Flip the image vertically
y_flip = cv2.flip(img, 0)

# Flip the image on both axes
xy_flip = cv2.flip(img, -1)

cv2.imshow('Original Image', img)
cv2.imshow('Horizontal flip', x_flip)
cv2.imshow('Vertical flip', y_flip)
cv2.imshow('Both axes', xy_flip)
cv2.waitKey(0)
cv2.destroyAllWindows()