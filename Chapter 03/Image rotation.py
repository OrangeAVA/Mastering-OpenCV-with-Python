import cv2

img = cv2.imread('image.jpg')

# Rotate clockwise by 90 degrees
rot_img_90cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Rotate counterclockwise by 90 degrees
rot_img_90ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Rotate by 180 degrees
rot_img_180 = cv2.rotate(img, cv2.ROTATE_180)

cv2.imshow('Original', img)
cv2.imshow('Rotated 90 CW', rot_img_90cw)
cv2.imshow('Rotated 90 CCW', rot_img_90ccw)
cv2.imshow('Rotated 180', rot_img_180)
cv2.waitKey(0)
cv2.destroyAllWindows()


