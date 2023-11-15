import cv2

# Load image in grayscale
img = cv2.imread('image.jpg', 0)

# Create CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

clahe2 = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8,8))

clahe3 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24,24))

# Apply CLAHE to image
clahe_img = clahe.apply(img)
clahe_img2 = clahe2.apply(img)
clahe_img3 = clahe3.apply(img)

cv2.imshow('Original', img)
cv2.imshow('CLAHE 1', clahe_img)
cv2.imshow('CLAHE 2', clahe_img2)
cv2.imshow('CLAHE 3', clahe_img3)
cv2.waitKey(0)
cv2.destroyAllWindows()