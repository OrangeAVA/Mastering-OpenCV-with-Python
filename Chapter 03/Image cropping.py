import cv2

img = cv2.imread("image.jpg")

# Define ROI coordinates
x1, y1 = 100, 100 # top-left corner
x2, y2 = 300, 400 # bottom-right corner

# Crop image
cropped_img = img[y1:y2, x1:x2]

cv2.imshow("Original Image", img)
cv2.imshow("Cropped Image", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()