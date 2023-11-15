import cv2

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(image, threshold1=100, threshold2=200)

cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()