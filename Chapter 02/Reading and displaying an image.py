import cv2

# Using imread to read out image
img = cv2.imread("Pictures/dog.jpg")

# Print the shape of the image
print(img.shape)

# Displaying the image
cv2.imshow("Dog Image", img)

# Wait until a key is pressed
cv2.waitKey(0)

# Close all Windows
cv2.destroyAllWindows()