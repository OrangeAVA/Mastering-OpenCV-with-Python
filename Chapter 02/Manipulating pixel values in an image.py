import cv2

# Load an image in grayscale mode
img = cv2.imread('ss.jpg')

# Get the pixel value at x=75, y=25
pixel_value = img[25, 75]

#Print this value
print(pixel_value)

#Manipulate value of this pixel
img[25, 75] = 0

#Rechecking value
print(pixel_value)