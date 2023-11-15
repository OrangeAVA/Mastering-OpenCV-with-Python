import cv2

# Load image
img = cv2.imread('ss.jpg')

# Define index values
x=50
y=60
w=75
h=75

# Extract ROI from the image
roi = img[y:y+h,x:x+w]

# Print shape of the extracted ROI
print(roi.shape)

# Assigning a colour to a different ROI
img[100:150,150:200] = (255,255,0)

cv2.imshow(‘Extracted ROI rectangle’, roi)
cv2.imshow(‘Image with ROI colour’, img)
cv2.waitKey(0)
cv2.destroyAllWindows()