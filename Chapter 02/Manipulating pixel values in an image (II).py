import cv2

# Load an image
img = cv2.imread('image.jpg')

# Access and manipulate the pixels
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        
        # Checking for every tenth column
        if j % 10 == 0:
            # Setting this value to 0 
            img[i,j] = [0,0,0]

# Display our result
cv2.imshow('Person', img)

cv2.waitKey(0)
cv2.destroyAllWindows()