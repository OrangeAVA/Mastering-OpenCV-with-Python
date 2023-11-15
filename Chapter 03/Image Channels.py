import cv2

img = cv2.imread("ss.jpg")

im1=img.copy()
im2=img.copy()
im3=img.copy()


im1[:,:,0]=0
im1[:,:,1]=0

im2[:,:,2]=0
im2[:,:,1]=0

im3[:,:,2]=0
im3[:,:,0]=0


cv2.imshow("Original Image",img)
cv2.imshow("Red channel",im1)
cv2.imshow("Blue channel",im2)
cv2.imshow("Green channel",im3)

cv2.waitKey(0)
cv2.destroyAllWindows()