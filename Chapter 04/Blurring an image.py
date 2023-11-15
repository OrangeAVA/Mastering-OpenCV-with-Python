import cv2

img = cv2.imread('1.jpg')

# Apply average blurring with kernel size 3
blurred_3 = cv2.blur(img, (3, 3))

# Apply average blurring with kernel size 7
blurred_7 = cv2.blur(img, (7, 7))

# Apply average blurring with kernel size 15
blurred_15 = cv2.blur(img, (15, 15))

cv2.imshow("img",img)
cv2.imshow("blurred (3,3)", blurred_3)
cv2.imshow("blurred (7,7)", blurred_7)
cv2.imshow("blurred (15,15)", blurred_15)
cv2.waitKey(0)
cv2.destroyAllWindows()