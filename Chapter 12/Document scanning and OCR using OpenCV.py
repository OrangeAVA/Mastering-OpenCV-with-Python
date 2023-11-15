# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 18:16:01 2023

@author: ayush
"""


import cv2
import numpy as np
import pytesseract

img = cv2.imread('document.jpg')

height, width, channels = img.shape

gray_img = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
edged_img = cv2.Canny(blur_img,75,200)
cv2.imwrite('edged.jpg',edged_img)
cv2.waitKey(0)

# Find Contours In Image
contours, hierarchy = cv2.findContours(edged_img,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  

# Find Biggest Contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
print(max_index)

# Find approxPoly Of Biggest Contour
epsilon = 0.1 * cv2.arcLength(contours[max_index], True)
approx = cv2.approxPolyDP(contours[max_index], epsilon, True)

# Crop The Image To approxPoly
pts1 = np.float32(approx)
pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (width, height))

flip = cv2.flip(result, 1) # Flip Image
# flip = cv2.rotate(flip, cv2.ROTATE_90_CLOCKWISE) # Rotate Image
   
cv2.imwrite('warp.jpg',flip)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Use Tesseract to extract text
text = pytesseract.image_to_string(flip)

# Print the extracted text
print(text)

