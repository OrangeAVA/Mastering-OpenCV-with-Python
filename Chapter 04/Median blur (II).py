import cv2
import numpy as np

noisy_img = cv2.imread('22.jpg')

# Apply Median Blur
denoised_img = cv2.medianBlur(noisy_img, 3)

cv2.imshow('Noisy Image', noisy_img)
cv2.imshow('Denoised Image', denoised_img)
cv2.waitKey(0)
cv2.destroyAllWindows()