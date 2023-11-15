import cv2
import numpy as np

source_image = cv2.imread('coin.jpg')
template = cv2.imread('template.jpg')

# Save width and height of template
template_height, template_width = template.shape[:2]

# Template matching
result = cv2.matchTemplate(source_image, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Get the top-left corner of the detected area
top_left = max_loc

# Get the bottom-right corner of the detected area
bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

cv2.rectangle(source_image, top_left, bottom_right, (0, 255, 0), 2)

cv2.imwrite(Output.jpg', source_image)
cv2.waitKey(0)
cv2.destroyAllWindows()