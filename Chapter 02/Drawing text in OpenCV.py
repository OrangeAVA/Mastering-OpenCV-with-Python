import numpy as np
import cv2

# create a blank image
img = np.zeros((600, 500, 3), dtype=np.uint8)

# define the text to be displayed
text = "Hello World!"

# set the text color and position
color = (255, 0, 0)
pos = (50, 200)

# display the text using cv2.putText()
cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

cv2.imshow("Image with text", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

