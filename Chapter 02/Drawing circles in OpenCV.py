import numpy as np
import cv2

# Create an empty canvas
canvas = np.zeros((500, 500, 3), dtype=np.uint8)

# Define the center point
center = (250, 250)

# Define the radii of the circles
radius1 = 50
radius2 = 100
radius3 = 150

# Define the colors of the circles
color1 = (0, 0, 255) 
color2 = (255, 0, 0) 
color3 = (0, 255, 0) 

# Define the thickness of the circles
thickness1 = -1 
thickness2 = 2
thickness3 = 10

# Draw the circles on the canvas
cv2.circle(canvas, center, radius1, color1, thickness1)
cv2.circle(canvas, center, radius2, color2, thickness2)
cv2.circle(canvas, center, radius3, color3, thickness3)

# Display the image
cv2.imshow("Image", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()