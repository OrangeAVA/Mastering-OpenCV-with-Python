import numpy as np
import cv2

# Create a black canvas
canvas = np.zeros((600, 500, 3), dtype=np.uint8)

# Define the vertices of the triangle
p1 = (250, 100)
p2 = (100, 400)
p3 = (400, 400)

# Draw the lines using cv2.line()
cv2.line(canvas, p1, p2, (0, 255, 0), 1)
cv2.line(canvas, p3, p1, (255, 0, 0), 3)
cv2.line(canvas, p2, p3, (255, 255, 255), 10)

# Display the image
cv2.imshow("Triangle", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()