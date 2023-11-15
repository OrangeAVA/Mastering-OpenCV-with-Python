import cv2
import numpy as np

source_image = cv2.imread('scene.jpg')
template = cv2.imread('house.jpg')

# Define a range of scales to consider
scales = np.linspace(0.5, 1.5, 5)


# Variables to store best case values
best_match_value = float('inf')
best_match_location = (0, 0)
best_scale = 1.0 

for scale in scales:
    print(scale)
    # Resize the template according to the current scale
    scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
    
    template_height, template_width = scaled_template.shape[:2]

    for y in range(0, source_image.shape[0] - template_height):
        for x in range(0, source_image.shape[1] - template_width):
            region = source_image[y:y + template_height, x:x + template_width]
            
            # Mean squared difference between the region and template
            diff = np.sum((region - scaled_template)**2)
            
            if diff < best_match_value:
                best_match_value = diff
                best_match_location = (x, y)
                best_scale = scale

# Actual size of the detected object
detected_width = int(template.shape[1] * best_scale)
detected_height = int(template.shape[0] * best_scale)

cv2.rectangle(source_image, best_match_location, (best_match_location[0] + detected_width, best_match_location[1] + detected_height), (0, 255, 0), 2)

cv2.imshow('Object Detection Result', source_image)
cv2.waitKey(0)
cv2.destroyAllWindows()