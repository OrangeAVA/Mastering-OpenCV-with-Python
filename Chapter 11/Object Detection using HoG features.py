from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

postitive_data= []
positive_labels = []

positive_folder='lfw'
positive_features = []

for folder in os.listdir(positive_folder):
    if not os.path.isdir(os.path.join(positive_folder, folder)):
        continue
    
    subfolder_path = os.path.join(positive_folder, folder)
    for filename in os.listdir(subfolder_path):
        img = cv2.imread(os.path.join(subfolder_path, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(64,64))
        feature = hog(img, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
        postitive_data.append(feature)
        positive_labels.append(1)        
        
negative_data= []
negative_labels = []

negative_folder='scene'
negative_features = []

for folder in os.listdir(negative_folder):
    if not os.path.isdir(os.path.join(negative_folder, folder)):
        continue
    
    subfolder_path = os.path.join(negative_folder, folder)
    for filename in os.listdir(subfolder_path)[:200]:
        img = cv2.imread(os.path.join(subfolder_path, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(64,64))
        feature = hog(img, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
        negative_data.append(feature)
        negative_labels.append(0)        

data = postitive_data+negative_data
labels = positive_labels+negative_labels

(train_data, test_data, train_labels, test_labels) = train_test_split(
	np.array(data), labels, test_size=0.2)

model = SVC(kernel='linear', C=1.0)
model.fit(train_data, train_labels)

predictions = model.predict(test_data)
print(classification_report(test_labels, predictions))

img= cv2.imread("coco1.jpg")
final = img.copy()

detections = []
confidences = []

(width, height)= (64,64)
windowSize=(width,height)
downscale=1.5

for scale, resized in enumerate(pyramid_gaussian(img, downscale=1.5)):
    for y in range(0, resized.shape[0] - height, 10):
        for x in range(0, resized.shape[1] - width, 10):
            window = resized[y: y + height, x: x + width]
            
            if window.shape[0] == height and window.shape[1] == width and window.shape[2] == 3:

                window = cv2.cvtColor((window*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)  
                features = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')
                features = features.reshape(1, -1)
                pred = model.predict(features)
                
                if pred == 1 and model.decision_function(features) > 0.8:
                    print("Detection:: Location -> ({}, {})".format(x, y))
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)),
                                       int(width * (downscale**scale)),
                                       int(height * (downscale**scale))))
                    confidences.append(model.decision_function(features))

for box in detections:
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

# Convert detected boxes NumPy array
boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.int32)
indices = cv2.dnn.NMSBoxes(boxes, np.array(confidences).ravel(), 0.5, 0.3)

for idx in indices:
    x, y, w, h = boxes[idx]
    cv2.rectangle(final, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

# Display the image with bounding boxes
cv2.imshow("All Detections", img)
cv2.imshow("After NMS", final)
cv2.waitKey(0)
cv2.destroyAllWindows()