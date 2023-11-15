import cv2
import os
import numpy as np

data_dir = 'LFW/'

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# List to store face samples and corresponding labels
faces = []
labels = []

# Iterate through the subdirectories (each person's images)
for label, person_dir in enumerate(os.listdir(data_dir)):
    person_path = os.path.join(data_dir, person_dir)
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
        faces.append(image)
        labels.append(label)
        
recognizer.train(faces, np.array(labels))

recognizer.save('trained_model.xml')

test_image = cv2.imread('lfw/Tim_Welsh/Tim_Welsh_0001.jpg', cv2.IMREAD_GRAYSCALE)

# Perform face recognition
label, confidence = recognizer.predict(test_image)

# Get the name corresponding to the predicted label
predicted_person = os.listdir(data_dir)[label]

cv2.putText(test_image, f"{predicted_person}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

print(f"Predicted person: {predicted_person}, Confidence: {confidence}")

cv2.imwrite('Result.jpg', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()