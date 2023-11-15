import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2

# Initialize empty lists for dataset and labels
dataset = []
labels = []

train_folder = "train"

# Load images from the "dogs" folder
dog_folder = os.path.join(train_folder, 'dogs')
for filename in os.listdir(dog_folder):
    if filename.endswith('.jpg'):  
        image = cv2.imread('train/dogs/'+filename,0)
        if image is not None:
            image = cv2.resize(image,(64, 64))  
            k = image.flatten()
            dataset.append(k)           
            labels.append(0)  # Label 0 for dog images

# Load images from the "cats" folder
cat_folder = os.path.join(train_folder, 'cats')
for filename in os.listdir(cat_folder):
    if filename.endswith('.jpg'): 
        image = cv2.imread('train/cats/'+filename,0)
        if image is not None:
            image = cv2.resize(image,(64, 64)) 
            k = image.flatten()
            dataset.append(k)           
            labels.append(1)  # Label 1 for cat images
            

# Convert the dataset and labels to NumPy arrays
dataset = np.array(dataset)
labels = np.array(labels)

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2)

# Create a logistic regression model
logreg = LogisticRegression()

# Train the model
logreg.fit(dataset, labels)

# Evaluate the model on the training set
train_predictions = logreg.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

# Evaluate the model on the testing set
test_predictions = logreg.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test Accuracy:", test_accuracy)

# Select three random images
image_files = np.random.choice(os.listdir("test"), size=5, replace=False)

images = []
# Iterate over the selected image files
for image_file in image_files:
    image_path = os.path.join("test", image_file)
    
    # Read the image
    image = cv2.imread(image_path)
    
    # Preprocess images
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(gray_image, (64, 64))

    flattened_image = image.flatten()
    reshaped_image = flattened_image.reshape(1, -1)
    
    # Make a prediction on the image
    predicted_label = logreg.predict(reshaped_image)[0]
    
    # Get the class name based on the predicted label
    class_names = ['dog', 'cat']  # Update with your class names
    predicted_class = class_names[predicted_label]
    
    # Draw the predicted class on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7

    cv2.putText(image, predicted_class, (20, 20), font, font_scale, (255,255,255), 1, cv2.LINE_AA)
    
    # Add the image with the predicted class to the list
    images.append(image)

output_image = np.hstack(images)

cv2.imshow("output", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()