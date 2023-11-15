import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV


# Set the paths to the image folders
folder_paths = ['airplanes', 'car_side', 'helicopter','motorbikes']

# Set the number of bins for the histogram
num_bins = 256

# Initialize lists to store the features and labels
features = []
labels = []

# Iterate over the image folders
for folder_index, folder_path in enumerate(folder_paths):
    # Get the class label from the folder name
    class_label = folder_index
    print(len(os.listdir(folder_path)))
    # Iterate over the images in the folder
    for filename in os.listdir(folder_path)[:80]:
        # Read the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute the histogram
        histogram = cv2.calcHist([gray_image], [0], None, [num_bins], [0, num_bins])
        
        # Flatten the histogram and append it to the features list
        features.append(histogram.flatten())
        
        # Append the class label to the labels list
        labels.append(class_label)

# Convert the lists to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Initialize a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=200, max_depth=25, max_features=30)
# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the training set
train_predictions = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

# Make predictions on the testing set
test_predictions = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test Accuracy:", test_accuracy)


# Display images with predicted class
num_images = 8
selected_indices = np.random.choice(len(X_test), num_images, replace=False)
selected_images = X_test[selected_indices]
selected_labels = [folder_paths[label] for label in test_predictions[selected_indices]]

for i, image_index in enumerate(selected_indices):
    folder_index = int(y_test[image_index])
    folder_path = folder_paths[folder_index]
    filename = os.listdir(folder_path)[image_index % 80]
    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path)
    
    # Draw the predicted class on the image
    cv2.putText(image, str(selected_labels[i]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow(f"Image {i+1}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()