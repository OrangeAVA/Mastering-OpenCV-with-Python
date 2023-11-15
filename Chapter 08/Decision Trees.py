import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier

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
    for filename in os.listdir(folder_path)[:50]:
     
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        gray_image = cv2.resize(gray_image, (200,200))
        # Compute the histogram
        histogram = cv2.calcHist([gray_image], [0], None, [num_bins], [0, num_bins])
        
        # Flatten the histogram and append it to the features list
        features.append(gray_image.flatten())
        
        # Append the class label to the labels list
        labels.append(class_label)

# Convert the lists to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Initialize a Random Forest classifier
classifier = DecisionTreeClassifier()

# Define the parameter grid for grid search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10,15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [5,10,15,20,25]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters and best score from grid search
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Use the best model from grid search for prediction
best_model = grid_search.best_estimator_
train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)