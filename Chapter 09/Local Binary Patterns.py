import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the parameters for LBP
radius = 3
n_points = 8 * radius

# Initialize lists to store features and labels
features = []
labels = []

# Iterate through the folders
folder_paths = ['line',  'dotted','honey']
for folder_path in folder_paths:
    class_label = folder_path.split('/')[-1] 
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Compute LBP features
        lbp = local_binary_pattern(image, n_points, radius)
        
        # Calculate histogram of LBP features
        hist, _ = np.histogram(lbp.ravel(), bins=n_points+3, range=(0, n_points+2))
        
        # Append the features and labels
        features.append(hist)
        labels.append(class_label)

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Initialize and train the Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=300)
random_forest.fit(X_train, y_train)

# Make predictions on the test data
y_pred = random_forest.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


def predict_image_class(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute LBP features
    lbp = local_binary_pattern(image, n_points, radius)
    
    # Calculate histogram of LBP features
    hist, _ = np.histogram(lbp.ravel(), bins=n_points+3, range=(0, n_points+2))
    
    # Reshape the feature vector to match the input shape of the classifier
    feature_vector = hist.reshape(1, -1)
    
    # Predict image class
    predicted_class = random_forest.predict(feature_vector)[0]
    
    # Draw predicted label on the image
    image_with_text = cv2.putText(image.copy(), predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the image with the predicted label
    cv2.imshow('Image with Predicted Class', image_with_text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict_image_class("line/banded_0002.jpg")