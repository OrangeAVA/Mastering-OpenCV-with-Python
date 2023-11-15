import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10

# Initialize empty lists for dataset and labels
dataset = []
labels = []

# Load CIFAR-10 dataset
(x_train, y_train), (_, _) = cifar10.load_data()

# Select the desired classes
selected_classes = [0, 1, 2, 3, 4, 5, 6 , 7, 8, 9]  
num_images_per_class = 100
num_classes = len(selected_classes)

selected_images = []
test = []
# Iterate over the dataset and extract the desired images
for class_idx in selected_classes:
    class_images = x_train[y_train.flatten() == class_idx]
    selected_images.extend(class_images[:num_images_per_class])

# Convert the list of selected images to a NumPy array
selected_images = np.array(selected_images)

# Reshape the images to a flattened shape
flattened_images = selected_images.reshape(-1, np.prod(selected_images.shape[1:]))

# Initialize these values to the dataset list created earlier
dataset = flattened_images

# Initialize labels for each class
labels = [0]*1000
labels = [i // 100 for i in range(1000)]

# Convert the dataset and labels to NumPy arrays
dataset = np.array(dataset)
labels = np.array(labels)

# Split the extracted images into Train and Test data
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2)

# Create a logistic regression model
logreg = LogisticRegression(C=0.1)

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

import cv2
import numpy as np
from tensorflow.keras.datasets import cifar10

# Function to draw predicted class on the image
def draw_predicted_class(image, predicted_class):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label = class_names[predicted_class]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv2.putText(image, label, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image

# Load CIFAR-10 dataset
(_, _), (x_test, y_test) = cifar10.load_data()

# Select one random test image
index = np.random.randint(len(x_test))
image = x_test[index]
true_label = y_test[index]

# Make a prediction on the selected image
selected_image = image.reshape(1, -1)
predicted_label = logreg.predict(selected_image)

# Draw predicted class on the image
image_with_label = draw_predicted_class(resized_image, predicted_label[0])

cv2.imshow("log_final_res.jpg", image_with_label)
cv2.waitKey(0)
cv2.destroyAllWindows()