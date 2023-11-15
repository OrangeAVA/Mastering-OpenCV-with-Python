from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

mnist = tf.keras.datasets.mnist

# Load the Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the images
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the KNN classifier
knn = KNeighborsClassifier()

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = knn.predict(X_val)

# Calculate the validation accuracy
val_accuracy = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", val_accuracy)

# Make predictions on the test set
y_pred_test = knn.predict(X_test)

# Calculate the test accuracy
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Test Accuracy:", test_accuracy)

# Select three random test images
indices = np.random.randint(0, len(X_test), size=3)
images = X_test[indices]
predicted_labels = y_pred_test[indices]

# Preprocess the images
reshaped_images = [cv2.cvtColor(image.reshape(28, 28), cv2.COLOR_GRAY2BGR) for image in images]

# Concatenate the images horizontally
concatenated_image = np.hstack(reshaped_images)

cv2.imshow("Images", padded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Predicted Labels:", predicted_labels)