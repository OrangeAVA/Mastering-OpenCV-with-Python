import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load the MNIST dataset from TensorFlow
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the image data
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# Initialize Decision Tree classifier
classifier = RandomForestClassifier(criterion = "entropy")

# Train out Decision Tree model
classifier.fit(x_train, y_train)

# Evaluate the model on the training set
train_predictions = classifier.predict(x_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

# Evaluate the model on the testing set
test_predictions = classifier.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test Accuracy:", test_accuracy)

# Generate confusion matrix
cm = confusion_matrix(y_test, test_predictions)
print("Confusion Matrix:")
print(cm)