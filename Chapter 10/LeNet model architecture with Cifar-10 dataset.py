import tensorflow as tf
from tensorflow.keras import layers, models, datasets

def lenet():

    # Initialize LeNet model
    model = models.Sequential(name="LeNet")
    
    # LeNet architecture
    model.add(layers.Conv2D(6, (5, 5), activation="tanh", input_shape=(32, 32, 1), name="Conv1"))
    model.add(layers.MaxPooling2D((2, 2), name="MaxPool1"))
    
    model.add(layers.Conv2D(16, (5, 5), activation="tanh", name="Conv2"))
    model.add(layers.MaxPooling2D((2, 2), name="MaxPool2"))
    
    model.add(layers.Flatten(name="Flatten"))
    
    model.add(layers.Dense(120, activation="tanh", name="Dense1"))
    model.add(layers.Dense(84, activation="tanh", name="Dense2"))
    model.add(layers.Dense(10, activation='softmax', name="Output"))
    
    return model

lenet_model = lenet()

# Display the summary of the model
lenet_model.summary()

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create the LeNet model using the Lenet function defined earlier
model = lenet()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50, batch_size=32, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

# Visualize accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()