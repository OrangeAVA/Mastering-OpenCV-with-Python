import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from tensorflow.keras.preprocessing import image

# Fix image dimensions
img_width, img_height = 256, 256

# ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 
)

# Training Set
train_set = train_datagen.flow_from_directory(
    'train',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    subset='training'
)


# Validation Set
validation_set = train_datagen.flow_from_directory(
    'train',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    subset='validation' 
)

# Create CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_set, epochs=10, validation_data=validation_set)

# Save model for future references
model.save('dog_cat_classifier.h5')

def predict_image_class(model, image_path, class_names):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 

    # Predict 
    prediction = model.predict(img_array)
    predicted_class = class_names[int(prediction[0][0])]

    plt.imshow(img)
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')
    plt.show()

# Load model
loaded_model = tf.keras.models.load_model('dog_cat_classifier.h5')

class_names = ['Cat', 'Dog']

image_path_to_predict = 'path_to_your_image.jpg'
predict_image_class(loaded_model, image_path_to_predict, class_names)