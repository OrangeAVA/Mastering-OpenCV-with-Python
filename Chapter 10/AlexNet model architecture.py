import tensorflow as tf
from tensorflow.keras import layers, models

def alexnet_model():
    model = models.Sequential(name="AlexNet")
    
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3), padding='valid'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    
    model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    
    model.add(layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same'))
    
    model.add(layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding='same'))
    
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(1000, activation='softmax'))
    
    return model

alexnet = alexnet_model()

alexnet.summary()

