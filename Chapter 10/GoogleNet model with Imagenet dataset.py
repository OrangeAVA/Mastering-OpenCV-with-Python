import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Parameters
num_classes = 101
input_shape = (224, 224, 3)
batch_size = 32
epochs = 10

# Using InceptionV3 model pre trained on ImageNet data
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze layers in base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top
x = base_model.output
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2) 

data_generator = data_gen.flow_from_directory(
    'images',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') 

val_generator = data_gen.flow_from_directory(
    'images',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')  

model.fit(
    data_generator,
    steps_per_epoch=data_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    callbacks=[early_stopping])

model.save('caltech101_inceptionv3.h5')