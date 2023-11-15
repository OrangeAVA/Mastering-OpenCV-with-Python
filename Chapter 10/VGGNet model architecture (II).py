import tensorflow as tf
from tensorflow.keras.applications import VGG16

# Load VGG16 model without weights
vgg_model = VGG16(weights=None, include_top=True)

vgg_model.summary()