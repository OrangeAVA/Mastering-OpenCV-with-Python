import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152

# Load ResNet models
resnet18 = ResNet50()
resnet34 = ResNet101(weights='imagenet')
resnet50 = ResNet152(weights='imagenet')