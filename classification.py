#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 02:49:37 2023

@author: saif
"""

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os

# Define the paths to your train, validation, and test directories
train_dir = 'train/'
val_dir = 'validation/'
test_dir = 'test/'

# Define the class labels
class_labels = ['lion', 'tiger', 'cheetah']

# data exploration
import exploration
exploration.explore(train_dir)

# data augmentation
import augment 
augment.create_augmented_images(train_dir)

def normalize_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            # Load image
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)

            # Normalize image
            normalized_image = image.astype(np.float32) / 255.0

            # Save normalized image with the same name
            cv2.imwrite(image_path, normalized_image * 255.0)

normalize_images(train_dir)


# resize images
def resize_and_pad(image, target_shape):
    # Get the original image shape
    height, width, channels = target_shape

    # Resize the image while maintaining the aspect ratio
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = target_shape[1]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_shape[0]
        new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))

    # Pad the resized image with zeros to match the target shape
    padded_image = np.zeros(target_shape, dtype=np.uint8)
    padded_image[:new_height, :new_width, :] = resized_image

    return padded_image

# Define the target shape for resizing
target_shape = (224, 224, 3)

# Iterate over each class directory
class_dirs = os.listdir(train_dir)
for class_dir in class_dirs:
    class_path = os.path.join(train_dir, class_dir)

   # Skip if the directory is the label directory
    if os.path.basename(class_path) == 'label/':
        continue

    # Iterate over each image file in the class directory
    image_files = os.listdir(class_path)
    for image_file in image_files:
        image_path = os.path.join(class_path, image_file)

        # Load and resize the image
        image = cv2.imread(image_path)
        
        resized_image = resize_and_pad(image, target_shape)

        # Save the resized image with its original name
        cv2.imwrite(image_path, resized_image)

# Load the pre-trained ResNet-50 model without the top (fully connected) layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers so they are not trained during fine-tuning
for layer in base_model.layers:
    layer.trainable = False

# Add new fully connected layers for the specific classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Assuming 3 classes: lion, tiger, cheetah

# Create the fine-tuned model by combining the base model and the new layers
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with an appropriate loss function, optimizer, and metrics
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the entire model (both pre-trained and new layers) using your labeled dataset
#model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))


# Define the data generator for training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Randomly rotate images within the range of 20 degrees
    width_shift_range=0.2,  # Randomly shift the width of images by 20%
    height_shift_range=0.2,  # Randomly shift the height of images by 20%
    shear_range=0.2,  # Apply random shear transformations
    zoom_range=0.2,  # Randomly zoom into images by 20%
    horizontal_flip=True  # Randomly flip images horizontally
)

# Load the training data from the directory
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Path to the directory containing the training data
    target_size=(224, 224),  # Resize images to the desired size
    batch_size=32,  # Number of images to include in each batch
    class_mode='categorical'  # Type of labels (e.g., categorical, binary)
)

# Create an ImageDataGenerator for the validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create a generator for the validation data
validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train the model using the generator and validation data
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
