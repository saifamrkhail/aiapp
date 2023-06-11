#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jun  9 02:49:37 2023

@author: saif
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import image_dataset_from_directory
# Data preprocessing
# Download Lion, Tiger and Cheetah data with oidv6

train_dir = 'train/'
validation_dir = 'validation/'
test_dir = 'test/'

BATCH_SIZE = 32
IMG_SIZE = (224, 224)


train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             label_mode='categorical')

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  label_mode='categorical')

test_dataset = image_dataset_from_directory(test_dir,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE,
                                            label_mode='categorical')

class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(3):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        # Convert one-hot encoded label to class label
        label_index = tf.argmax(labels[i])
        class_label = class_names[label_index]

        plt.title(class_label)
        plt.axis("off")

# print('Number of validation batches: %d' %
#      tf.data.experimental.cardinality(validation_dataset))
# print('Number of test batches: %d' %
#      tf.data.experimental.cardinality(test_dataset))

# Configure the dataset for performance

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Use data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(
        factor=(-0.2, 0.3),
        fill_mode='constant',
        interpolation='bilinear',
        seed=None,
        fill_value=0.0),
    tf.keras.layers.Resizing(
        224,
        224,
        interpolation='bilinear',
        crop_to_aspect_ratio=True),
    tf.keras.layers.Rescaling(
        scale=1./255,
        offset=0.0),
    tf.keras.layers.RandomFlip(
        mode='horizontal_and_vertical',
        seed=None),
    tf.keras.layers.RandomContrast(
        0.2,
        seed=None),
    tf.keras.layers.RandomZoom(
        height_factor=(0.2, 0.3),
        width_factor=(0.2, 0.3),
        fill_mode='constant',
        interpolation='bilinear',
        seed=None,
        fill_value=0.0)
])

for image, _ in train_dataset.take(1):
    plt.figure(figsize=(256, 256))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0])
        plt.axis('off')

preprocess_input = tf.keras.applications.resnet50.preprocess_input

# Create the base model from the pre-trained convnets

# Load the VGG19 model (pretrained on ImageNet)
base_model = VGG19(include_top=False,
                   weights='imagenet',
                   input_shape=(224, 224, 3))
# freeze the base model
base_model.trainable = False

# Let's take a look at the base model architecture
base_model.summary()

# Specify the layer until which you want to keep the layers unchanged
layer_name = 'block4_conv4'

# Set the flag to indicate whether to freeze the layers or not
freeze_layers = True

index_of_block4_conv4_layer = None
for i, layer in enumerate(base_model.layers):
    if layer.name == 'block4_conv4':
        index = i
        break


# Split the model into two parts based on the desired index
# base = base_model.layers[:index+1]
# unused_layers = base_model.layers[index+1:]

# Add a naive inception layer
# (output filter size should be 512, each padding same, activations leaky relu)
naiv_inception = Conv2D(512,
                        (1, 1),
                        padding='same',
                        activation='relu')(base_model.layers[:index+1].output)

# Add conv layer (kernel 3x3,  filters 512, padding valid, stride 2, activation relu)
conv3 = Conv2D(512,
               (3, 3),
               padding='valid',
               strides=2,
               activation='relu')(naiv_inception)

# Add conv layer (kernel 1x1, filters 640, padding valid, stride 1, activation relu)
conv1 = Conv2D(640,
               (1, 1),
               padding='valid',
               strides=1,
               activation='relu')(conv3)

# Freeze conv2 layers and before
x = naiv_inception
for layer in base_model.layers[index+1:]:
    layer.trainable = False  # Freeze each layer
    x = layer(x)

#for layer in base_model.layers:
#    print(layer.name, layer.trainable)

# Create the final prediction layer
x = Flatten()(x)
prediction = Dense(3, activation='softmax')(x)
rebuild_vgg = Model(inputs=base_model.input, outputs=prediction)
rebuild_vgg.summary()


#base_model.compile(optimizer=keras.optimizers.Adam(),
#                   loss=keras.losses.CategoricalCrossentropy(from_logits=True),
#                   metrics=[keras.metrics.CategoricalAccuracy()])

#base_model.fit(train_dataset,
#               epochs=10,
#               validation_data=validation_dataset,
#               verbose=1)
