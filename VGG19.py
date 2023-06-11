#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jun  9 02:49:37 2023

Download the data from Open data with oidv6.
Make sure 

train_dir = 'train/'
validation_dir = 'validation/'
test_dir = 'test/'

are pointing to the correct directories.

@author: saif
"""
import matplotlib.pyplot as plt
from sympy import false
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.utils import image_dataset_from_directory
from keras.utils import load_img
from keras.utils import img_to_array

from keras.preprocessing.image import ImageDataGenerator


from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import RandomFlip
from keras.layers import RandomContrast
from keras.layers import RandomTranslation
from keras.layers import GlobalAveragePooling2D

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import glob


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


# Load the VGG19 model with pre-trained weights
vgg19 = VGG19(include_top=False,
              weights='imagenet',
              input_shape=(224, 224, 3))

# Add a new output layer
x = vgg19.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(3, activation='softmax')(x)

# Create the new model
vgg19 = Model(inputs=vgg19.input, outputs=predictions)
# Freeze the layers in the base model
vgg19.trainable = False

# Compile the model
vgg19.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
history = vgg19.fit(train_dataset,
                    epochs=10,
                    validation_data=validation_dataset,
                    verbose=1)
print("\nVGG19 model as it comes from keras")
# Learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# Evaluate the model on the test dataset
test_loss, test_accuracy = vgg19.evaluate(test_dataset)
# Print the test accuracy
print('Test Accuracy:', test_accuracy)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    'test/',
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

predictions = vgg19.predict_generator(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

true_labels = test_generator.classes

# Create a confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Get the categories/labels
categories = ['cheetah', 'lion', 'tiger']

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Get a list of all subdirectories in the test directory
subdirectories = glob.glob(os.path.join(test_dir, '*'))

# Iterate over the subdirectories and find the first image file
for subdir in subdirectories:
    img_files = glob.glob(os.path.join(subdir, '*.jpg'))
    if len(img_files) > 0:
        img_path = img_files[0]
        break

if img_path is None:
    print("No image files found in the subdirectories of the test directory.")
else:
    # Calculate inference time
    start_time = tf.timestamp()

    image = load_img(img_path, target_size=(224, 224))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = vgg19.predict(input_arr)

    end_time = tf.timestamp()

    inference_time = end_time - start_time
    print("Inference time for VGG as it comes from keras:", inference_time)

# Use data augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip(
        mode='horizontal_and_vertical',
        seed=None),

    RandomContrast(
        0.2,
        seed=None),

    RandomTranslation(
        height_factor=(-0.2, 0.3),
        width_factor=(-0.2, 0.3),
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


vgg19 = VGG19(include_top=False,
              weights='imagenet',
              input_shape=(224, 224, 3))

# Add a classification head
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = vgg19(x, training=False)
x = GlobalAveragePooling2D()(x)
predictions = Dense(3, activation='softmax')(x)

# Create the new model
vgg19 = Model(inputs=inputs, outputs=predictions)

# Freeze the layers in the base model
vgg19.trainable = False

# Compile the model
vgg19.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(from_logits=False),
             metrics=['accuracy'])

# Train the model
history = vgg19.fit(train_dataset,
                    epochs=10,
                    validation_data=validation_dataset,
                    verbose=1)

print("\nVGG19 with data augmentation")

# Learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# Evaluate the model on the test dataset
test_loss, test_accuracy = vgg19.evaluate(test_dataset)

# Print the test accuracy
print('Test Accuracy:', test_accuracy)

predictions = vgg19.predict_generator(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

true_labels = test_generator.classes

# Create a confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Get the categories/labels
categories = ['cheetah', 'lion', 'tiger']

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

if img_path is None:
    print("No image files found in the subdirectories of the test directory.")
else:
    # Calculate inference time
    start_time = tf.timestamp()

    image = load_img(img_path, target_size=(224, 224))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = vgg19.predict(input_arr)

    end_time = tf.timestamp()

    inference_time = end_time - start_time
    print("Inference time for VGG with augmented data:", inference_time)


# Load the VGG19 model (pretrained on ImageNet)
base_model = VGG19(include_top=False,
                   weights='imagenet',
                   input_shape=(224, 224, 3))
# freeze the base model
base_model.trainable = False

# Let's take a look at the base model architecture
# base_model.summary()

# Specify the layer until which you want to keep the layers unchanged
layer_name = 'block4_conv4'

block4_conv4 = base_model.get_layer('block4_conv4')

# Add a naive inception layer
# (output filter size should be 512, each padding same, activations leaky relu)
naiv_inception = Conv2D(512,
                        (1, 1),
                        padding='same',
                        activation='relu')(block4_conv4.output)
naiv_inception.trainable = True

# Add conv layer (kernel 3x3,  filters 512, padding valid, stride 2, activation relu)
conv3 = Conv2D(512,
               (3, 3),
               padding='valid',
               strides=2,
               activation='relu')(naiv_inception)
conv3.trainable = True

# Add conv layer (kernel 1x1, filters 640, padding valid, stride 1, activation relu)
conv1 = Conv2D(640,
               (1, 1),
               padding='valid',
               strides=1,
               activation='relu')(conv3)
conv1.trainable = True

# Create the final prediction layer
x = Flatten()(conv1)
prediction = Dense(3, activation='softmax')(x)
rebuilt_vgg = Model(inputs=base_model.input, outputs=prediction)
rebuilt_vgg.summary()


rebuilt_vgg.compile(optimizer=keras.optimizers.Adam(),
                    loss=keras.losses.CategoricalCrossentropy(
                        from_logits=False),
           metrics=['accuracy'])

history = rebuilt_vgg.fit(train_dataset,
                          epochs=10,
                          validation_data=validation_dataset,
                          verbose=1)
print("\nVGG19 rebuilt")

# Learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# Evaluate the model on the test dataset
test_loss, test_accuracy = rebuilt_vgg.evaluate(test_dataset)

# Print the test accuracy
print('Test Accuracy:', test_accuracy)

predictions = vgg19.predict_generator(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

true_labels = test_generator.classes

# Create a confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Get the categories/labels
categories = ['cheetah', 'lion', 'tiger']

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

if img_path is None:
    print("No image files found in the subdirectories of the test directory.")
else:
    # Calculate inference time
    start_time = tf.timestamp()

    image = load_img(img_path, target_size=(224, 224))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = vgg19.predict(input_arr)

    end_time = tf.timestamp()

    inference_time = end_time - start_time
    print("Inference time for VGG rebuilt:", inference_time)
    
    
"""_summary_
What accuracy can be achieved? What is the accuracy of the train vs. test set?

VGG19:
train: loss: 7.5164 - accuracy: 0.2533
test: loss: 7.5900 - accuracy: 0.2421

VGG19 with augmented data:
train: loss: 23.2545 - accuracy: 0.7067
test: loss: 25.2565 - accuracy: 0.95

VGG19 with rebuilt layers:
train: loss: 0.2564 - accuracy: 0.9800
test: loss: 0.2912 - accuracy: 0.9098
"""

"""_summary_
On what infrastructure did you train it? What is the inference time?

The experiment was conducted on a Lenovo ThinkPad P14s with Ryzen 7 PRO 4750U and 32GB RAM.
The training was done on the CPU, since the AMD GPU could not be used with TensorFlow.

The inference time for the VGG19 model was:
tf.Tensor(0.3206009864807129, shape=(), dtype=float64)

The inference time for the VGG19 model with augmented data was:
tf.Tensor(1.0711560249328613, shape=(), dtype=float64)


The inference time for the VGG19 model with rebuilt layers was:
tf.Tensor(0.9215613581289741, shape=(), dtype=float64)

"""

"""_summary_
What are the number of parameters of the model?

The VGG19 model has the following parameters:
Total params: 20,024,384
Trainable params: 0
Non-trainable params: 20,024,384

The rebuilt VGG19 model has the following parameters:
Total params: 13,860,419
Trainable params: 3,275,267
Non-trainable params: 10,585,152

"""

"""_summary_
Which categories are most likely to be confused by the algorithm? Show results in a confusion matrix.

See generated confusion.

"""

"""
Data cleansing was done by hand.
Since automating the process would have been too time consuming, I decided to do it manually.
For cheetah there pictures of leopards.
There pictures lion, tiger and cheetah mixed with humans and other objects.
I was not sure what to do, I left them there, because in real life we encounter such situations.
"""