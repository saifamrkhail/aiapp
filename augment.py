#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 02:49:37 2023

@author: saif
"""

import os
import cv2
#import numpy as np
from imgaug import augmenters as iaa
import shutil

# Define the augmentation parameters
augmentation = iaa.Sequential([
    iaa.Flipud(0.10),  # Flip vertically with a probability of 0.5
    iaa.Affine(rotate=(-45, 45)),  # Rotate between -45 and 45 degrees
    iaa.GaussianBlur(sigma=(0, 0.1)),  # Apply Gaussian blur with a sigma between 0 and 1.0
    iaa.Multiply((0.9, 1.1)),  # Multiply pixel values by a factor between 0.8 and 1.2
    iaa.GammaContrast(gamma=(0.9, 1.1)),  # Apply gamma contrast adjustment between 0.8 and 1.2
    iaa.Affine(scale=(0.9, 1.1)),  # Scale the image by a factor between 0.5 and 1.5
    iaa.Affine(shear=(-2, 2))  # Apply shear transformation between -16 and 16 degrees
])

def augment_image(image):
    # Apply augmentation to the image
    augmented_image = augmentation.augment_image(image)
    return augmented_image

def create_augmented_images(directory):
    image_extension = '.jpg'
    label_extension = '.txt'
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext.lower() == image_extension:
                image_path = os.path.join(root, file)
                label_path = os.path.join(root, 'labels', file_name + label_extension)
                
                # Load the original image
                image = cv2.imread(image_path)
                
                # Augment the image
                augmented_image_1 = augment_image(image)
                augmented_image_2 = augment_image(image)
                
                # Save the augmented images
                augmented_image_path_1 = os.path.join(root, file_name + '_augm_1' + image_extension)
                augmented_image_path_2 = os.path.join(root, file_name + '_augm_2' + image_extension)
                cv2.imwrite(augmented_image_path_1, augmented_image_1)
                cv2.imwrite(augmented_image_path_2, augmented_image_2)
                
                # Update the label paths for augmented images
                augmented_label_path_1 = os.path.join(root, 'labels', file_name + '_augm_1' + label_extension)
                augmented_label_path_2 = os.path.join(root, 'labels', file_name + '_augm_2' + label_extension)
                
                # Copy the original label to the augmented label paths
                os.makedirs(os.path.dirname(augmented_label_path_1), exist_ok=True)
                os.makedirs(os.path.dirname(augmented_label_path_2), exist_ok=True)
                shutil.copyfile(label_path, augmented_label_path_1)
                shutil.copyfile(label_path, augmented_label_path_2)
                
                print(f"Augmented images created for {image_path}")