#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 02:49:37 2023

@author: saif
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os

def explore(directory):
    
    # Define the class labels
    class_labels = ['lion', 'tiger', 'cheetah']
    
    # Count the number of images in each class
    class_counts = []
    for class_label in class_labels:
        class_dir = os.path.join(directory, class_label)
        num_images = len(os.listdir(class_dir))
        class_counts.append(num_images)
    
    # Plotting the class distribution
    plt.figure(figsize=(8, 6))
    plt.bar(class_labels, class_counts)
    plt.xlabel('Class Labels')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.pie(class_counts, labels=class_labels, autopct='%1.1f%%')
    plt.title('Class Distribution')
    plt.show()
    
    # Define the class labels and corresponding indices
    class_labels = ['lion', 'tiger', 'cheetah']
    
    # Initialize variables for statistics
    mean_values = []
    std_values = []
    min_values = []
    max_values = []
    
    # Loop through each class directory
    for class_label in os.listdir(directory):
        class_dir = os.path.join(directory, class_label)
    
        # Loop through each image in the class directory
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            
            # Skip directories and process only image files
            if not os.path.isfile(image_path):
                continue
            
            image = Image.open(image_path)
                        
            # Retrieve the shape of the image
            image_shape = cv2.imread(image_path).shape
            
            # Print the shape to determine the input shape for the fine-tuning code
            print(image_shape)

            
    
            # Convert image to numpy array
            image_array = np.array(image)
    
            # Calculate statistics
            mean = np.mean(image_array)
            std = np.std(image_array)
            minimum = np.min(image_array)
            maximum = np.max(image_array)
    
            # Append statistics to respective lists
            mean_values.append(mean)
            std_values.append(std)
            min_values.append(minimum)
            max_values.append(maximum)
    
    # Calculate overall statistics
    overall_mean = np.mean(mean_values)
    overall_std = np.mean(std_values)
    overall_min = np.min(min_values)
    overall_max = np.max(max_values)
    
    # Print the calculated statistics
    print(f"Overall Mean: {overall_mean}")
    print(f"Overall Standard Deviation: {overall_std}")
    print(f"Overall Minimum Pixel Value: {overall_min}")
    print(f"Overall Maximum Pixel Value: {overall_max}")
