# Classification of Lion, Tiger and Cheetah

The project must contain the following parts:

- Dataset preparation: Train/validation/test split, data cleansing
- Experiment: Transfer learning (training the network on your data with random weights vs. using pre-trained weights)
- Experiment: Data augmentation (eg. crop, mosaic, rotate, color) and what effect does it have on the evaluation metric

Tasks
- [ ] Transfer learning
- [ ] Data cleansing
- [ ] Data augmentation
- [ ] Structural changes in the neural net

1. Create yourself a dataset (from Open Images Dataset V7)
2. Use Keras library
3. Topics covered
4. Image Processing (Data augmentation)
5. Data cleansing
6. Transfer Learning
7. Design own convolutional layers
8. Answer questions about the dataset


## Training model

1. Load the data: You need to load the images from the respective train, validation, and test directories into memory. You can use a library like OpenCV or PIL to load the images. It's common to convert the images to a standardized size, such as 224x224 pixels, to ensure consistency during training.

2. Preprocess the data: Preprocessing steps may include resizing the images, normalizing pixel values, and applying any other necessary transformations. For instance, you can normalize the pixel values to a range of 0 to 1 by dividing them by 255. Additionally, you might consider applying data augmentation techniques like random rotations, flips, or zooms to increase the variability of your training data.

3. Prepare labels: Assign appropriate labels to each image in your dataset. Since you are classifying lions, tigers, and cheetahs, you can assign labels like 0 for lions, 1 for tigers, and 2 for cheetahs. Ensure that the labels are consistent across your train, validation, and test sets.

4. Split the data: If you haven't already split your downloaded dataset into train, validation, and test sets, you can do so now. As mentioned earlier, a common split is 70% for training, 15% for validation, and 15% for testing. Ensure that the data is shuffled before splitting to avoid any bias.

5. Build your image classifier: Now that your data is prepared, you can proceed to build your image classifier. You have several options for creating a model, including using popular deep learning frameworks like TensorFlow or PyTorch. You can choose a pre-existing architecture such as VGG, ResNet, or MobileNet, and fine-tune it for your specific classification task. Alternatively, you can build your own custom architecture.

6. Train your model: Feed your prepared training data into the model and train it. Adjust hyperparameters like learning rate, batch size, and number of epochs to achieve the desired performance. Monitor the model's performance on the validation set to avoid overfitting.

7. Evaluate the model: After training, evaluate the performance of your model using the test set. Calculate metrics like accuracy, precision, recall, and F1 score to assess the model's classification performance.

8. Fine-tune and iterate: If your model's performance is not satisfactory, you can try fine-tuning the hyperparameters, adjusting the architecture, or exploring different augmentation techniques to improve the results. Iterate on the training process until you achieve the desired performance.

