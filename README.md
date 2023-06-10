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

Tasks

- [ ] Preparation: Split dataset into a 80/20 Train/test split

- [ ] Transfer learning: Use a imagenet pretrained VGG19 architecture, train the model and estimate the testset accuracy 

- [ ] Data cleansing: Remove “bad” images from the dataset. Which did you remove? How many? Discuss results.

- [ ] Add data augmentation and train again, discuss results
        - [ ] Random flip

        - [ ] Random contrast

        - [ ] Random translation

- [ ] Rebuild VGG19. After layer block4_conv4 (32, 32, 512):

- [ ] Add a naive inception layer (output filter size should be 512, each padding same, activations leaky relu)

- [ ] Add conv layer (kernel 3x3,  filters 512, padding valid, stride 2, activation relu)

- [ ] Add conv layer (kernel 1x1, filters 640, padding valid, stride 1, activation relu)

- [ ] Freeze conv2 layers and before

- [ ] Test a few of your own images and present the results

- [ ] Answer the following questions:

    - [ ] What accuracy can be achieved? What is the accuracy of the train vs. test set?

    - [ ] On what infrastructure did you train it? What is the inference time?

    - [ ] What are the number of parameters of the model?

    - [ ] Which categories are most likely to be confused by the algorithm? Show results in a confusion matrix.



Compare the results of the experiments.