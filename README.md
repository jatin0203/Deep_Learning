# Fashion MNIST Classification with Feedforward Neural Network using wandb Tool
This repository contains an implementation of a feedforward neural network that can classify the Fashion MNIST dataset with the addition of different optimizer algorithms implemented from scratch. We conducted extensive research using the wandb library to arrive at the best configuration for classifying the dataset, even for mnist dataset.
# Description
The Fashion MNIST dataset contains 70,000 grayscale images,60,000 for training and 10,000 for testing of 10 fashion categories, each consisting of 28x28 pixels. We have used 10% of the training data as validation data for hyper parameter tuninig. Our implementation uses a feedforward neural network with multiple hidden layers to classify the images. We also flattened the data to give as input to the neural network and also normalised the data before using it.
# Code Files
1. The neuralNetwork.py files contains the implementation of the neural network with forward propogation and backward propogation code, initializers, loss functions, activation functions, derivative of loss, activation functions every parameter of neural network is set as a parameter of the object of the neural network class<br>
2. The opitimizer.py files contains all the optimizers which works on the neural network, the function optimize in optimizer class takes as input an object of neuralNetwork class, you can easily add any other optimizer that you want by just adding the function of that optimizer and adding the update rule used, everything else in the neural network class remains same.<br>
3. The train.py is script file where inputs can be passed with format mentioned in the question.<br>
4. The Q1.py file contains the code used to print one image of all the classes of image present in the dataset.<br>
# Usage
1. We have provided a train.py script file which can also take arguments with format as mentioned in the question, you can run the train.py file and the wandb logs will be generated for 1 sweep as the count is set to 1, and also a confusion matrix will be generated for the test dataset.<br>
2. To create the sweeps in wandb and check for the best configuration you can rum the main.py file.<br>
3. The sweeps generated to compare MSE and cross entropy loss code is in mse.py file<br>
