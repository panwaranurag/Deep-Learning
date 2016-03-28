# Deep Neural Network
Project part of deep learning

nn.py
In this project, we are trying to develop deep neural network from screatch using user defined number of hidden layers, number of neurons and batch size for schostic gradient descent. This project can be use by anyone who want to build model on deep neural network. This implement is very flexible according to user need.

Designed deep neural network from scratch
- Input file: hw2_data4.txt
- Number of attributes/input neurons = 3 (Program calculate automatocally based on input file you provide)
- Number of output neurons = 2 (Classification problem with 2 class labels)
- Hidden layer: User defined on runtime
- Number of neurons in each hidden layer: User defined as an argument
* Developed feed-forward and back-propagation from scratch
* Implemented Dropout to overcome problem of overfitting
* Investigated performance of sigmoid, relu and tanh as activation function (Perceptrons)
* Implemented stochastic gradient descent for faster training
* Applied centralization and normalization to training data

CNN (Convolutional Neural Network)

* Implemented convolutional neural networks (CNN) for MNIST dataset 
* Implemented CNN with 2 convolutional layers and two Multi-Layer Perceptrons (MLP) layers 
* Used 10 and 20 receptive fields for 1st and 2nd convolutional layers 
* Implemented Softmax at the output layer 
* Max-pooling
* Achieved accuracy of 99.65% on testing dataset
