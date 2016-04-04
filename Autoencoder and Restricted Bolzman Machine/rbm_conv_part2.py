from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import datasets, cross_validation, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sknn.mlp import Classifier, Convolution, Layer

#Download MNIST dataset from Sklearn
mnist = datasets.fetch_mldata('MNIST original')
X, Y = mnist.data, mnist.target
X = np.asarray( X, 'float32')
# Scaling between 0 and 1
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
# Convert to binary images
X = X > 0.5
print(X.shape)
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split( X, Y, test_size=0.2, random_state=0)

#Bernoaulli based RBM with number of units are 400(Conpresssed features get from 786 features)
rbm = BernoulliRBM(n_components=400, learning_rate=0.01, batch_size=10, n_iter=10, 
verbose=True, random_state=None)

#Create a neural network that uses convolution to scan the input images with two fully connected layer
nn = Classifier(
    layers=[
        Convolution('Tanh', channels=20, kernel_shape=(5, 5), border_mode='valid'),
        Layer('Sigmoid', units=100),
        Layer('Softmax')],
    learning_rate=0.002,
    valid_size=0.2,
    n_stable=5,
    verbose=True)
    
classifier = Pipeline(steps=[('rbm', rbm), ('cnn', nn)])

###############################################################################
# Training RBM-CNN Pipeline
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print('Score:  ',(metrics.classification_report(Y_test, Y_pred)))



