# Implementation of Softmax

import numpy as np

# Mock activation at 5 different nodes in the output layer to do softmax on them
activation = np.random.randn(5)

# 1st Step - Exponentiate to make all positive

expVector = np.exp(activation)
answerVector = expVector / expVector.sum()

print answerVector.sum()

# The value of answerVector.sum() should converge to 1

# Let's try with a matrix of 100 samples and 5 classes

activationMatrix = np.random.randn(100, 5)
expActvationMatrix = np.exp(activationMatrix)

# Sum along the rows
answerA = expActvationMatrix / expActvationMatrix.sum(axis = 1, keepdims = True)

# Check sum along the rows converges to 1

print answerA.sum(axis = 1)

# Check the shape

print expActvationMatrix.sum(axis = 1, keepdims = True).shape
