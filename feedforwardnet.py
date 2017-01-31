# Feed Forward Net

# o -> o -> o
# o -> o -> o
#        -> o

import numpy as np
import matplotlib.pyplot as plt

numberSamplesPerClass = 500

gaussianCloud1 = np.random.randn(numberSamplesPerClass, 2) + np.array([0, -2])
gaussianCloud2 = np.random.randn(numberSamplesPerClass, 2) + np.array([2, 2])
gaussianCloud3 = np.random.randn(numberSamplesPerClass, 2) + np.array([-2, 2])
X = np.vstack([gaussianCloud1, gaussianCloud2, gaussianCloud3])

Y = np.array( [0] * numberSamplesPerClass + [1] * numberSamplesPerClass + [2] * numberSamplesPerClass)

plt.scatter(X[:, 0], X[:, 1], c = Y, s = 100, alpha = 0.5)
plt.show()

numberInputNodes = 2
hiddenLayerSize = 2
numberOutputClasses = 3

weight1 = np.random.randn(numberInputNodes, hiddenLayerSize)
bias1 = np.random.randn(hiddenLayerSize)

weight2 = np.random.randn(hiddenLayerSize, numberOutputClasses)
bias2 = np.random.randn(numberOutputClasses)

def feedforward(X, weight1, bias1, weight2, bias2):
    # use sigmoid in hidden layer
    Z = 1 / (1 + np.exp(-X.dot(weight1) - bias1))
    # Calculate softmax of next layer
    activation = Z.dot(weight2) + bias2
    expActivation = np.exp(activation)
    output = expActivation / expActivation.sum(axis = 1, keepdims = True)
    return output

def classification_rate(output, prediction):
    numCorrect = 0
    numTotal = 0
    for i in xrange(len(output)):
        numTotal += 1
        if output[i] == prediction[i]:
            numCorrect += 1
    return float(numCorrect) / numTotal

# Calculate forward function with randomly generated weights
probability_Y_given_X = feedforward(X, weight1, bias1, weight2, bias2)
predictions = np.argmax(probability_Y_given_X, axis = 1)
assert(len(predictions) == len(Y))

print "Without training, the classification rate is : ", classification_rate(Y, predictions)
