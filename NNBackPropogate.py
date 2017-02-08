from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # Seed the random # generator, so it generates the same numbers every time the program execs

        random.seed(1)

        # Model 1 Neuron with 3 Inputs & 1 Output
        # Assign random weights to a 3 x 1 matrix with values in the range -1 to 1 and mean of 0
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # The sigmoid function, which describes an S shaped curve
    # We pass the weighted sum of the inputs through this function to normalize them between 0 & 1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Gradient of Sigmoid
    def __sigmoid_derivative(self, x):
        return x * (1-x)

    # Pass our inputs through the neuron to get the output
    def predict(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

    # Training with BackPropogation
    def train(self, trainingSetInputs, trainingSetOutputs, numberTrainingIterations):
        for iteration in xrange(numberTrainingIterations):
            # Pass the training set through our NN
            output = self.predict(trainingSetInputs)

            # Calculate the Error, difference between desired output & predicted output
            error = trainingSetOutputs - output

            # We want to iteratively minimize our weights
            # Computing Dot product
            # Input transposed & error x gradient of the sigmoid curve (Less confident weights are adjusted more)
            # = > Gradient Descent
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # adjust the weights
            self.synaptic_weights += adjustment



if __name__ == '__main__':

    # Initialize a single neuron neural network
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights:"
    print neural_network.synaptic_weights

    # Our Training Set - We have 4 examples.
    # Each consists of 3 input values & 1 ouput value

    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the NN using a training Set
    # Do it 10, 000 times and make small adjustments each time

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New Synaptic Weights are Training:"
    print neural_network.synaptic_weights

    # Test the NN with a new input

    print "Considering new input [1, 0, 0] -> ?: "
    print neural_network.predict(array([1, 0, 0]))
