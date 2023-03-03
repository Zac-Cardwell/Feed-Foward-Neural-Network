import numpy as np
import shells

# Class for creating a fully connected layer, works by creating a mxn matrix of weights with
# m being the input size and n being the output size the dot product between the matrix and input is found and
# the result is given as the output
# For backwards pass, the error is given to the function and the weights and biases are
# changed accordingly to their derivatives


class FCLayer(shells.layer):

    def __init__(self, input_size, output_size, activation_func):
        self.output_size = output_size
        self.input_size = input_size
        self.act_func = activation_func

        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forwardPass(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.bias
        return self.act_func.run(self.outputs)

    def backwardPass(self, error, lr):
        error = self.act_func.derv(error)
        dx = np.dot(error, self.weights.T)
        dw = np.dot(self.inputs.T, error)
        self.weights -= dw * lr
        self.bias -= error * lr
        return dx
