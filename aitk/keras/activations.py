import numpy as np

class Activation():
    def __init__(self):
        self.input_layers = []

    def add_input_layer(self, input_layer):
        self.input_layers.append(input_layer)

    def get_shape(self):
        return self.input_layers[0].get_shape()

    def compile(self):
        pass

class Relu(Activation):
    def __call__(self, outputs):
        return np.maximum(0, outputs)

    def derivative(self, X):
        return np.where(X > 0, 1, 0)

class Tanh(Activation):
    def __call__(self, outputs):
        return np.tanh(outputs)

class Sigmoid(Activation):
    def __call__(self, outputs):
        return 1/(1 + np.exp(-outputs))

class Linear(Activation):
    def __call__(self, outputs):
        return outputs

ACTIVATIONS = {
    "relu": Relu,
    "tanh": Tanh,
    "linear": Linear,
    "sigmoid": Sigmoid,
}
