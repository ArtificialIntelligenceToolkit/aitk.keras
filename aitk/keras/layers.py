import numpy as np

from .activations import ACTIVATIONS

class Layer():
    def __init__(self, name=None):
        self.name = name
        self.input_layers = []

    def add_input_layer(self, input_layer):
        self.input_layers.append(input_layer)
        
    def compile(self):
        pass

    def get_shape(self):
        return self.shape

    def get_activation(self, activation):
        if isinstance(activation, str):
            act_name = activation.lower()
            if act_name in ACTIVATIONS:
                return ACTIVATIONS[act_name]()
            else:
                raise AttributeError("no such activation function: '%s'" % act_name)
        else:
            return activation

class InputLayer(Layer):
    def __init__(self, input_shape=None, name=None):
        super().__init__(name=name)
        self.shape = input_shape

    def __call__(self, inputs):
        return inputs
        
class Dense(Layer):
    def __init__(self, shape, input_shape=None, activation=None, name=None):
        super().__init__(name=name)
        self.shape = shape
        self.input_shape = input_shape
        self.activation = self.get_activation(activation)
        self.weights = []
        self.biases = []

    def get_weights(self):
        if len(self.input_layers) == 1:
            return [self.weights[0], self.biases[0]]
        else:
            return [self.weights, self.biases]

    def set_weights(self, weights):
        if len(self.input_layers) == 1:
            self.weights = [weights[0]]
            self.biases = [weights[1]]
        else:
            self.weights = weights[0]
            self.biases = weights[1]

    def compile(self):
        self.weights.clear()
        self.biases.clear()
        for layer in self.input_layers:
            in_layer_shape = layer.get_shape()
            out_layer_shape = self.get_shape()
            # FIXME: are these too large?
            self.weights.append(1 - np.random.random_sample((in_layer_shape, out_layer_shape)) * 2)
            # In keras, biases are zero by default
            self.biases.append(np.zeros(out_layer_shape, dtype=float))

    def __call__(self, input_tensor):
        # FIXME: how to get input for each input_tensor
        for i in range(len(self.input_layers)):
            output = np.dot(input_tensor, self.weights[i]) + self.biases[i]
            if self.activation is not None:
                output = self.activation(output)
        return output

class BatchNormalization(Layer):
    def __init__(self):
        super().__init__()
        self.gamma = 1
        self.beta = 1
        self.mean = 1
        self.std = 1
        self.epsilon = 1

    def get_shape(self):
        return self.input_layers[0].get_shape()

    def compile(self):
        pass

    def __call__(self, input_tensor):
        x = (input_tensor - self.mean) / np.sqrt(self.std + self.epsilon)
        x = self.gamma * x + self.beta
        return x

def Activation(activation):
    return ACTIVATIONS[activation]()

