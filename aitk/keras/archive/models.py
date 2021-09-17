import numpy as np
import operator
import numbers
import functools

from .activations import ACTIVATIONS
from .layers import InputLayer

class Model():
    def __init__(self, name=None, layers=None):
        self.name = name
        self.compiled = False
        self._layers = layers if layers is not None else []
        self.train = False

    @property
    def layers(self):
        if len(self._layers) > 0:
            return self._layers[1:]
        else:
            return []

    def get_weights(self, flat=False):
        array = []
        if flat:
            for layer in self.layers:
                for weight in layer.get_weights():
                    array.extend(weight.flatten())
        else:
            for layer in self.layers:
                array.extend(layer.get_weights())
        return array

    def set_weights(self, weights):
        """
        Set the weights in a network.

        Args:
            weights: a list of pairs of weights and biases for each layer,
                or a single (flat) array of values
        """
        if len(weights) > 0 and isinstance(weights[0], numbers.Number):
            current = 0
            for layer in self.layers:
                orig = layer.get_weights()
                new_weights = []
                for item in orig:
                    total = functools.reduce(operator.mul, item.shape, 1)
                    w = np.array(weights[current:current + total], dtype=float)
                    new_weights.append(w.reshape(item.shape))
                    current += total
                layer.set_weights(new_weights)
        else:
            # FIXME: assumes weights, biases
            i = 0
            for layer in self.layers:
                layer.set_weights([weights[i], weights[i+1]])
                i += 2

class Sequential(Model):
    def __init__(self, layers=None, name=None):
        super().__init__(name=name, layers=layers)
        
    def add(self, layer):
        if len(self._layers) == 0:
            if isinstance(layer, InputLayer):
                self._layers.append(layer)
            else:
                input_layer = InputLayer(input_shape=layer.input_shape)
                self._layers.append(input_layer)
                self._layers.append(layer)
                layer.add_input_layer(input_layer)
        else:
            layer.add_input_layer(self._layers[-1])
            self._layers.append(layer)

    def predict(self, inputs):
        if not self.compiled:
            self.compile()
        for layer in self._layers:
            if self.train:
                layer.previous_inputs = inputs
            outputs = inputs = layer(inputs)
            if self.train:
                layer.previous_outputs = outputs
        return outputs

    def compile(self):
        for layer in self._layers:
            layer.compile()
        self.compiled = True

    def fit(self, inputs, targets, batch_size=None, epochs=1):
        self.step = 0
        self.train = True # signal for predict to save intermediate values
        for epoch in range(epochs):
            for batch_data in self.enumerate_batches(inputs, targets, batch_size):
                self.train_batch(batch_data)
                self.step += 1
        self.train = False

    def enumerate_batches(self, inputs, targets, batch_size):
        # FIXME: break into batches
        yield (inputs, targets)

    def compute_loss(self, outputs, targets):
        # respect things like sme, etc
        return (targets - outputs)

    def train_batch(self, batch_data):
        inputs, targets = batch_data
        outputs = self.predict(inputs)
        loss = self.compute_loss(outputs, targets)
        self.update_weights(loss, len(inputs))

    def update_weights(self, loss, n):
        for i, layer in enumerate(reversed(self.layers)):
            layer.delta_weights = 1/n * (loss @ layer.previous_inputs.T)
            layer.delta_biases = np.mean(loss, axis=1, keepdims=True)
            dAl = layer.weights[0].T @ loss
            # Compute loss for next layer:
            loss = layer.activation.derivative(layer.previous_outputs) * dAl
