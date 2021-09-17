from ..layers import InputLayer, Activation

import numpy as np
import numbers
import functools
import operator

class Model():
    def __init__(self, name=None, layers=None):
        self.name = name
        self._layers = layers if layers is not None else []

    def compile(self):
        pass

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
        elif isinstance(layer, Activation):
            # FIXME:
            self._layers[-1].act_fn = laer.activation
        else:
            layer.add_input_layer(self._layers[-1])
            self._layers.append(layer)

    def predict(self, inputs):
        # FIXME: what type should inputs be?
        inputs = np.array(inputs, dtype=float)
        for layer in self._layers:
            outputs = inputs = layer.forward(inputs)
        return outputs
