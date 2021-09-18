from aitk.keras.layers import InputLayer, Activation
from aitk.keras.losses import MeanSquaredError, CrossEntropy
from aitk.keras.initializers import OptimizerInitializer

import numpy as np
import numbers
import functools
import operator

LOSS_FUNCTIONS = {
    "mse": MeanSquaredError,
    "crossentropy": CrossEntropy,
}

class Model():
    def __init__(self, name=None, layers=None):
        self.name = name
        self._layers = layers if layers is not None else []
        self.train = True
        self.step = 0

    def compile(self, optimizer, loss):
        for layer in self.layers:
            layer.optimizer = OptimizerInitializer(optimizer)()
        loss_function = LOSS_FUNCTIONS[loss]
        self.loss_function = loss_function()

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

    def fit(self, inputs, targets, batch_size=None, epochs=1):
        inputs = np.array(inputs, dtype=float)
        targets = np.array(targets, dtype=float)
        self.flush_gradients()
        for epoch in range(epochs):
            batch_loss = 0
            for batch_data in self.enumerate_batches(inputs, targets, batch_size):
                batch_loss += self.train_batch(batch_data)
                self.step += 1
            #print(self.step, batch_loss)

    def flush_gradients(self):
        for layer in self.layers:
            layer.flush_gradients()

    def enumerate_batches(self, inputs, targets, batch_size):
        # FIXME: break into batches
        yield (inputs, targets)

    def train_batch(self, dataset):
        inputs, targets = dataset
        outputs = self.predict(inputs, True)

        dY_pred = self.loss_function.grad(
            targets,
            outputs,
        )

        for layer in reversed(self.layers):
            dY_pred = layer.backward(dY_pred)

        batch_loss = self.loss_function(targets, outputs)
        # Update every batch:
        self.update(batch_loss)
        return batch_loss

    def update(self, batch_loss):
        for layer in reversed(self.layers):
            layer.update(batch_loss)
        self.flush_gradients()


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

    def predict(self, inputs, retain_derived=False):
        inputs = np.array(inputs, dtype=float)
        for layer in self._layers:
            outputs = inputs = layer.forward(inputs, retain_derived=retain_derived)
        return outputs
