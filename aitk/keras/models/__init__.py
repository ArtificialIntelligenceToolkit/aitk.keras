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

NAME_CACHE = {}

class Model():
    def __init__(self, inputs=None, outputs=None, name=None, layers=None):
        self.name = self.make_name(name)
        self._layers = layers if layers is not None else []
        self.train = True
        self.step = 0
        if inputs is not None:
            while True:
                self.add(inputs)
                if inputs == outputs:
                    break
                # FIXME: using only one input; need to make graph
                inputs = inputs.output_layers[0]

    def make_name(self, name):
        if name is None:
            class_name = self.__class__.__name__.lower()
            count = NAME_CACHE.get(class_name, 0)
            new_name = "%s_%s" % (class_name, count + 1)
            NAME_CACHE[class_name] = count + 1
            return new_name
        else:
            return name

    def summary(self):
        print(f'Model: "{self.name}"')
        print('_' * 65)
        print("Layer (type)                 Output Shape              Param #")
        print("=" * 65)
        total_parameters = 0
        # FIXME: sum up other, non-trainable params
        other_params = 0
        for i, layer in enumerate(self.layers):
            layer_name = "%s (%s)" % (layer.name, layer.__class__.__name__)
            parameters = sum([np.prod(item.shape) for item in layer.parameters.values()])
            total_parameters += parameters
            output_shape = (None, layer.n_out) if isinstance(layer.n_out, numbers.Number) else layer.n_out
            print(f"{layer_name:20s} {str(output_shape):>20s} {parameters:>20}")
            if i != len(self.layers) - 1:
                print("_" * 65)
        print("=" * 65)
        print(f"Total params: {total_parameters}")
        print(f"Trainable params: {total_parameters + other_params}")
        print(f"Non-trainable params: {other_params}")
        print("_" * 65)

    def compile(self, optimizer, loss):
        for layer in self.layers:
            layer.optimizer = OptimizerInitializer(optimizer)()
        loss_function = LOSS_FUNCTIONS[loss]
        self.loss_function = loss_function()
        # now, let's force the layers to initialize:
        if isinstance(self._layers[0].n_out, numbers.Number):
            shape = (1, self._layers[0].n_out)
        else:
            shape = tuple([1] + list(self._layers[0].n_out))
        inputs = np.ndarray(shape)
        self.predict(inputs)

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
            # Flat
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
            i = 0
            for layer in self.layers:
                orig = layer.get_weights()
                count = len(orig)
                layer.set_weights(weights[i:i+count])
                i += count

    def fit(self, inputs, targets, batch_size=32, epochs=1):
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
        current_row = 0
        while current_row < len(inputs):
            # if one input bank, one output bank:
            batch_inputs = inputs[current_row:current_row + batch_size]
            batch_targets = targets[current_row:current_row + batch_size]
            # FIXME: either may be composed of banks
            current_row += len(batch_inputs)
            yield (batch_inputs, batch_targets)

    def train_batch(self, dataset):
        inputs, targets = dataset
        # Use predict to forward the activations, saving
        # needed information:
        outputs = self.predict(inputs, True)

        # Compute the derivative with respect
        # to this batch of the dataset:
        dY_pred = self.loss_function.grad(
            targets,
            outputs,
        )

        for layer in reversed(self.layers):
            dY_pred = layer.backward(dY_pred)

        batch_loss = self.loss_function(targets, outputs)
        # FIXME: compute other metrics, and log them
        # Update every layer:
        # FIXME: scale this proportional?
        self.update(batch_loss)
        return batch_loss

    def update(self, batch_loss):
        for layer in reversed(self.layers):
            layer.update(batch_loss)
        self.flush_gradients()

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

class Sequential(Model):
    def __init__(self, layers=None, name="sequential"):
        super().__init__(name=name, layers=layers)
