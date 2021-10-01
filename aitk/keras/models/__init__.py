# -*- coding: utf-8 -*-
# **************************************************************
# aitk.keras: A Python Keras model API
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.keras
#
# **************************************************************

from ..layers import Input, Activation, Concatenate
from ..losses import MeanSquaredError, CrossEntropy
from ..initializers import OptimizerInitializer
from ..callbacks import History
from ..utils import topological_sort

import numpy as np
import math
import numbers
import functools
import operator

LOSS_FUNCTIONS = {
    "mse": MeanSquaredError,
    "crossentropy": CrossEntropy,
    # FIXME: add more error functions
}

NAME_CACHE = {}

class Model():
    def __init__(self, inputs=None, outputs=None, name=None):
        self.sequential = False
        self.history = History()
        self.name = self.make_name(name)
        self.layers = []
        self.layer_map = {}
        self._input_layers = None
        self._output_layers = None
        self.step = 0
        # Build a model graph from inputs to outputs:
        if inputs is not None and outputs is not None:
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            queue = [] if inputs is None else inputs
            if not isinstance(queue, (list, tuple)):
                queue = [queue]
            while len(queue) > 0:
                layer = queue.pop(0)
                if layer not in self.layers:
                    if layer.name in self.layer_map:
                        raise AttributeError("duplicate layer name: '%s'" % layer.name)
                    self.layers.append(layer)
                    self.layer_map[layer.name] = layer
                if layer in outputs:
                    # Make sure no more layers:
                    layer.output_layers = []
                else:
                    queue.extend(layer.output_layers)
            self.sequential = self.is_sequential()

    def is_sequential(self):
        return ((len(self.get_input_layers()) == 1) and
                (len(self.get_output_layers()) == 1) and
                (not any([isinstance(layer, Concatenate)
                          for layer in self.layers])))

    def get_input_layers(self):
        if self._input_layers is None:
            return [layer for layer in self.layers if len(layer.input_layers) == 0]
        else:
            return self._input_layers

    def get_output_layers(self):
        if self._output_layers is None:
            return [layer for layer in self.layers if len(layer.output_layers) == 0]
        else:
            return self._output_layers

    def connect(self, in_layer, out_layer):
        """
        Connect first layer to second layer.
        """
        if in_layer not in out_layer.input_layers:
            out_layer.input_layers.append(in_layer)
        if out_layer not in in_layer.output_layers:
            in_layer.output_layers.append(out_layer)

    def make_name(self, name):
        if name is None:
            class_name = self.__class__.__name__.lower()
            count = NAME_CACHE.get(class_name, 0)
            if count == 0:
                new_name = class_name
            else:
                new_name = "%s_%s" % (class_name, count)
            NAME_CACHE[class_name] = count + 1
            return new_name
        else:
            return name

    def summary(self):
        if self._input_layers is None:
            print(f'Model: "{self.name}" (uncompiled)')
        else:
            print(f'Model: "{self.name}"')
        print('_' * 65)
        print("Layer (type)                 Output Shape              Param #")
        print("=" * 65)
        total_parameters = 0
        # FIXME: sum up other, non-trainable params
        other_params = 0
        for i, layer in enumerate(topological_sort(self.get_input_layers())):
            layer_name = ("%s (%s)" % (layer.name, layer.__class__.__name__))[:25]
            parameters = sum([np.prod(item.shape) for item in layer.parameters.values() if item is not None])
            total_parameters += parameters
            output_shape = (None, layer.n_out) if isinstance(layer.n_out, numbers.Number) else layer.n_out
            print(f"{layer_name:25s} {str(output_shape)[:15]:>15s} {parameters:>20,}")
            if i != len(self.layers) - 1:
                print("_" * 65)
        print("=" * 65)
        print(f"Total params: {total_parameters:,}")
        print(f"Trainable params: {total_parameters + other_params:,}")
        print(f"Non-trainable params: {other_params:,}")
        print("_" * 65)

    def compile(self, optimizer, loss, metrics=None):
        self._input_layers = [layer for layer in self.layers if len(layer.input_layers) == 0]
        self._output_layers = [layer for layer in self.layers if len(layer.output_layers) == 0]
        for layer in self.layers:
            if not isinstance(layer, Input):
                layer.optimizer = OptimizerInitializer(optimizer)()
                loss_function = LOSS_FUNCTIONS[loss]
                self.loss_function = loss_function()
        self.metrics = metrics if metrics is not None else []
        # now, let's force the layers to initialize:
        inputs = self.build_inputs()
        self.predict(inputs)

    def get_layer_output_shape(self, layer, n=1):
        """
        Get the shape of the layer with a dataset
        size of n.
        """
        if isinstance(layer.n_out, numbers.Number):
            shape = (n, layer.n_out)
        else:
            shape = tuple([n] + list(layer.n_out))
        return shape

    def get_layer_output_array(self, layer):
        """
        Get an output array of a layer (dataset, n = 1).
        """
        shape = self.get_layer_output_shape(layer)
        output = np.ndarray(shape)
        return output

    def build_inputs(self):
        if self.sequential:
            inputs = self.get_layer_output_array(self.layers[0])
        else:
            if len(self.get_input_layers()) > 1:
                inputs = [self.get_layer_output_array(input)
                          for input in self._input_layers]
            else:
                inputs = self.get_layer_output_array(self._input_layers[0])
        return inputs

    def get_weights(self, flat=False):
        array = []
        if flat:
            for layer in self.layers:
                if layer.has_trainable_params():
                    for weight in layer.get_weights():
                        if isinstance(weight, numbers.Number):
                            array.extend(weight)
                        else:
                            array.extend(weight.flatten())
        else:
            for layer in self.layers:
                if layer.has_trainable_params():
                    array.extend(layer.get_weights())
        return array

    def copy_weights(self, model):
        """
        Copy the weights from another model by layer name.
        """
        for layer in model.layers:
            weights = layer.get_weights()
            self.layer_map[layer.name].set_weights(weights)

    def get_weights_by_name(self):
        """
        Copy the weights from another model by layer name.
        """
        return {layer.name: layer.get_weights() for layer in self.layers}

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
                if layer.has_trainable_params():
                    orig = layer.get_weights()
                    new_weights = []
                    for item in orig:
                        if isinstance(item, numbers.Number):
                            total = 1
                            new_weights.append(item)
                        else:
                            total = functools.reduce(operator.mul, item.shape, 1)
                            w = np.array(weights[current:current + total], dtype=float)
                            new_weights.append(w.reshape(item.shape))
                        current += total
                    layer.set_weights(new_weights)
        else:
            i = 0
            for layer in self.layers:
                if layer.has_trainable_params():
                    orig = layer.get_weights()
                    count = len(orig)
                    layer.set_weights(weights[i:i+count])
                    i += count

    def fit(self, inputs, targets, batch_size=32, epochs=1, verbose="auto", callbacks=None,
            shuffle=True):
        # FIXME: use shuffle
        verbose = 1 if verbose == "auto" else verbose
        callbacks = [] if callbacks is None else callbacks
        callbacks.append(self.history)
        inputs = np.array(inputs, dtype=float)
        targets = np.array(targets, dtype=float)
        self.flush_gradients()
        for callback in callbacks:
            callback.on_train_begin()
        for epoch in range(epochs):
            for metric in self.metrics:
                metric.reset_state()

            for callback in callbacks:
                callback.on_epoch_begin(epoch, {"step": self.step})
            loss = 0
            total_batches = math.ceil(self.get_length_of_inputs(inputs) / batch_size)
            for batch, length, batch_data in self.enumerate_batches(inputs, targets, batch_size):
                loss += self.train_batch(batch_data, batch, length, callbacks)
                self.step += length
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}")
                    print(f"{batch+1}/{total_batches} [==============================] - Xs time/step - loss: {loss}")
            logs = {
                "loss": loss,
                "step": self.step,
            }
            for metric in self.metrics:
                logs[metric.name] = metric.result()

            for callback in callbacks:
                callback.on_epoch_end(
                    epoch,
                    logs
                )
        for callback in callbacks:
            callback.on_train_end({"loss": loss})
        return self.history

    def flush_gradients(self):
        for layer in self.layers:
            if layer.has_trainable_params():
                layer.flush_gradients()

    def enumerate_batches(self, inputs, targets, batch_size):
        current_row = 0
        batch = 0
        while (current_row * batch_size) < self.get_length_of_inputs(inputs):
            batch_inputs = self.get_batch_inputs(
                inputs, current_row, batch_size)
            batch_targets = self.get_batch_targets(
                targets, current_row, batch_size)
            current_row += 1
            yield batch, self.get_length_of_inputs(batch_inputs), (batch_inputs, batch_targets)
            batch += 1

    def get_length_of_inputs(self, inputs):
        if len(self.get_input_layers()) == 1:
            return len(inputs)
        else:
            return len(inputs[0])

    def get_batch_inputs(self, inputs, current_row, batch_size):
        if len(self.get_input_layers()) == 1:
            return inputs[current_row:current_row + batch_size]
        else:
            return [np.array(inputs[i][current_row:current_row + batch_size])
                    for i in range(len(self.get_input_layers()))]

    def get_batch_targets(self, targets, current_row, batch_size):
        if not isinstance(targets, (list, tuple)):
            # Numpy, one bank:
            return targets[current_row:current_row + batch_size]
        else:
            return [np.array(targets[i][current_row:current_row + batch_size])
                    for i in range(len(self.get_output_layers()))]

    def train_batch(self, dataset, batch, length, callbacks):
        inputs, targets = dataset
        # Use predict to forward the activations, saving
        # needed information:
        outputs = self.predict(inputs, True)
        # Compute the derivative with respect
        # to this batch of the dataset:
        batch_loss = 0
        for callback in callbacks:
            callback.on_train_batch_begin(batch)
        if self.sequential:
            dY_pred = self.loss_function.grad(
                targets,
                outputs,
            )
            queue = [(self.get_output_layers()[0], dY_pred)]
            while len(queue) > 0:
                layer, dY_pred = queue.pop(0)
                if not isinstance(layer, Input):
                    dY_pred = layer.backward(dY_pred)
                    for input_layer in layer.input_layers:
                        queue.append((input_layer, dY_pred))

            batch_loss = self.loss_function(targets, outputs)
            for metric in self.metrics:
                metric.update_state(targets, outputs)
        else:
            for out_n in range(len(self.get_output_layers())):
                dY_pred = self.loss_function.grad(
                    targets[out_n],
                    outputs[out_n],
                )
                queue = [(self.get_output_layers()[out_n], dY_pred)]
                while len(queue) > 0:
                    layer, dY_pred = queue.pop(0)
                    if not isinstance(layer, Input):
                        dY_pred = layer.backward(dY_pred)
                        for input_layer in layer.input_layers:
                            queue.append((input_layer, dY_pred))

                batch_loss += self.loss_function(targets[out_n], outputs[out_n])
                for metric in self.metrics:
                    metric.update_state(targets[out_n], outputs[out_n])

        for callback in callbacks:
            callback.on_train_batch_end(batch, {"batch_loss": batch_loss})
        self.update(batch_loss)
        return batch_loss

    def update(self, batch_loss):
        for layer in self.layers:
            if not isinstance(layer, Input):
                layer.update(batch_loss)
        self.flush_gradients()

    def predict(self, inputs, retain_derived=False):
        inputs = np.array(inputs, dtype=float)
        results = []
        # First, load the inputs:
        if self.sequential:
            cache = {self._input_layers[0].name: inputs}
        else:
            if len(self._input_layers) > 1:
                cache = {self._input_layers[i].name: input for i, input in enumerate(inputs)}
            else:
                cache = {self._input_layers[0].name: inputs}

        # Propagate in topological order:
        for layer in topological_sort(self.get_input_layers()):
            if not isinstance(layer, Input):
                inputs = [cache[in_layer.name] for in_layer in layer.input_layers]
                if len(inputs) == 1:
                    cache[layer.name] = layer.forward(inputs[0], retain_derived=retain_derived)
                else:
                    cache[layer.name] = layer.forward(inputs, retain_derived=retain_derived)

        for layer in self.get_output_layers():
            results.append(cache[layer.name])
        if self.sequential:
            return results[0]
        else:
            return results

class Sequential(Model):
    def __init__(self, layers=None, name="sequential"):
        super().__init__(name=name)
        self.sequential = True
        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        if layer.name in self.layer_map:
            raise AttributeError("duplicate layer name: '%s'" % layer.name)
        self.layer_map[layer.name] = layer
        if len(self.layers) == 0:
            if isinstance(layer, Input):
                self.layers.append(layer)
            else:
                input_layer = Input(input_shape=layer.input_shape)
                self.connect(input_layer, layer)
                self.layers.append(input_layer)
                self.layers.append(layer)
        elif isinstance(layer, Activation):
            self.layers[-1].act_fn = layer.activation
        else:
            input_layer = self.layers[-1]
            self.connect(input_layer, layer)
            self.layers.append(layer)
