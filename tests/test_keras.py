import numpy as np
import numbers

# XOR:
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
inputs1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
inputs2 = [np.array([[0], [0], [1], [1]]),
           np.array([[0], [1], [0], [1]])]
targets = [[0], [1], [1], [0]]
targets2 = [
    np.array([[0], [1], [1], [0]]),
    np.array([[0], [1], [1], [0]]),
]

def build_model_tf(optimizer="adam", loss="mse"):
    from tensorflow.keras.layers import InputLayer, Dense, Activation
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(InputLayer(2, name="input"))
    model.add(Dense(8, activation="tanh", name="hidden"))
    model.add(Dense(1, activation="sigmoid", name="output"))
    model.compile(optimizer=optimizer, loss=loss)
    return model

def build_model_aitk(optimizer="adam", loss="mse"):
    from aitk.keras.layers import InputLayer, Dense, Activation
    from aitk.keras.models import Sequential

    model = Sequential()
    model.add(InputLayer(2, name="input"))
    model.add(Dense(8, activation="tanh", name="hidden"))
    model.add(Dense(1, activation="sigmoid", name="output"))
    model.compile(optimizer=optimizer, loss=loss)
    return model

def build_model_aitk_functional(optimizer="adam", loss="mse"):
    from aitk.keras.layers import Input, Dense, Activation
    from aitk.keras.models import Model

    l1 = Input(2, name="input")
    l2 = Dense(8, activation="tanh", name="hidden")
    l3 = Dense(1, activation="sigmoid", name="output")

    output_layer = l3(l2(l1))
    input_layer = l1
    model = Model(input_layer, output_layer)
    model.compile(optimizer=optimizer, loss=loss)
    return model

def build_model_tf_functional(optimizer="adam", loss="mse"):
    from tensorflow.keras.layers import Input, Dense, Activation
    from tensorflow.keras.models import Model

    l1 = Input(2, name="input")
    l2 = Dense(8, activation="tanh", name="hidden")
    l3 = Dense(1, activation="sigmoid", name="output")

    output_layer = l3(l2(l1))
    input_layer = l1
    model = Model(input_layer, output_layer)
    model.compile(optimizer=optimizer, loss=loss)
    return model

def build_model_tf_multiple_inputs(optimizer="adam", loss="mse"):
    from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
    from tensorflow.keras.models import Model

    input1 = Input(1, name="input1")
    input2 = Input(1, name="input2")
    input_cat = Concatenate()([input1, input2])
    hidden = Dense(8, activation="tanh", name="hidden")(input_cat)
    output = Dense(1, activation="sigmoid", name="output1")(hidden)

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer=optimizer, loss=loss)
    return model

def build_model_tf_multiple_both(optimizer="adam", loss="mse"):
    from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
    from tensorflow.keras.models import Model

    input1 = Input(1, name="input1")
    input2 = Input(1, name="input2")
    input_cat = Concatenate()([input1, input2])
    hidden = Dense(8, activation="tanh", name="hidden")(input_cat)
    output1 = Dense(1, activation="sigmoid", name="output1")(hidden)
    output2 = Dense(1, activation="sigmoid", name="output2")(hidden)

    model = Model(inputs=[input1, input2], outputs=[output1, output2])
    model.compile(optimizer=optimizer, loss=loss)
    return model

def build_model_tf_multiple_outputs(optimizer="adam", loss="mse"):
    from tensorflow.keras.layers import Input, Dense, Activation
    from tensorflow.keras.models import Model

    input = Input(2, name="input")
    hidden = Dense(8, activation="tanh", name="hidden")(input)
    output1 = Dense(1, activation="sigmoid", name="output1")(hidden)
    output2 = Dense(1, activation="sigmoid", name="output2")(hidden)

    model = Model(inputs=input, outputs=[output1, output2])
    model.compile(optimizer=optimizer, loss=loss)
    return model

def build_model_aitk_multiple_outputs(optimizer="adam", loss="mse"):
    from aitk.keras.layers import Input, Dense, Activation
    from aitk.keras.models import Model

    input = Input(2, name="input")
    hidden = Dense(8, activation="tanh", name="hidden")(input)
    output1 = Dense(1, activation="sigmoid", name="output1")(hidden)
    output2 = Dense(1, activation="sigmoid", name="output2")(hidden)

    model = Model(inputs=input, outputs=[output1, output2])
    model.compile(optimizer=optimizer, loss=loss)
    return model

def test_predict_shape():
    model_aitk = build_model_aitk("adam", "mse")
    outputs = model_aitk.predict(inputs)

    assert outputs.shape == np.array(targets).shape

def test_weights():
    model_tf = build_model_tf("adam", "mse")
    tf_weights = model_tf.get_weights()

    model_aitk = build_model_aitk("adam", "mse")
    model_aitk.set_weights(tf_weights)
    aitk_weights = model_aitk.get_weights()

    for w1, w2 in zip(tf_weights, aitk_weights):
        for v1, v2 in zip(w1, w2):
            if isinstance(v1, numbers.Number):
                assert (v1 - v2) < 0.1, "weights are too different"
            else:
                for j1, j2 in zip(v1, v2):
                    assert abs(j1 - j2) < 0.01, "weights are too different"

def test_predict():
    model_tf = build_model_tf("adam", "mse")
    tf_weights = model_tf.get_weights()

    model_aitk = build_model_aitk("adam", "mse")
    model_aitk.set_weights(tf_weights)

    outputs_tf = model_tf.predict([[1, 1]])
    outputs_aitk = model_aitk.predict([[1, 1]])

    for j1, j2 in zip(outputs_tf, outputs_aitk):
        assert abs(j1 - j2) < 0.01, "outputs are too different"

def compare_models(optimizer, loss):
    model_tf = build_model_tf(optimizer, loss)
    tf_weights = model_tf.get_weights()

    model_aitk = build_model_aitk(optimizer, loss)
    model_aitk.set_weights(tf_weights)

    for i in range(10):
        outputs_tf = model_tf.predict(inputs)
        outputs_aitk = model_aitk.predict(inputs)

        epochs = 10
        #print("epoch", optimizer, loss, i * epochs)
        for j, (j1, j2) in enumerate(zip(outputs_tf, outputs_aitk)):
            #print(i, j, j1, j2)
            assert abs(j1 - j2) < 0.01, ("%s %s: outputs are too different" % (optimizer, loss))

        model_tf.fit(inputs, targets, epochs=epochs, verbose=0, shuffle=False)
        model_aitk.fit(inputs, targets, epochs=epochs, verbose=0, shuffle=False)

def compare_models_functional(optimizer, loss):
    model_tf = build_model_tf_functional(optimizer, loss)
    tf_weights = model_tf.get_weights()

    model_aitk = build_model_aitk_functional(optimizer, loss)
    model_aitk.set_weights(tf_weights)

    for i in range(10):
        outputs_tf = model_tf.predict(inputs)
        outputs_aitk = model_aitk.predict(inputs)

        epochs = 10
        #print("epoch", optimizer, loss, i * epochs)
        for j, (j1, j2) in enumerate(zip(outputs_tf, outputs_aitk)):
            #print(i, j, j1, j2)
            assert abs(j1 - j2) < 0.01, ("%s %s: outputs are too different" % (optimizer, loss))

        model_tf.fit(inputs, targets, epochs=epochs, verbose=0, shuffle=False)
        model_aitk.fit(inputs, targets, epochs=epochs, verbose=0, shuffle=False)

def test_fit():
    for optimizer in ["adam", "rmsprop"]: # sgd diverages; lr and momentum the same
        for loss in ["mse"]:
            compare_models(optimizer, loss)

def test_fit_functional():
    for optimizer in ["adam", "rmsprop"]: # sgd diverages; lr and momentum the same
        for loss in ["mse"]:
            compare_models_functional(optimizer, loss)

def test_fit_sgd():
    from aitk.keras.optimizers import SGD
    model_aitk = build_model_aitk(SGD(lr=.1, momentum=.9), "mse")
    model_aitk.fit(inputs, targets, epochs=50)
    outputs = model_aitk.predict(inputs)
    assert [round(v[0]) for v in outputs] == [0, 1, 1, 0]


def test_multiple_outputs():
    model_tf = build_model_tf_multiple_outputs("adam", "mse")
    tf_weights = model_tf.get_weights()

    model_aitk = build_model_aitk_multiple_outputs("adam", "mse")
    model_aitk.set_weights(tf_weights)

    for i in range(10):
        outputs_tf = model_tf.predict(inputs1)
        outputs_aitk = model_aitk.predict(inputs1)

        for out_tf, out_aitk in zip(outputs_tf, outputs_aitk):
            for row_tf, row_aitk in zip(out_tf, out_aitk):
                for item_tf, item_aitk in zip(row_tf, row_aitk):
                    assert abs(item_tf - item_aitk) < 0.1

        # Need to fix backward
        model_tf.fit(inputs1, targets2, epochs=1)
        model_aitk.fit(inputs1, targets2, epochs=1)

def build_topological_sort():
    from aitk.keras.layers import Input, Dense, Activation
    from aitk.keras.models import Model

    i1 = Input(1)
    i2 = Input(1)
    h1 = Dense(5)
    h2 = Dense(5)
    h3 = Dense(5)
    o1 = Dense(2)
    o2 = Dense(2)

    h4 = h1(i1)
    h5 = h2(i2)

    h6 = h3(h4)
    h7 = h3(h5)

    out1 = o1(h6)
    out1 = o1(i1) # shortcut
    out2 = o2(h7)

    model = Model(inputs=[i1, i2], outputs=[out1, out2])
    # FIXME: layers with multiple inputs need proper
    # initialized parameters.
    #model.compile(optimizer="adam", loss="mse") # fix forward
    return model

def test_topological_sort():
    model = build_topological_sort()
    assert len(model.layers) == 7
