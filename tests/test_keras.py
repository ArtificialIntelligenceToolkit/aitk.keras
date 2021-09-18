import numpy as np
import numbers

# XOR:
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [[0], [1], [1], [0]]

def build_model_tf():
    from tensorflow.keras.layers import InputLayer, Dense, Activation
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(InputLayer(2, name="input"))
    model.add(Dense(2, activation="relu", name="hidden"))
    model.add(Dense(1, activation="sigmoid", name="output"))
    model.compile(optimizer="adam", loss="mse")
    return model

def build_model_aitk():
    from aitk.keras.layers import InputLayer, Dense, Activation
    from aitk.keras.models import Sequential

    model = Sequential(optimizer="adam")
    model.add(InputLayer(2, name="input"))
    model.add(Dense(2, activation="relu", name="hidden"))
    model.add(Dense(1, activation="sigmoid", name="output"))
    model.compile(optimizer="adam", loss="mse")
    return model

def test_predict():
    model_aitk = build_model_aitk()
    outputs = model_aitk.predict(inputs)

    assert outputs.shape == np.array(targets).shape

def test_weights():
    model_tf = build_model_tf()
    tf_weights = model_tf.get_weights()

    model_aitk = build_model_aitk()
    model_aitk.set_weights(tf_weights)
    aitk_weights = model_aitk.get_weights()

    for w1, w2 in zip(tf_weights, aitk_weights):
        for v1, v2 in zip(w1, w2):
            if isinstance(v1, numbers.Number):
                assert (v1 - v2) < 0.1, "weights are too different"
            else:
                for j1, j2 in zip(v1, v2):
                    assert (j1 - j2) < 0.1, "weights are too different"

def test_fit():
    model_tf = build_model_tf()
    tf_weights = model_tf.get_weights()

    model_aitk = build_model_aitk()
    model_aitk.set_weights(tf_weights)

    outputs_tf = model_tf.predict([[1, 1]])
    outputs_aitk = model_aitk.predict([[1, 1]])

    for j1, j2 in zip(outputs_tf, outputs_aitk):
        assert (j1 - j2) < 0.1, "outputs are too different"
