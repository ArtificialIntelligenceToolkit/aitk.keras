import numpy as np
import numbers

# XOR:
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [[0], [1], [1], [0]]

def build_model_tf(optimizer, loss):
    from tensorflow.keras.layers import InputLayer, Dense, Activation
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import SGD

    model = Sequential()
    model.add(InputLayer(2, name="input"))
    model.add(Dense(8, activation="tanh", name="hidden"))
    model.add(Dense(1, activation="sigmoid", name="output"))
    #model.compile(optimizer=SGD(lr=0.01), loss="mse")
    model.compile(optimizer=optimizer, loss=loss)
    return model

def build_model_aitk(optimizer, loss):
    from aitk.keras.layers import InputLayer, Dense, Activation
    from aitk.keras.models import Sequential
    #from aitk.keras.optimizers import SGD

    model = Sequential()
    model.add(InputLayer(2, name="input"))
    model.add(Dense(8, activation="tanh", name="hidden"))
    model.add(Dense(1, activation="sigmoid", name="output"))
    #model.compile(optimizer="sgd(lr=0.01)", loss="mse")
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
        print("epoch", optimizer, loss, i * epochs)
        for j, (j1, j2) in enumerate(zip(outputs_tf, outputs_aitk)):
            print(i, j, j1, j2)
            assert abs(j1 - j2) < 0.01, "outputs are too different"

        model_tf.fit(inputs, targets, epochs=epochs, verbose=0)
        model_aitk.fit(inputs, targets, epochs=epochs)

def test_fit():
    for optimizer in ["adam", "rmsprop", "sgd"]:
        for loss in ["mse"]:
            compare_models(optimizer, loss)
