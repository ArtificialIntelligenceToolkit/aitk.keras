import numpy as np
import numbers

# XOR:
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
inputs1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
inputs2 = [np.array([[0], [0], [1], [1]]),
           np.array([[0], [1], [0], [1]])]
targets = [[0], [1], [1], [0]]
targets1 = [np.array([[0], [1], [1], [0]])]
targets2 = [
    np.array([[0], [1], [1], [0]]),
    np.array([[0], [1], [1], [0]]),
]

MAX_DIFF = 0.001

def build_model(framework, optimizer="adam", loss="mse", metrics=None):
    if framework == "tf":
        from tensorflow.keras.layers import InputLayer, Dense, Activation
        from tensorflow.keras.models import Sequential
    else:
        from aitk.keras.layers import InputLayer, Dense, Activation
        from aitk.keras.models import Sequential

    model = Sequential()
    model.add(InputLayer(2, name="input"))
    model.add(Dense(8, activation="tanh", name="hidden"))
    model.add(Dense(1, activation="sigmoid", name="output"))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def build_model_functional(framework, optimizer="adam", loss="mse"):
    if framework == "tf":
        from tensorflow.keras.layers import Input, Dense, Activation
        from tensorflow.keras.models import Model
    else:
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

def build_model_multiple_inputs(framework, optimizer="adam", loss="mse"):
    if framework == "tf":
        from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
        from tensorflow.keras.models import Model
    else:
        from aitk.keras.layers import Input, Dense, Activation, Concatenate
        from aitk.keras.models import Model

    input1 = Input(1, name="input1")
    input2 = Input(1, name="input2")
    input_cat = Concatenate()([input1, input2])
    hidden = Dense(8, activation="tanh", name="hidden")(input_cat)
    output = Dense(1, activation="sigmoid", name="output1")(hidden)

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer=optimizer, loss=loss)
    return model

def build_model_multiple_both(framework, optimizer="adam", loss="mse"):
    if framework == "tf":
        from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
        from tensorflow.keras.models import Model
    else:
        from aitk.keras.layers import Input, Dense, Activation, Concatenate
        from aitk.keras.models import Model

    input1 = Input(1, name="input1")
    input2 = Input(1, name="input2")
    input_cat = Concatenate()([input1, input2])
    hidden = Dense(8, activation="tanh", name="hidden")(input_cat)
    output1 = Dense(1, activation="sigmoid", name="output1")(hidden)
    output2 = Dense(1, activation="sigmoid", name="output2")(hidden)

    model = Model(inputs=[input1, input2], outputs=[output1, output2])
    model.compile(optimizer=optimizer, loss=loss)
    return model

def build_model_multiple_outputs(framework, optimizer="adam", loss="mse"):
    if framework == "tf":
        from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
        from tensorflow.keras.models import Model
    else:
        from aitk.keras.layers import Input, Dense, Activation, Concatenate
        from aitk.keras.models import Model

    input = Input(2, name="input")
    hidden = Dense(8, activation="tanh", name="hidden")(input)
    output1 = Dense(1, activation="sigmoid", name="output1")(hidden)
    output2 = Dense(1, activation="sigmoid", name="output2")(hidden)

    model = Model(inputs=input, outputs=[output1, output2])
    model.compile(optimizer=optimizer, loss=loss)
    return model

def build_model_multiple_hiddens(framework, optimizer="adam", loss="mse"):
    if framework == "tf":
        from tensorflow.keras.layers import Input, Dense, Concatenate
        from tensorflow.keras.models import Model
    else:
        from aitk.keras.layers import Input, Dense, Concatenate
        from aitk.keras.models import Model

    input = Input(2, name="input")
    hidden1 = Dense(4, activation="tanh", name="hidden1")(input)
    hidden2 = Dense(4, activation="tanh", name="hidden2")(input)
    concat = Concatenate()([hidden1, hidden2])
    output = Dense(1, activation="sigmoid", name="output")(concat)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss=loss)
    return model

def build_model_multiple_all(framework, optimizer="adam", loss="mse"):
    if framework == "tf":
        from tensorflow.keras.layers import Input, Dense, Concatenate
        from tensorflow.keras.models import Model
    else:
        from aitk.keras.layers import Input, Dense, Concatenate
        from aitk.keras.models import Model

    input1 = Input(1, name="input1")
    input2 = Input(1, name="input2")
    concat = Concatenate()([input1, input2])
    hidden1 = Dense(4, activation="tanh", name="hidden1")(concat)
    hidden2 = Dense(4, activation="tanh", name="hidden2")(concat)
    output1 = Dense(1, activation="sigmoid", name="output1")(hidden1)
    output2 = Dense(1, activation="sigmoid", name="output2")(hidden2)

    model = Model(inputs=[input1, input2], outputs=[output1, output2])
    model.compile(optimizer=optimizer, loss=loss)
    return model

def test_predict_shape():
    model_aitk = build_model("aitk", "adam", "mse")
    outputs = model_aitk.predict(inputs)

    assert outputs.shape == np.array(targets).shape

def test_weights():
    model_tf = build_model("tf", "adam", "mse")
    tf_weights = model_tf.get_weights()

    model_aitk = build_model("aitk", "adam", "mse")
    model_aitk.set_weights(tf_weights)
    aitk_weights = model_aitk.get_weights()

    compare_weights(model_tf, model_aitk)

    for w1, w2 in zip(tf_weights, aitk_weights):
        for v1, v2 in zip(w1, w2):
            if isinstance(v1, numbers.Number):
                assert (v1 - v2) < MAX_DIFF, "weights are too different"
            else:
                for j1, j2 in zip(v1, v2):
                    assert abs(j1 - j2) < MAX_DIFF, "weights are too different"

def test_predict():
    model_tf = build_model("tf", "adam", "mse")
    tf_weights = model_tf.get_weights()

    model_aitk = build_model("aitk", "adam", "mse")
    model_aitk.set_weights(tf_weights)

    compare_weights(model_tf, model_aitk)

    outputs_tf = model_tf.predict([[1, 1]])
    outputs_aitk = model_aitk.predict([[1, 1]])

    for j1, j2 in zip(outputs_tf, outputs_aitk):
        assert abs(j1 - j2) < MAX_DIFF, "outputs are too different"

def compare_models(optimizer, loss):
    model_tf = build_model("tf", optimizer, loss)
    tf_weights = model_tf.get_weights()

    model_aitk = build_model("aitk", optimizer, loss)
    model_aitk.set_weights(tf_weights)

    compare_weights(model_tf, model_aitk)

    for i in range(30):
        outputs_tf = model_tf.predict(inputs)
        outputs_aitk = model_aitk.predict(inputs)

        epochs = 10
        #print("epoch", optimizer, loss, i * epochs)
        for j, (j1, j2) in enumerate(zip(outputs_tf, outputs_aitk)):
            #print(i, j, j1, j2)
            assert abs(j1 - j2) < MAX_DIFF, ("%s %s: outputs are too different" % (optimizer, loss))

        model_tf.fit(inputs, targets, epochs=epochs, verbose=0, shuffle=False)
        model_aitk.fit(inputs, targets, epochs=epochs, verbose=0, shuffle=False)

def compare_weights(m1, m2):
    assert all([(w1 == w2).all()
                for w1, w2 in zip(m1.get_weights(), m2.get_weights())])

def compare_output(out1, out2):
    assert all([(w1 == w2).all() for w1, w2 in zip(out1, out2)])

def compare_models_functional(optimizer, loss):
    model_tf = build_model_functional("tf", optimizer, loss)
    tf_weights = model_tf.get_weights()

    model_aitk = build_model_functional("aitk", optimizer, loss)
    model_aitk.set_weights(tf_weights)

    compare_weights(model_tf, model_aitk)

    for i in range(30):
        outputs_tf = model_tf.predict(inputs)
        outputs_aitk = model_aitk.predict(inputs)

        epochs = 10
        for j, (j1, j2) in enumerate(zip(outputs_tf, outputs_aitk)):
            print(i, j, j1, j2)
            assert abs(j1 - j2) < MAX_DIFF, ("%s %s: outputs are too different" % (optimizer, loss))

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
    model_aitk = build_model("aitk", SGD(lr=.1, momentum=.9), "mse")
    model_aitk.fit(inputs, targets, epochs=50, shuffle=False)
    outputs = model_aitk.predict(inputs)
    assert [round(v[0]) for v in outputs] == [0, 1, 1, 0]


def test_multiple_outputs():
    model_tf = build_model_multiple_outputs("tf", "adam", "mse")
    tf_weights = model_tf.get_weights()

    model_aitk = build_model_multiple_outputs("aitk", "adam", "mse")
    model_aitk.set_weights(tf_weights)

    compare_weights(model_tf, model_aitk)

    for i in range(30):
        outputs_tf = model_tf.predict(inputs1)
        outputs_aitk = model_aitk.predict(inputs1)

        for out_tf, out_aitk in zip(outputs_tf, outputs_aitk):
            for row_tf, row_aitk in zip(out_tf, out_aitk):
                for item_tf, item_aitk in zip(row_tf, row_aitk):
                    assert abs(item_tf - item_aitk) < MAX_DIFF

        model_tf.fit(inputs1, targets2, epochs=1, shuffle=False)
        model_aitk.fit(inputs1, targets2, epochs=1, shuffle=False)

def test_multiple_inputs():
    model_tf = build_model_multiple_inputs("tf", "adam", "mse")
    tf_weights = model_tf.get_weights()

    model_aitk = build_model_multiple_inputs("aitk", "adam", "mse")
    model_aitk.set_weights(tf_weights)

    compare_weights(model_tf, model_aitk)

    for i in range(30):
        outputs_tf = model_tf.predict(inputs2)
        outputs_aitk = model_aitk.predict(inputs2)

        for out_tf, out_aitk in zip(outputs_tf, outputs_aitk):
            for item_tf, item_aitk in zip(out_tf, out_aitk):
                assert abs(item_tf - item_aitk) < MAX_DIFF

        model_tf.fit(inputs2, targets1, epochs=1, shuffle=False)
        model_aitk.fit(inputs2, targets1, epochs=1, shuffle=False)

def test_multiple_both():
    model_tf = build_model_multiple_both("tf", "adam", "mse")
    tf_weights = model_tf.get_weights()

    model_aitk = build_model_multiple_both("aitk", "adam", "mse")
    model_aitk.set_weights(tf_weights)

    compare_weights(model_tf, model_aitk)

    for i in range(30):
        outputs_tf = model_tf.predict(inputs2)
        outputs_aitk = model_aitk.predict(inputs2)

        for out_tf, out_aitk in zip(outputs_tf, outputs_aitk):
            for row_tf, row_aitk in zip(out_tf, out_aitk):
                for item_tf, item_aitk in zip(row_tf, row_aitk):
                    assert abs(item_tf - item_aitk) < MAX_DIFF

        model_tf.fit(inputs2, targets2, epochs=1, shuffle=False)
        model_aitk.fit(inputs2, targets2, epochs=1, shuffle=False)

def test_multiple_all():
    model_tf = build_model_multiple_all("tf", "adam", "mse")
    tf_weights = model_tf.get_weights()

    model_aitk = build_model_multiple_all("aitk", "adam", "mse")
    model_aitk.set_weights(tf_weights)

    compare_weights(model_tf, model_aitk)

    for i in range(30):
        outputs_tf = model_tf.predict(inputs2)
        outputs_aitk = model_aitk.predict(inputs2)

        for out_tf, out_aitk in zip(outputs_tf, outputs_aitk):
            for row_tf, row_aitk in zip(out_tf, out_aitk):
                for item_tf, item_aitk in zip(row_tf, row_aitk):
                    assert abs(item_tf - item_aitk) < MAX_DIFF

        model_tf.fit(inputs2, targets2, epochs=1, shuffle=False)
        model_aitk.fit(inputs2, targets2, epochs=1, shuffle=False)

def build_topological_sort(framework):
    if framework == "tf":
        from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
        from tensorflow.keras.models import Model
    else:
        from aitk.keras.layers import Input, Dense, Activation, Concatenate
        from aitk.keras.models import Model

    i1 = Input(1, name="input1")
    i2 = Input(1, name="input2")
    h1 = Dense(5, name="hidden1")
    h2 = Dense(5, name="hidden2")
    h3 = Dense(5, name="hidden3")
    o1 = Dense(1, name="output1")
    o2 = Dense(1, name="output2")

    h4 = h1(i1)
    h5 = h2(i2)

    h6 = h3(h4)

    concat1 = Concatenate()([h6, i2])
    out1 = o1(concat1)
    out2 = o2(h5)

    model = Model(inputs=[i1, i2], outputs=[out1, out2])
    model.compile(optimizer="adam", loss="mse")
    return model

def test_topological_sort():
    model1 = build_topological_sort("tf")
    assert len(model1.layers) == 8

    out1 = model1.predict(inputs2)

    model2 = build_topological_sort("aitk")
    assert len(model2.layers) == 8

    model2.copy_weights(model1)

    out2 = model2.predict(inputs2)

    # FIXME? different, perhaps by order of concat?
    #compare_output(out1, out2)

def test_diff_multiple_inputs():
    model1 = build_model_multiple_inputs("aitk")
    model2 = build_model("aitk")

    w = model1.get_weights(flat=True)
    model2.set_weights(w)

    compare_weights(model1, model2)

    out1 = model1.predict(inputs2)
    out2 = model2.predict(inputs)

    compare_output(out1[0], out2)

    model1.fit(inputs2, targets1, epochs=1, shuffle=False)
    model2.fit(inputs, targets, epochs=1, shuffle=False)

    out1 = model1.predict(inputs2)
    out2 = model2.predict(inputs)

    compare_output(out1[0], out2)

def test_diff_functional():
    model1 = build_model_functional("aitk")
    model2 = build_model("aitk")

    w = model1.get_weights(flat=True)
    model2.set_weights(w)

    compare_weights(model1, model2)

    out1 = model1.predict(inputs)
    out2 = model2.predict(inputs)

    compare_output(out1, out2)

    model1.fit(inputs, targets, epochs=1, shuffle=False)
    model2.fit(inputs, targets, epochs=1, shuffle=False)

    out1 = model1.predict(inputs)
    out2 = model2.predict(inputs)

    compare_output(out1, out2)

def test_callbacks():
    model = build_model("aitk")
    from aitk.keras.callbacks import Callback

    class MyCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("epoch", epoch)

    model.fit(inputs, targets, callbacks=[MyCallback()], shuffle=False)

def test_metrics():
    from aitk.keras.metrics import tolerance_accuracy, ToleranceAccuracy
    from aitk.keras.optimizers import SGD

    tolerance_accuracy_class = ToleranceAccuracy(.2)
    tolerance_accuracy_class.name = "tolerance_accuracy_class"
    tolerance_accuracy.tolerance = 0.2

    model = build_model("aitk",
                        optimizer=SGD(learning_rate=0.1, momentum=0.9),
                        metrics=[tolerance_accuracy_class, tolerance_accuracy])

    history = model.fit(inputs, targets, epochs=300, shuffle=False,
                        batch_size=3)
    for ta1, ta2 in zip(history.history["tolerance_accuracy"],
                        history.history["tolerance_accuracy_class"]):
        assert ta1 == ta2

def test_shuffle():
    from aitk.keras.optimizers import SGD

    # Expect these to vary, even with same weights:
    model1 = build_model(
        "aitk",
        optimizer=SGD(learning_rate=0.1, momentum=0.9),
    )
    weights = model1.get_weights()

    model2 = build_model(
        "aitk",
        optimizer=SGD(learning_rate=0.1, momentum=0.9),
    )
    model2.set_weights(weights)

    history1 = model1.fit(inputs, targets, epochs=300, shuffle=True,
                          batch_size=3)
    history2 = model2.fit(inputs, targets, epochs=300, shuffle=True,
                          batch_size=3)

    same = True
    for ta1, ta2 in zip(history1.history["loss"],
                        history2.history["loss"]):
        if ta1 != ta2:
            same = False
            break
    assert not same

def test_no_shuffle():
    from aitk.keras.optimizers import SGD

    # Expect these to be exactly the same:
    model1 = build_model(
        "aitk",
        optimizer=SGD(learning_rate=0.1, momentum=0.9),
    )
    weights = model1.get_weights()

    model2 = build_model(
        "aitk",
        optimizer=SGD(learning_rate=0.1, momentum=0.9),
    )
    model2.set_weights(weights)

    history1 = model1.fit(inputs, targets, epochs=300, shuffle=False,
                          batch_size=3)
    history2 = model2.fit(inputs, targets, epochs=300, shuffle=False,
                          batch_size=3)

    same = True
    for ta1, ta2 in zip(history1.history["loss"],
                        history2.history["loss"]):
        if ta1 != ta2:
            same = False
            break
    assert same
