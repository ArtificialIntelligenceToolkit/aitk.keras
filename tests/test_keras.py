
def build_model_tf():
    from tensorflow.keras.layers import InputLayer, Dense, Activation
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(InputLayer(2))
    model.add(Dense(2, activation="relu"))
    model.add(Dense(1, activation="tanh"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile()
    return model

def build_model_aitk():
    from aitk.keras.layers import InputLayer, Dense, Activation
    from aitk.keras.models import Sequential

    model = Sequential()
    model.add(InputLayer(2))
    model.add(Dense(2, activation="relu"))
    model.add(Dense(1, activation="tanh"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile()
    return model

model_tf = build_model_tf()
print(model_tf.predict([[1, 1]]))

w = model_tf.get_weights()

model_aitk = build_model_aitk()
model_aitk.set_weights(w)
print(model_aitk.predict([[1, 1]]))

