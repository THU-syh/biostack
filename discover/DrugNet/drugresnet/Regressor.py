import keras
from keras import Input, Model
from keras.layers import Dense, BatchNormalization, Dropout


def create_regressor(n_features, layers, n_outputs, optimizer=None):
    input_layer = Input(shape=(n_features,))
    dense = Dense(layers[0], activation='relu', name="dense_0")(input_layer)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)

    for i, layer in enumerate(layers[1:]):
        dense = Dense(layer, activation='relu', name="dense_{0}".format(i + 1))(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.5)(dense)
    dense = Dense(n_outputs, activation='sigmoid', name="output")(dense)
    model = Model(inputs=input_layer, outputs=dense)
    if optimizer is None:
        optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(optimizer=optimizer, loss=["mse"], metrics=["mae"])
    return model


model = create_regressor(x_train.shape[1], [1024, 256, 64, 4], 1, None)
model.summary()
model.fit(x=x_train,
          y=y_train,
          batch_size=batch_size,
          epochs=n_epochs,
          validation_data=(x_test, y_test),
          verbose=2,
          shuffle=True,
          callbacks=[csv_logger])
