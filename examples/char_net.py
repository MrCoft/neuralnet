import numpy as np
import keras
import matplotlib.pyplot as plt

import os
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
import sys
sys.path.append(root)

from Dataset import *
import Train

mem_size = 100

batch_size = 32
epochs = 40

from dataset.Char import *
lib = CharLib(os.path.expanduser("~/datasets/text"))
data_train = DatasetMemsize(lib["eminem"], mem_size)[:]
data_test = data_train

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(mem_size, lib.classes)))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(lib.classes))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])
model.summary()

from Display import *
train = Train.train(
    root + "/cache/char_net",

    model,
    data_train,

    data_test=data_test,

    displays=[progress, predict_seq(multidim=False)],
)
train(batch_size=batch_size,
      epochs=epochs)