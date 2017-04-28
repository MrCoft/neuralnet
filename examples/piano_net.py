import numpy as np
import keras
import matplotlib.pyplot as plt

import os
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
import sys
sys.path.append(root)

from Dataset import *
import Train

mem_size = 16

batch_size = 32
epochs = 40

from dataset.Piano import *
lib = PianoLib(os.path.expanduser("~/datasets/Piano-midi.de.pickle"))
data_train = DatasetMemsize(lib["train"], mem_size)[:]
data_test = DatasetMemsize(lib["test"], mem_size)[:]

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(mem_size, lib.classes)))
model.add(Dropout(0.4))
model.add(LSTM(128))
model.add(Dropout(0.4))
model.add(Dense(lib.classes))
model.add(Activation("sigmoid"))

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=[])
model.summary()

from Display import *
train = Train.train(
    root + "/cache/piano_net",

    model,
    data_train,

    data_test=data_test,
    metrics=["levenshtein"],

    displays=[progress, predict_seq(multidim=True), demo_midi(lib)],
)
train(batch_size=batch_size,
      epochs=epochs)