import numpy as np
import keras
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Dataset import *
import Train

mem_size = 16

batch_size = 32
epochs = 40

from dataset.Piano import *
lib = PianoLib(os.path.expanduser("~/datasets/Piano-midi.de.pickle"))
data_train = DatasetMemsize(lib["train"], mem_size)[:1000]
data_test = DatasetMemsize(lib["test"], mem_size)[:1000]

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

model = Sequential()
model.add(LSTM(24, return_sequences=False, input_shape=(mem_size, lib.classes)))
model.add(Dropout(0.4))
#model.add(LSTM(1024))
#model.add(Dropout(0.4))
model.add(Dense(lib.classes))
model.add(Activation("sigmoid"))

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=[])
model.summary()

train = Train.train(
    "piano_net",

    model,
    data_train,

    data_test=data_test,
    metrics=["levenshtein"],

    displays=["predict_seq"],
)
train(batch_size=batch_size,
      epochs=epochs)