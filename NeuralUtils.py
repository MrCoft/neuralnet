import numpy as np
from functools import reduce
import operator
import os

product = lambda vec: reduce(operator.mul, vec, 1)

def flatten_layers(model):
    if isinstance(model, NeuralNetwork):
        model = model.layers
    layers = []
    for layer in model:
        if type(layer) is Sequence:
            layers.extend(flatten_layers(layer.layers))
        elif type(layer) is Concat:
            layers.extend(flatten_layers(layer.layers))
        else:
            layers.append(layer)
    return layers

def layout_tensor(data, axes):
    shape = data.shape
    w, h = shape[axes[0]], shape[axes[1]]
    num = np.round(product(shape) / w / h).astype("int32")
    s = np.ceil(np.sqrt(num)).astype("int32")
    rng = list(range(data.ndim))
    rng.remove(axes[0])
    rng.remove(axes[1])
    data = np.transpose(data, rng + axes)
    data = data.reshape((num, w, h))
    res = np.zeros((s * w, s * h))
    i = 0
    for x in range(s):
        for y in range(s):
            res[x*w:(x+1)*w,y*h:(y+1)*h] = data[i]
            i += 1
            if i >= num:
                return res
    return res

def slice_index(index, length):
    if isinstance(index, int): index = slice(index, index + 1)
    start, stop = index.start, index.stop
    if start is None: start = 0
    if stop is None: stop = length
    if start < 0: start += length
    if stop < 0: stop += length
    return slice(start, stop)

def parse_log(path):
    log = open(path).read()

    entries = {}
    metrics = set()
    for line in log.splitlines():
        items = line.split()
        epoch = int(items[1])
        entries[epoch] = {}

        items = items[3:]
        items = [item for item in items if item != "-"]
        for metric, value in zip(items[::2], items[1::2]):
            metric = metric[:-1]

            entries[epoch][metric] = float(value)
            metrics.add(metric)

    epochs = sorted(entries)

    table = {}
    for metric in metrics:
        data = np.full(epochs[-1], float("nan"))
        for epoch in epochs:
            if metric in entries[epoch]:
                data[epoch - 1] = entries[epoch][metric]

        table[metric] = data

    return table

def compare_models(dir):
    model_dirs = next(os.walk('.'))[1]
    logs = [parse_log(dir + "/log.txt") for dir in model_dirs]

    import matplotlib.pyplot as plt
    for metric in logs[0]:
        plt.title(metric)
        for log in logs:
            plt.plot(log[metric], label=model_dirs[0])
        plt.xlabel("Epoch")
        plt.savefig("{}.png".format(metric))
        plt.close()
