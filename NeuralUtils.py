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
        epoch = int(items[1].split("/")[0])
        entries[epoch] = {}

        items = [item for item in items if item != "-"]
        items = items[2:]
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

def compare_models(dir, names,
                   xlim=None, labels=None, metrics=None):
    logs = [parse_log(os.path.join(dir, name, "log.txt")) for name in names]

    import matplotlib.pyplot as plt
    for metric in logs[0]:
        fig = plt.figure()
        fig.set_size_inches(9.6, 6)
        ax = plt.subplot(111)
        plt.ylabel(metrics[metric] if metrics else metric)
        for i, log in enumerate(logs):
            data = log[metric][:xlim if xlim else -1]
            ax.plot(range(1, len(data) + 1), data, label=labels[i] if labels else names[i])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
        plt.xlabel("Epoch")
        plt.savefig("{}.png".format(metric))
        plt.close()