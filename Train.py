import numpy as np
import natsort
import glob
import os
import shutil
import Loss
import traceback

def train(
        output_dir,

        model,
        data_train,

        data_test=None,
        metrics=None,

        displays=None,
        ipython=False,
):
    if metrics is None: metrics = []
    if displays is None: displays = []

    epoch = 0
    model_files = natsort.natsorted(glob.glob(output_dir + "/model_*"))
    if model_files:
        epoch = int(os.path.splitext(model_files[-1])[0][len(output_dir + "/model_"):])
        print("Found model at epoch {}".format(epoch))

    if os.path.exists(output_dir):
        if not model_files or input("Restart training (y/N)? ").lower() == "y":
            shutil.rmtree(output_dir)
            epoch = 0
        else:
            model.load_weights(model_files[-1])
    def mkdir(path):
        parent = os.path.dirname(path)
        if not os.path.exists(parent):
            mkdir(parent)
        if not os.path.exists(path):
            os.mkdir(path)
    mkdir(output_dir)

    class Metric: pass
    custom_metrics = []

    validate = data_test is not None
    if not validate:
        data_test = data_train

    for metric_name in metrics:
        metric = Metric()
        metric.name = metric_name
        metric.cmp, n = Loss.metrics[metric_name]
        metric.train_pts = np.random.randint(0, len(data_train[0]), (n,))
        if validate:
            metric.test_pts = np.random.randint(0, len(data_test[0]), (n,))
        custom_metrics.append(metric)

    def measure(cmp, data, pts):
        x, y = data
        correct = None
        predict = None
        score = 0
        for i in pts:
            mem = x[i]
            correct = y[i]
            vec = model.predict(np.expand_dims(mem, axis=0))[0]
            predict = np.random.random(y.shape[1:]) < vec
            score += cmp(correct, predict)
        score /= len(pts)
        return score

    scope = locals()
    def cb_log(e, logs):
        nonlocal epoch
        epoch += 1
        scope["epoch"] = epoch

        model.save_weights(output_dir + "/model_{}.h5".format(epoch))

        msg = ""
        msg += "Epoch {}".format(epoch)
        for metric in custom_metrics:
            try:
                value = measure(metric.cmp, data_train, metric.train_pts)
            except:
                traceback.print_exc()
                value = float("nan")
            logs[metric.name] = value
            if validate:
                try:
                    value = measure(metric.cmp, data_test, metric.test_pts)
                except:
                    traceback.print_exc()
                    value = float("nan")
                logs["val_" + metric.name] = value
        msg += " - "
        msg += " - ".join("{}: {}".format(key, value) for key, value in sorted(logs.items()))
        print(msg, file=open(output_dir + "/log.txt", "a"))

        if ipython:
            from IPython.display import clear_output
            clear_output()
        print("\r" + msg)
        display()

    if ipython:
        from IPython.core.display import display, HTML
        display(HTML('''
            <style>
                .output_wrapper, .output {
                    height:auto !important;
                    max-height:none;
                }
                .output_scroll {
                    box-shadow:none !important;
                    webkit-box-shadow:none !important;
                }
            </style>
        '''))

    def display():
        for display in displays:
            try:
                display(scope)
            except:
                traceback.print_exc()

    def train(epochs=1, **kwargs):
        from keras.callbacks import LambdaCallback
        epochs -= epoch

        if epoch:
            display()
        model.fit(*data_train,
                  epochs=epochs,
                  validation_data=data_test if validate else None,
                  verbose=1,
                  callbacks=[LambdaCallback(on_epoch_end=cb_log)],
                  **kwargs)
    return train
