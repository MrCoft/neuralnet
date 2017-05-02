import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import traceback
from NeuralUtils import parse_log

def progress(train):
    output_dir = train["output_dir"]

    table = parse_log(output_dir + "/log.txt")

    for metric, data in sorted(table.items()):
        plt.title(metric)
        plt.plot(data)
        plt.xlabel("Epoch")
        plt.savefig(output_dir + "/{}.png".format(metric))
        if train["ipython"]:
            plt.show()
        plt.close()

def predict_seq(multidim):

    def display(train):
        demo_samples = 100

        output_dir = train["output_dir"]
        x, y = train["data_test"]
        x, y = x[:demo_samples], y[:demo_samples]

        guided_predict = np.zeros((demo_samples,) + y.shape[1:])
        raw_predict = np.zeros((demo_samples,) + y.shape[1:])

        for i in range(demo_samples):
            mem = x[i]
            vec = train["model"].predict(np.expand_dims(mem, axis=0))[0]
            raw_predict[i] = vec
            if multidim:
                guided_predict[i] = np.random.random(y.shape[1:]) < vec
            else:
                choice = np.random.choice(x.shape[-1], p=vec/np.sum(vec))
                guided_predict[i][choice] = 1

        plt.imshow(y.T, aspect="auto", origin="lower", cmap="Blues", interpolation="nearest")
        plt.savefig(output_dir + "/correct_predict.png")
        if train["ipython"]:
            plt.show()
        plt.close()

        try:
            plt.imshow(guided_predict.T, aspect="auto", origin="lower", cmap="Blues", interpolation="nearest")
            plt.savefig(output_dir + "/guided_predict_{}.png".format(train["epoch"]))
            if train["ipython"]:
                plt.show()
            plt.close()
        except:
            traceback.print_exc()

        try:
            plt.imshow(raw_predict.T, aspect="auto", origin="lower", norm=matplotlib.colors.LogNorm(), cmap="jet", interpolation="nearest")
            plt.colorbar()
            plt.savefig(output_dir + "/raw_predict_{}.png".format(train["epoch"]))
            if train["ipython"]:
                plt.show()
            plt.close()
        except:
            traceback.print_exc()

        try:
            if train["epoch"] % 10 == 0:
                import tools.Image
                tools.Image.images2gif(output_dir + "/guided_predict_", output_dir + "/guided_predict.gif")
                tools.Image.images2gif(output_dir + "/raw_predict_", output_dir + "/raw_predict.gif")
        except:
            traceback.print_exc()

    return display
