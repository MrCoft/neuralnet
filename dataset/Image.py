import numpy as np
import os

def open_image_dataset(path, size):
    files = [os.path.join(path, file) for file in next(os.walk(path))[2]]

    data = np.zeros((len(files),) + size + (3,))
    print("Loading images...")
    for i, file in enumerate(files):
        data[i] = open_image(file, size)
        if i % 10 == 0:
            print("{} %\r".format(int((i + 1) / len(files) * 100)), end="")
    print("100 %")

    return data

def open_image(path, size):
    from PIL import Image
    img = Image.open(path).convert('RGB')
    img = np.array(img)

    w, h = size
    r = w / h
    src_w, src_h = img.shape[:2]
    src_r = src_w / src_h
    if src_r > r:
        dw = round((src_w - src_h * r) / 2)
        img = img[dw:dw + int(src_h * r)]
    else:
        dh = round((src_h - src_w / r) / 2)
        img = img[:, dh:dh + int(src_w / r)]

    from scipy.misc import imresize
    img = imresize(img, size, "lanczos")

    img = img.astype("float32")
    img /= 255.

    return img

def demo_ae(grid=(4, 4)):
    def display(train):
        output_dir = train["output_dir"]
        x, y = train["data_test"]

        w, h, c = x[0].shape
        correct = np.zeros((w * grid[0], h * grid[1], 3))
        decoded = np.zeros((w * grid[0], h * grid[1], 3))
        i = 0
        for _x in range(grid[0]):
            for _y in range(grid[1]):
                mem = x[i].reshape((1,) + x.shape[1:])
                vec = train["model"].predict(mem)[0]

                correct[_x*w:(_x+1)*w,_y*h:(_y+1)*h] = y[i]
                decoded[_x*w:(_x+1)*w,_y*h:(_y+1)*h] = vec

                i += 1

        import matplotlib.pyplot as plt

        plt.imshow(correct)
        plt.title("Correct")
        plt.savefig(output_dir + "/correct_image.png")
        if train["ipython"]:
            plt.show()
        plt.close()

        plt.imshow(decoded)
        plt.title("Decoded")
        plt.savefig(output_dir + "/decoded_image_{}.png".format(train["epoch"]))
        if train["ipython"]:
            plt.show()
        plt.close()

        if train["epoch"] % 5 == 0:
            import tools.Image
            tools.Image.images2gif(output_dir + "/decoded_image_", output_dir + "/decoded_image.gif")

    return display

# generator - read imagemagick

# on next item, produce from magick - ~ 90s
# reading sequential from memory - ~ 12 s
# save in memory - 600 MB

# set cache file, read h5 if possible, otherwise create, hold in memory, iterator option
# test - load Frozen (27 GB)
    # 553 s reading (276.5s)

    # ffmpeg stream vs seek
    # calls speed

    # try on 1GB total data