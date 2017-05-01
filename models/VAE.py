import numpy as np
from keras import backend as K
from keras import metrics

def vae_loss(z_mean, z_log_var, input_shape):

    def loss(x, x_decoded):
        x = K.flatten(x)
        x_decoded = K.flatten(x_decoded)
        reconstruction_loss = np.product(input_shape) * metrics.binary_crossentropy(x, x_decoded)
        kl_per_example = .5 * (K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=1))
        kl_normal_loss = K.mean(kl_per_example)
        return reconstruction_loss + kl_normal_loss

    return loss

def demo_vae_gen(ae, grid=(4, 4)):
    encoder, decoder = ae

    def display(train):
        output_dir = train["output_dir"]

        batch_size, encoded_dim = decoder.input_shape
        batch_size, w, h, c = decoder.output_shape
        img = np.zeros((w * grid[0], h * grid[1], 3))
        for _x in range(grid[0]):
            for _y in range(grid[1]):
                z = np.random.normal(size=encoded_dim)
                vec = decoder.predict(np.expand_dims(z, axis=0))[0]

                img[_x*w:(_x+1)*w,_y*h:(_y+1)*h] = vec

        import matplotlib.pyplot as plt
        import scipy.misc

        plt.imshow(np.transpose(img, (1, 0, 2)))
        plt.title("Generated")
        scipy.misc.imsave(output_dir + "/generated_image_{}.png".format(train["epoch"]), np.transpose(img, (1, 0, 2)))
        if train["ipython"]:
            plt.show()
        plt.close()

    return display
