from functools import partial
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#Variational autoencoder tested with the MNIST dataset.

custom_layer = partial(tf.layers.dense, activation = tf.nn.elu, kernel_initializer = tf.variance_scaling_initializer())


from keras.datasets.mnist import load_data
(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.reshape((-1, 28*28))/255
X_test = X_test.reshape((-1, 28*28))/255


def batch_iterator(X, batch_size=128):
    m = len(X)
    for i in range(0, m, batch_size):
        start = i
        end = min(start + batch_size, m)
        yield X[start:end]


class VariationalAutoencoder():
    def __init__(self, n_inputs=28 * 28, n_hidden1=300, n_hidden2=150, n_codings=100, use_bernoulli=False):
        #The encoder will have two hidden layers.
        #Then we will add the codings layer
        #Finally, we have the decoder with two hidden layers again.


        #Use bernoulli is useful if the user wants to input images of 0 and 1's and output also 0 and 1's.
        #The final layer logits will be the input to a Bernoulli distribution model, from which we will sample
        #to get the reconstuction.

        self.eps = 1e-9
        self.n_inputs = n_inputs;
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_codings = n_codings
        self.n_hidden4 = n_hidden2
        self.n_hidden5 = n_hidden1
        self.n_outputs = n_inputs;
        self.use_bernoulli = use_bernoulli;
        self.sess = None;

    def fit(self, X_train, n_epochs=10, learning_rate=0.001):

        self.learning_rate = learning_rate;

        if (self.sess is not None):
            self.sess.close();

        self._build();

        self.sess = tf.Session();
        self.sess.run(self.init)
        for epoch in range(n_epochs):
            X_train_shuffled = shuffle(X_train)
            total_loss = 0
            for X_batch in batch_iterator(X_train_shuffled):
                batch_loss, __ = self.sess.run([self.elbo, self.opt_step], feed_dict={self.X: X_batch})
                total_loss += batch_loss;
            print("Epoch:", epoch, "Loss", -total_loss)

    def _build(self):

        n_inputs = self.n_inputs;
        n_hidden1 = self.n_hidden1
        n_hidden2 = self.n_hidden2
        n_codings = self.n_codings
        n_hidden4 = self.n_hidden4
        n_hidden5 = self.n_hidden5
        n_outputs = self.n_outputs;
        self.sess = None;

        tf.reset_default_graph()

        X = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])
        hidden1 = custom_layer(X, n_hidden1)
        hidden2 = custom_layer(hidden1, n_hidden2)
        codings_mean = custom_layer(hidden2, n_codings, activation=None)
        codings_sigma = tf.nn.softplus(custom_layer(hidden2, n_codings, activation=None))

        normal = tf.distributions.Normal(loc=tf.zeros(tf.shape(codings_mean)), scale=tf.ones(tf.shape(codings_sigma)))
        #Reparametrization trick in order to propagate the gradients after sampling from the normal distribution.
        sample = codings_mean + codings_sigma * normal.sample()

        hidden4 = custom_layer(sample, n_hidden4)
        hidden5 = custom_layer(hidden4, n_hidden5)

        logits = custom_layer(hidden5, n_outputs, activation=None)

        if self.use_bernoulli:
            #Create a Bernoulli distribution and sample to get an output of 0's and 1's
            bernoulli_model = tf.distributions.Bernoulli(logits=logits)
            reconstructions = bernoulli_model.sample();
            xentropy = -bernoulli_model.log_prob(X)
            #We can also calculate the xentropy in the usual way.
            #xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits);

        else:
            reconstructions = tf.sigmoid(logits);
            xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits);

        #We calculate the reconstruction loss by suming over all xentropies. Important not to get the mean as the
        #objective will be wrong.
        reconstruction_loss = tf.reduce_sum(xentropy)
        #KLD divergence loss between a normal distribution with codings_sigma standard deviation/codings_mean mean and
        #a standard normal.
        kld_loss = 0.5 * tf.reduce_sum(
            tf.square(codings_mean) + tf.square(codings_sigma) - 1 - tf.log(self.eps + codings_sigma));

        # Evidence lower bound
        elbo = - reconstruction_loss - kld_loss;
        # We try to maximize the ELBO
        opt_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(-elbo)

        init = tf.global_variables_initializer()

        self.codings_mean = codings_mean;
        self.codings_sigma = codings_sigma;
        self.reconstructions = reconstructions;
        self.sample = sample;
        self.X = X
        self.init = init
        self.opt_step = opt_step
        self.elbo = elbo

    def sample_image_from_prior(self, latent_sample=None):
        #Samples from a standard normal distribution and returns the reconstruction.
        #We can provide the normal sample or let the model do it.
        if latent_sample is None:
            latent_sample = np.random.randn(self.n_codings)
        output = self.sess.run(self.reconstructions, feed_dict={self.sample: [latent_sample]})
        return output;

    def sample_image_from_posterior(self, x):
        #Returns the reconstruction given an image as an input.
        return self.sess.run(self.reconstructions, feed_dict={self.X: [x]})

    def get_latent_space(self, X):
        #Returns the component corresponding to the "means" of the latent space.
        return self.sess.run(self.codings_mean, feed_dict={self.X: X})

def basic_latent_space_visualization(model, data):
    Z = model.get_latent_space(np.where(data > 0.5, 1.0, 0.0))
    plt.scatter(Z[:, 0], Z[:, 1], c=y_train)
    plt.title("Latent space")
    plt.axis('off')

def advanced_latent_space_visualization(model):
    n_elements = 20
    image_gen = np.zeros((28 * n_elements, 28 * n_elements))

    x_positions = np.linspace(-4, 4, n_elements)
    y_positions = np.linspace(-4, 4, n_elements)

    for index_i, i in enumerate(x_positions):
        for index_j, j in enumerate(y_positions):
            prior_sample = np.array([i, j])
            reconstruction = model.sample_image_from_prior(prior_sample)
            image_gen[28 * index_i:28 * index_i + 28, 28 * index_j: 28 * index_j + 28] = reconstruction.reshape(
                (28, 28));

    plt.figure(figsize=(10, 10))
    plt.imshow(image_gen, cmap="Greys")
    plt.title("Latent space with samples")
    plt.axis('off')

def generated_digits_visualization(model):
    plt.figure(figsize=(5, 10))

    for i in range(10):
        x = X_train[y_train == i].mean(axis=0);
        plt.subplot(10, 2, 2 * i + 1)
        if (i == 0):
            plt.title("Digit mean")
        plt.imshow(x.reshape(28, 28), cmap="Greys", interpolation='nearest')
        plt.axis('off')
        plt.subplot(10, 2, 2 * i + 2)
        if (i == 0):
            plt.title("Reconstruction")
        plt.imshow(model.sample_image_from_posterior(x).reshape(28, 28), cmap="Greys", interpolation='nearest')
        plt.axis('off')
    plt.show()

def main():
    model = VariationalAutoencoder(n_hidden1=300, n_hidden2=150, n_codings=2, use_bernoulli=True)
    model.fit(np.where(X_train > 0.5, 1.0, 0.0), n_epochs=10)


    basic_latent_space_visualization(model, X_train)
    advanced_latent_space_visualization(model)
    generated_digits_visualization(model)


if __name__ == "__main__":
    main();
