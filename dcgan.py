import tensorflow as tf
import numpy as np
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import imageio
from utils import get_faces_data, get_mnist_data;

conv2d = partial(tf.layers.conv2d, padding = "SAME", activation = tf.nn.leaky_relu)
dense = partial(tf.layers.dense, activation = tf.nn.leaky_relu)
deconv2d = partial(tf.layers.conv2d_transpose, padding = "SAME", activation = tf.nn.leaky_relu)

#Utility class that allows to sample batches from the faces dataset.
class FacesIterator():
    def __init__(self, image_dir, n_total, batch_size):
        self.image_dir = image_dir;
        self.n_total = n_total;
        self.batch_size = batch_size;

    def get_batch(self):
        images = [imageio.imread(self.image_dir + "/" + str(imgidx) + ".jpg")/255.0 for imgidx in np.random.choice(self.n_total, size = self.batch_size)];
        return np.array(images)

#Utility class that allows to sample batches from the MNIST dataset.
class MnistIterator():
    def __init__(self, X_train, n_total, batch_size):
        self.X_train = X_train;
        self.n_total = n_total;
        self.batch_size = batch_size;

    def get_batch(self):
        idx = np.random.randint(0, self.n_total, self.batch_size)
        return self.X_train[idx]

#Generative Adversarial Network
#This class allows to modify hyperparameters and number of neurons on each layer. The type, number and the position
# of the layers are hardcoded but they are enough to fit both the mnist and the faces datasets.
class GAN:
    def __init__(self, d_params, g_params, learning_rate=0.00015):
        self.d_params = d_params;
        self.g_params = g_params;
        self.learning_rate = learning_rate;
        self.latent_dim = g_params['latent_dim']
        self.sess = None;

    def _build(self):

        tf.reset_default_graph()

        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=(None, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        self.z = tf.placeholder(dtype=tf.float32, shape=(None, self.latent_dim))

        self.g_is_training = tf.placeholder(dtype=tf.bool)
        self.d_is_training = tf.placeholder(dtype=tf.bool)


        #Create the discriminator NN.
        discriminator_out = self._build_discriminator(self.images, self.d_is_training)
        #Create the combined model. A generator plus the discriminator built previously.
        self.generated_images = self._build_generator(self.z, self.g_is_training);
        combined_out = self._build_discriminator(self.generated_images, self.d_is_training, reuse=True)

        #Gather the variables corresponding to the genrator and discriminator to control their training weights.
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator");
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator");

        real_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(discriminator_out),
                                                                logits=discriminator_out)
        fake_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(combined_out), logits=combined_out)

        # Multiply by 0.5 because we are considering the double of data for the discriminator loss than for the generator.
        self.d_loss = 0.5 * tf.reduce_mean((real_xentropy + fake_xentropy))

        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(combined_out), logits=combined_out)
        self.g_loss = tf.reduce_mean(xentropy)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);
        with tf.control_dependencies(update_ops):
            self.d_opt_step = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.d_loss,
                                                                                                   var_list=d_vars);
            self.g_opt_step = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.g_loss,
                                                                                                   var_list=g_vars);

        self.init = tf.global_variables_initializer()

    def fit(self, X_iterator, total_steps=50000, save_interval=1000, title="Mnist"):

        self.title = title;
        sample = X_iterator.get_batch();
        self.image_shape = sample.shape[1:]
        self._build();
        batch_size = sample.shape[0];

        if self.sess is not None:
            self.sess.close();

        self.sess = tf.Session()
        self.sess.run(self.init)

        g_loss_ = np.inf
        d_loss_ = np.inf

        for step in range(total_steps):

            noise = np.random.rand(batch_size, self.latent_dim)
            X_batch = X_iterator.get_batch();

            #Technique that helps to alleviate the collapse mode.
            train_generator = True;
            if (g_loss_ * 1.5 < d_loss_):
                train_generator = False;

            train_discriminator = True;
            if (d_loss_ * 2.0 < g_loss_):
                train_discriminator = False;

            if (train_discriminator):
                # print("Training discriminator:", g_loss_, d_loss_)
                self.sess.run([self.d_opt_step],
                              feed_dict={self.images: X_batch, self.z: noise, self.g_is_training: False})

            if (train_generator):
                # print("Training generator:", g_loss_, d_loss_)
                self.sess.run([self.g_opt_step], feed_dict={self.z: noise, self.g_is_training: True})

            d_loss_, g_loss_ = self.sess.run([self.d_loss, self.g_loss], feed_dict={self.images: X_batch,
                                                                                    self.z: noise,
                                                                                    self.g_is_training: False})

            # We need to evaluate the loss here. Otherwise the d_loss or the g_loss will not be updated because of the
            # above if statements and in the loss_comparisons we would end up using
            # 0o9old d_loss and g_loss values.

            if (step % 1000 == 0):
                print("Step:", step, "Discriminator loss:", d_loss_, "Generator loss:", g_loss_, "Step:", step)
                self._save_images(self.sess, step)

    #Construct discriminator
    def _build_discriminator(self, X_in, d_is_training, reuse=None):

        dropout_rate = self.d_params['d_dropout_rate']

        conv1_params = self.d_params['conv1_params']
        conv2_params = self.d_params['conv2_params']
        conv3_params = self.d_params['conv3_params']

        dense1_params = self.d_params['dense1_params']
        dense2_params = self.d_params['dense2_params']

        #The same weights will be shared by the discriminator and the generator
        with tf.variable_scope("discriminator", reuse=reuse):
            conv1 = conv2d(X_in, filters=conv1_params[0], kernel_size=conv1_params[1], strides=conv1_params[2])
            conv1 = tf.layers.dropout(conv1, rate=dropout_rate)

            conv2 = conv2d(conv1, filters=conv2_params[0], kernel_size=conv2_params[1], strides=conv2_params[2])
            conv2 = tf.layers.dropout(conv2, rate=dropout_rate)

            conv3 = tf.layers.conv2d(conv2, filters=conv3_params[0], kernel_size=conv3_params[1],
                                     strides=conv3_params[2])
            conv3 = tf.layers.dropout(conv3, rate=dropout_rate)

            flatten = tf.layers.flatten(conv3);

            dense1 = dense(flatten, dense1_params[0])

            dense2 = dense(dense1, dense2_params[0], activation=None)

            return dense2;

    #Construct generator
    def _build_generator(self, z, is_training):

        self.dropout_rate = self.g_params['g_dropout_rate']

        # Create generator

        pp1 = self.g_params['pp1']
        pp2 = self.g_params['pp2']

        # Filters, kernel_size, stride

        dp1 = self.g_params['dp1']
        dp2 = self.g_params['dp2']
        dp3 = self.g_params['dp3']
        dp4 = self.g_params['dp4']

        momentum = 0.99;
        with tf.variable_scope("generator"):
            dense1 = dense(z, pp1[0] * pp1[0] * pp1[1])
            # dense1 = tf.layers.dropout(dense1, rate = dropout_rate);
            dense1 = tf.contrib.layers.batch_norm(dense1, is_training=is_training, decay=momentum)

            projection = tf.reshape(dense1, (-1, pp1[0], pp1[0], pp1[1]))
            projection = tf.image.resize_images(projection, [pp2[0], pp2[1]])

            deconv1 = deconv2d(projection, filters=dp1[0], kernel_size=dp1[1], strides=dp1[2])
            # deconv1 = tf.layers.dropout(deconv1, rate = dropout_rate);
            deconv1 = tf.contrib.layers.batch_norm(deconv1, is_training=is_training, decay=momentum)

            deconv2 = deconv2d(deconv1, filters=dp2[0], kernel_size=dp2[1], strides=dp2[2])
            # deconv2 = tf.layers.dropout(deconv2, rate = dropout_rate);
            deconv2 = tf.contrib.layers.batch_norm(deconv2, is_training=is_training, decay=momentum)

            deconv3 = deconv2d(deconv2, filters=dp3[0], kernel_size=dp3[1], strides=dp3[2])
            # deconv3 = tf.layers.dropout(deconv3, rate = dropout_rate);
            deconv3 = tf.contrib.layers.batch_norm(deconv3, is_training=is_training, decay=momentum)

            deconv4 = deconv2d(deconv3, filters=dp4[0], kernel_size=dp4[1], strides=dp4[2], activation=tf.nn.sigmoid);

            return deconv4

    #Save images to disk at invervals
    def _save_images(self, sess, epoch):
        r, c = 5, 5
        z_noise = np.random.rand(r * c, self.latent_dim)

        images_gen = self.sess.run(self.generated_images, feed_dict={self.z: z_noise, self.g_is_training: False})

        if (self.image_shape[2] == 1):
            images_gen = images_gen.reshape((-1, self.image_shape[0], self.image_shape[1]))
        else:
            images_gen = images_gen.reshape((-1, self.image_shape[0], self.image_shape[1], self.image_shape[2]))

        # images_gen = images_gen
        fig, axs = plt.subplots(r, c);
        fig.suptitle("%s at epoch %d" % (self.title, epoch))
        count = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(images_gen[count], cmap="Greys")
                axs[i, j].axis('off')
                count += 1

        plt.savefig("%s %d.png" % (self.title, epoch))

#Loads the faces ("http://vis-www.cs.umass.edu/lfw/lfw.tgz") dataset and creates a GAN.
def faces():
    g_params = {
        "latent_dim": 64,
        "g_dropout_rate": 0.6,
        # Projection params

        "pp1": [4, 3],
        "pp2": [10, 10],

        # Filters, kernel_size, stride

        "dp1": [256, 5, 2],
        "dp2": [128, 5, 2],
        "dp3": [64, 5, 1],
        "dp4": [3, 5, 1],
    }

    d_params = {
        "d_dropout_rate": 0.5,
        "conv1_params": [256, 5, 2],
        "conv2_params": [128, 5, 1],
        "conv3_params": [64, 5, 1],

        "dense1_params": [128],
        "dense2_params": [1]
    }

    new_directory, n_files = get_faces_data();
    model = GAN(d_params, g_params);
    faces_iterator = FacesIterator(new_directory, n_files, 64)
    model.fit(faces_iterator, title="Faces")

#Loads the MNIST dataset and creates a GAN.
def mnist():

    g_params = {
        "latent_dim" : 64,
        "g_dropout_rate" : 0.6,
        #Projection params

        "pp1" : [4, 1],
        "pp2" : [7, 7],

        #Filters, kernel_size, stride

        "dp1" : [64, 5, 2],
        "dp2" : [64, 5, 2],
        "dp3" : [64, 5, 1],
        "dp4" : [1, 5, 1],
    }

    d_params = {
        "d_dropout_rate" : 0.5,
        "conv1_params" : [64, 5, 2],
        "conv2_params" : [64, 5, 1],
        "conv3_params" : [64, 5, 1],

        "dense1_params" : [128],
        "dense2_params" : [1]
    }

    X_train = get_mnist_data();
    model = GAN(d_params, g_params);
    mnist_iterator = MnistIterator(X_train, len(X_train), 64)
    model.fit(mnist_iterator, title = "Mnist")


if __name__ == "__main__":
    mnist()
    #faces()