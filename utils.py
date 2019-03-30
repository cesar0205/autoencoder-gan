import os
import numpy as np
import urllib
import tarfile
import imageio
import skimage
import warnings
from keras.datasets.mnist import load_data


def get_faces_data():
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz";
    directory = "../large_files/";
    images_file = "lfw.tgz";

    if not os.path.isdir(directory):
        if not os.path.isfile(images_file):
            urllib.request.urlretrieve(url, images_file)
        tarf = tarfile.open(images_file)
        tarf.extractall(path=directory)

    count = 0
    filenames = []
    for dir_, __, files in os.walk(os.path.join(directory, "lfw")):
        # relpath = os.path.relpath(dir_, directory)
        for file_ in files:
            full = os.path.join(dir_, file_)
            filenames.append(full)
            count += 1;

    new_directory = "../large_files/lfw_reshaped/"
    if not os.path.isdir(new_directory):
        os.mkdir(new_directory)
        for i, filename in enumerate(filenames):
            image = imageio.imread(filename)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_image = skimage.transform.resize(image, (40, 40), )
                imageio.imwrite(new_directory + str(i) + ".jpg", np.uint8(new_image * 255));

    return new_directory, count;




def get_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist = load_data()
    X_train = np.concatenate((X_train, X_test), axis = 0)
    y_train = np.concatenate((y_train, y_test), axis = 0)
    X_train = X_train.reshape((-1, 28, 28, 1))
    return X_train/255.0