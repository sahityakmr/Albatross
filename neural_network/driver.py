from neural_network.l_layer_neural_network import L2NN
import h5py
import numpy as np


def load_data():
    train_dataset = h5py.File('./datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('./datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    train_set_x_orig = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
    train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0], -1)).T
    test_set_x_orig = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T
    test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0], -1)).T
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


train_x, train_y, test_x, test_y = load_data()
model = L2NN()
model.fit(train_x, train_y, [20, 7, 5, 1], learning_rate=0.0075, iterations=3000)
