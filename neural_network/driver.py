import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np

from neural_network.l_layer_neural_network import L2NN


def load_data():
    train_dataset = h5py.File('./datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('./datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_x = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
    train_y = train_set_y_orig.reshape((train_set_y_orig.shape[0], -1)).T
    test_x = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T
    test_y = test_set_y_orig.reshape((test_set_y_orig.shape[0], -1)).T

    # show_image(train_set_x_orig[10], train_y[0, 10], classes[train_y[0, 10]])

    return train_x, train_y, test_x, test_y, classes


def train_model(model_type, layer_sizes, file_name, learning_rate=0.0075, iterations=2500):
    train_x, train_y, test_x, test_y, classes = load_data()
    model = model_type
    model.fit(train_x, train_y, layer_sizes, learning_rate, iterations)
    pickle.dump(model, open(file_name, 'wb'))


def load_model(file_name):
    return pickle.load(open(file_name, 'rb'))


def show_image(image, y, imclass):
    plt.imshow(image)
    plt.show()
    print("y = " + str(y) + ". It is a " + imclass.decode("utf-8") + " picture.")


if __name__ == "__main__":
    # train_model(L2NN(), [20, 7, 5, 1], 'l2nn.sav')

    train_x, train_y, test_x, test_y, classes = load_data()
    model: L2NN = load_model('l2nn.sav')
    model.plot_cost()
    model.predict(train_x, train_y)
    predictions = model.predict(test_x, test_y)
    # L2NN.draw_mislabeled_images(classes, test_x, test_y, predictions)
    # model.predict_custom("./datasets/cat3.jpg", 1, 64)
