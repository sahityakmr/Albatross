import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

FEATURE_SIZE_INDEX = 0
SAMPLE_SIZE_INDEX = 1


class LayerCache:
    def __init__(self):
        self.weight = None
        self.bias = None
        self.output = None
        self.activated_output = None
        self.layer_size = None
        self.dZ = None
        self.dW = None
        self.dB = None
        self.dA = None
        self.activation = None
        self.reverse_activation = None


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


class L2NN:

    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.feature_size = None
        self.sample_size = None
        self.layer_sizes: list = list()

    def fit(self, train_x, train_y, layer_sizes: list, learning_rate=0.05, iterations=2000, log_cost=True,
            log_step=100, diminishing_factor=0.01):

        # expects each sample as a column,
        # so that number of rows is input vector size and number of column is a sample size
        self.train_x = train_x
        self.train_y = train_y

        self.feature_size = train_x.shape[FEATURE_SIZE_INDEX]
        self.sample_size = train_x.shape[SAMPLE_SIZE_INDEX]

        # doesn't expect size of input layer
        self.layer_sizes = layer_sizes

        # data standardization
        self.train_x = self.train_x / 255  # (division specific for image inputs)

        self.__set_hyper_parameters(learning_rate, iterations, log_cost, log_step, diminishing_factor)
        self.__initialize_parameters()
        self.__train()

    def __set_hyper_parameters(self, learning_rate, iterations, log_cost, log_step,
                               diminishing_factor):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.log_cost = log_cost
        self.log_step = log_step
        self.diminishing_factor = diminishing_factor

    def __initialize_parameters(self):
        np.random.seed(1)

        self.layer_sizes.insert(0, self.feature_size)  # layer0 as input layer
        self.layer_count = len(self.layer_sizes)
        self.caches = list()
        for i in range(self.layer_count):
            self.caches.append(LayerCache())

        # activation output of input layer is input itself
        self.caches[0].activated_output = self.train_x
        self.caches[0].layer_size = self.feature_size

        for index in range(self.layer_count - 1):
            result_layer_size = self.caches[index + 1].layer_size = self.layer_sizes[index + 1]
            input_layer_size = self.caches[index].layer_size
            self.caches[index].weight = np.random.randn(result_layer_size, input_layer_size) / np.sqrt(
                input_layer_size)  # here using diminishing factor to keep weights close to zero
            self.caches[index].bias = np.zeros((result_layer_size, 1))
            self.caches[index].activation = relu
            self.caches[index].reverse_activation = relu_backward

        # setting sigmoid activation for last function
        self.caches[-1].activation = sigmoid
        self.caches[-1].reverse_activation = sigmoid_backward

    def __train(self):
        self.cost_list = list()
        for i in range(self.iterations):
            self.__make_prediction()
            self.__calculate_cost()
            self.__calculate_gradient()
            self.__gradient_descent()
            self.__log_cost(i)

    def __make_prediction(self):
        for layer in range(1, self.layer_count):
            self.__propagate(layer)

    def __propagate(self, layer):
        self.caches[layer].output = np.dot(self.caches[layer - 1].weight, self.caches[layer - 1].activated_output) + \
                                    self.caches[layer - 1].bias
        self.caches[layer].activated_output = self.caches[layer].activation(self.caches[layer].output)

    def __calculate_cost(self):
        cost = -np.average(np.multiply(np.log(self.caches[-1].activated_output), self.train_y) +
                           np.multiply(np.log(1 - self.caches[-1].activated_output), (1 - self.train_y)))
        cost = float(np.squeeze(cost))
        self.curr_cost = cost

    def __calculate_gradient(self):
        self.prev_dA = -(np.divide(self.train_y, self.caches[-1].activated_output) -
                         np.divide(1 - self.train_y, 1 - self.caches[-1].activated_output))

        for layer in reversed(range(1, self.layer_count)):
            self.__backward(layer)

    def __backward(self, layer):
        self.caches[layer].dA = self.prev_dA
        self.caches[layer].dZ = self.caches[layer].reverse_activation(self.caches[layer].dA, self.caches[layer].output)

        self.caches[layer - 1].dW = 1. * np.dot(self.caches[layer].dZ, self.caches[layer - 1].activated_output.T) / \
                                    self.sample_size
        self.caches[layer - 1].dB = 1. * np.sum(self.caches[layer].dZ, axis=1, keepdims=True) / \
                                    self.sample_size
        self.prev_dA = np.dot(self.caches[layer - 1].weight.T, self.caches[layer].dZ)

    def __gradient_descent(self):
        for layer in range(self.layer_count - 1):
            self.caches[layer].weight = self.caches[layer].weight - self.learning_rate * self.caches[layer].dW
            self.caches[layer].bias = self.caches[layer].bias - self.learning_rate * self.caches[layer].dB

    def __log_cost(self, index):
        if index % self.log_step == 0:
            self.cost_list.append(self.curr_cost)
            if self.log_cost:
                print("Cost after %i iterations : %f" % (index, self.curr_cost))

    def plot_cost(self):
        plt.plot(np.squeeze(self.cost_list))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning Rate : " + str(self.learning_rate))
        plt.show()

    def predict(self, test_x, test_y):
        activated_output = test_x / 255  # (division specific for image inputs)
        sample_size = test_x.shape[1]
        predictions = np.zeros((1, sample_size))
        for index in range(1, self.layer_count):
            activated_output = self.caches[index].activation(
                np.dot(self.caches[index - 1].weight, activated_output) + self.caches[index - 1].bias)
        for index in range(activated_output.shape[1]):
            if activated_output[0, index] > 0.5:
                predictions[0, index] = 1
            else:
                predictions[0, index] = 0

        print("Accuracy : " + str(np.sum((predictions == test_y) / sample_size)))
        return predictions

    @staticmethod
    def draw_mislabeled_images(classes, x, y, predictions):
        a = predictions + y
        mislabeled_indices = np.asarray(np.where(a == 1))
        plt.rcParams['figure.figsize'] = (40.0, 40.0)
        num_mismatches = len(mislabeled_indices[0])
        print(mislabeled_indices)
        for i in range(num_mismatches):
            index = mislabeled_indices[1][i]

            plt.subplot(2, num_mismatches, i + 1)
            plt.imshow(x[:, index].reshape(64, 64, 3), interpolation='nearest')
            plt.axis('off')
            plt.title("Prediction: " + classes[int(predictions[0, index])].decode("utf-8") + "\n Class: " + classes[
                y[0, index]].decode("utf-8"))
        plt.show()

    def predict_custom(self, file, y, num_px):
        orig_image = np.array((Image.open(file)).resize((num_px, num_px)))
        image = orig_image.reshape((num_px * num_px * 3, 1))
        image = image / 255
        prediction = self.predict(image, y)
        plt.imshow(orig_image)
        plt.show()
        print("y = " + str(y) + " prediction = " + str(prediction))
