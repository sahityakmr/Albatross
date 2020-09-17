import numpy as np

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
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.feature_size = None
        self.sample_size = None
        self.layer_sizes: list = list()

    def fit(self, train_x, train_y, layer_sizes: list, learning_rate=0.05, iterations=2000, log_cost=True,
            log_step=100, diminishing_factor=0.01):
        self.train_x = train_x / 255
        self.train_y = train_y
        self.feature_size = train_x.shape[FEATURE_SIZE_INDEX]
        self.sample_size = train_x.shape[SAMPLE_SIZE_INDEX]
        self.layer_sizes = layer_sizes

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
        self.layer_count = len(self.layer_sizes)
        self.caches = list()
        for i in range(self.layer_count + 1):
            self.caches.append(LayerCache())

        self.caches[0].activated_output = self.train_x
        self.caches[0].layer_size = self.feature_size

        for index in range(self.layer_count):
            result_layer_size = self.caches[index + 1].layer_size = self.layer_sizes[index]
            input_layer_size = self.caches[index].layer_size
            self.caches[index].weight = np.random.randn(result_layer_size, input_layer_size) / np.sqrt(input_layer_size)
            self.caches[index].bias = np.zeros((result_layer_size, 1))
            self.caches[index].activation = relu
            self.caches[index].reverse_activation = relu_backward

        self.caches[self.layer_count].activation = sigmoid
        self.caches[self.layer_count].reverse_activation = sigmoid_backward

    def __train(self):
        self.cost_list = list()
        for i in range(self.iterations):
            self.__make_prediction()
            self.__calculate_cost()
            self.__calculate_gradient()
            self.__gradient_descent()
            self.__log_cost(i)

    def __make_prediction(self):
        for layer in range(1, self.layer_count + 1):
            self.__propagate(layer)

    def __propagate(self, layer):
        self.caches[layer].output = np.dot(self.caches[layer - 1].weight, self.caches[layer - 1].activated_output) + \
                                    self.caches[layer - 1].bias
        self.caches[layer].activated_output = self.caches[layer].activation(self.caches[layer].output)

    def __calculate_cost(self):
        cost = -np.average(np.multiply(np.log(self.caches[self.layer_count].activated_output), self.train_y) +
                           np.multiply(np.log(1 - self.caches[self.layer_count].activated_output), (1 - self.train_y)))
        cost = float(np.squeeze(cost))
        self.curr_cost = cost

    def __calculate_gradient(self):
        self.prev_dA = -(np.divide(self.train_y, self.caches[self.layer_count].activated_output) -
                         np.divide(1 - self.train_y, 1 - self.caches[self.layer_count].activated_output))

        for layer in reversed(range(self.layer_count)):
            self.__backward(layer, relu_backward)

    def __backward(self, layer, reverse_activation):
        self.caches[layer + 1].dA = self.prev_dA
        self.caches[layer + 1].dZ = self.caches[layer + 1].reverse_activation(self.caches[layer + 1].dA,
                                                                              self.caches[layer + 1].output)
        self.caches[layer].dW = np.dot(self.caches[layer + 1].dZ, self.caches[layer].activated_output.T) / \
                                self.caches[layer].layer_size
        self.caches[layer].dB = np.sum(self.caches[layer + 1].dZ, axis=1, keepdims=True) / \
                                self.caches[layer].layer_size
        self.prev_dA = np.dot(self.caches[layer].weight.T, self.caches[layer + 1].dZ)

    def __gradient_descent(self):
        for layer in range(self.layer_count):
            self.caches[layer].weight = self.caches[layer].weight - self.learning_rate * self.caches[layer].dW
            self.caches[layer].bias = self.caches[layer].bias - self.learning_rate * self.caches[layer].dB

    def __log_cost(self, index):
        if index == 0 or (index + 1) % self.log_step == 0:
            self.cost_list.append(self.curr_cost)
            if self.log_cost:
                print("Cost after %i iterations : %f" % (index + 1, self.curr_cost))
