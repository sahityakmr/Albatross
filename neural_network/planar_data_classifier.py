import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
import sklearn

ROWS = 0
COLUMNS = 1


def load_planar_dataset():
    np.random.seed(1)
    m = 400
    n = int(m / 2)
    d = 2
    x = np.zeros((m, d))
    y = np.zeros((m, 1), dtype='uint8')
    a = 4

    for j in range(2):
        ix = range(n * j, n * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, n) + np.random.randn(n) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(n) * 0.2
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    x = x.T
    y = y.T

    return x, y


def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure


def plot_decision_boundary(model, x, y):
    x_min, x_max = x[0, :].min() - 1, x[0, :].max() + 1
    y_min, y_max = x[1, :].min() - 1, x[1, :].max() + 1

    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    z = model(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.contourf(xx, yy, z)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[0, :], x[1, :], c=y[0])
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SingleLayerNeuralNetwork:

    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.hidden_layer_size = None
        self.input_layer_size = None
        self.output_layer_size = None
        self.weight_il = None
        self.weight_hl = None
        self.bias_il = None
        self.bias_hl = None
        self.sample_size = None
        self.learning_rate = None
        self.iterations = None

    def fit(self, train_x, train_y, hidden_layer_size=4, learning_rate=0.05, iterations=10000):
        self.train_x = train_x
        self.train_y = train_y
        self.sample_size = train_x.shape[COLUMNS]
        self.set_hyper_parameters(learning_rate, hidden_layer_size, iterations)
        self.initialize_parameters()
        self.__train()

    def set_hyper_parameters(self, learning_rate, hidden_layer_size, iterations):
        self.learning_rate = learning_rate
        self.hidden_layer_size = hidden_layer_size
        self.iterations = iterations

    def initialize_parameters(self):
        np.random.seed(2)
        self.input_layer_size = self.train_x.shape[ROWS]
        self.output_layer_size = 1
        self.weight_il = np.random.randn(self.hidden_layer_size, self.input_layer_size) * 0.01
        self.bias_il = np.zeros((self.hidden_layer_size, 1))
        self.weight_hl = np.random.randn(self.output_layer_size, self.hidden_layer_size) * 0.01
        self.bias_hl = np.zeros((self.output_layer_size, 1))

    def __train(self):
        for index in range(self.iterations):
            self.__propagation()
            self.__calculate_cost()
            self.__calculate_gradient()
            self.__update_parameters()
            self.__log_cost(index)

    def __propagation(self):
        self.output_hl = np.dot(self.weight_il, self.train_x) + self.bias_il
        self.activated_output_hl = np.tanh(self.output_hl)
        self.output_ol = np.dot(self.weight_hl, self.activated_output_hl) + self.bias_hl
        self.activated_output_ol = sigmoid(self.output_ol)

    def __calculate_gradient(self):
        self.dZ2 = self.activated_output_ol - self.train_y
        self.dW2 = np.dot(self.dZ2, self.activated_output_hl.T) / self.sample_size
        self.db2 = np.sum(self.dZ2, axis=1, keepdims=True) / self.sample_size
        self.dZ1 = np.multiply(np.dot(self.weight_hl.T, self.dZ2), (1 - np.power(self.activated_output_hl, 2)))
        self.dW1 = np.dot(self.dZ1, self.train_x.T) / self.sample_size
        self.db1 = np.sum(self.dZ1, axis=1, keepdims=True) / self.sample_size

    def __calculate_cost(self):
        self.cost = float(np.squeeze(-np.average(
            np.multiply(np.log(self.activated_output_ol), self.train_y) +
            np.multiply(np.log(1 - self.activated_output_ol), 1 - self.train_y))))

    def __update_parameters(self):
        self.weight_il -= self.learning_rate * self.dW1
        self.bias_il -= self.learning_rate * self.db1
        self.weight_hl -= self.learning_rate * self.dW2
        self.bias_hl -= self.learning_rate * self.db2

    def __log_cost(self, index):
        if index % 1000 == 0:
            print("Cost after iteration %i : %f" % (index, self.cost))

    def predict(self, test_x):
        output_hl = np.dot(self.weight_il, test_x) + self.bias_il
        activated_output_hl = np.tanh(output_hl)
        output_ol = np.dot(self.weight_hl, activated_output_hl) + self.bias_hl
        activated_output_ol = sigmoid(output_ol)

        predictions = np.where(activated_output_ol > 0.5, 1., 0.)
        return predictions


if __name__ == "__main__":
    x, y = load_planar_dataset()

    plt.scatter(x[0, :], x[1, :], c=y[0])
    plt.show()

    clf = LogisticRegressionCV()
    clf.fit(x.T, y.T)

    plot_decision_boundary(lambda _x: clf.predict(_x), x, y)
    plt.title("Logistic Regression")

    LR_predictions = clf.predict(x.T)

    print('Accuracy of logistic regression: %d ' % float(
        (np.dot(y, LR_predictions) + np.dot(1 - y, 1 - LR_predictions)) / float(y.size) * 100) +
          '% ' + "(percentage of correctly labelled datapoints)")

    model = SingleLayerNeuralNetwork()
    model.fit(x, y, learning_rate=1.2)
    plot_decision_boundary(lambda _x: model.predict(_x.T), x, y)

    SLNN_predictions = model.predict(x)

    print('Accuracy of logistic regression: %d ' % float(
        (np.dot(y, SLNN_predictions.T) + np.dot(1 - y, 1 - SLNN_predictions.T)) / float(y.size) * 100) +
          '% ' + "(percentage of correctly labelled datapoints)")

# do later tasks:
'''
# Check with different size of HL
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
'''

'''
# Check for other dataset
# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

### START CODE HERE ### (choose your dataset)
dataset = "noisy_moons"
### END CODE HERE ###

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y%2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
'''
