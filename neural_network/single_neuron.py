import numpy as np
import matplotlib.pyplot as plt


def sigmoid_activation(input):
    return 1 / (1 + np.exp(-input))


FEATURE_SIZE_INDEX = 0
SAMPLE_SIZE_INDEX = 1


class LogisticRegression:

    # expects input matrix of dimension [feature_size x sample_size]
    def __init__(self):
        self.feature_size = None
        self.sample_size = None
        self.train_x = None
        self.train_y = None

    def __initialize_parameters(self):
        self.weight = np.zeros((1, self.feature_size))
        self.bias = 0
        pass

    def __set_hyper_parameters(self, learning_rate, iterations, log_cost, log_step):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.log_cost = log_cost
        self.log_step = log_step

    def fit(self, train_x, train_y, learning_rate=0.05, iterations=2000, log_cost=True, log_step=100):
        self.feature_size = train_x.shape[FEATURE_SIZE_INDEX]
        self.sample_size = train_x.shape[SAMPLE_SIZE_INDEX]
        self.train_x = train_x
        self.train_y = train_y

        self.__initialize_parameters()
        self.__set_hyper_parameters(learning_rate, iterations, log_cost, log_step)

        self.__train()

    def predict(self, test_x):
        predict_y = np.where(sigmoid_activation(np.dot(self.weight, test_x) + self.bias) > 0.5, 1, 0)
        return predict_y

    def __train(self):
        self.cost_list = list()

        for i in range(self.iterations + 1):
            self.__propagate()
            self.__gradient_descent()

            if i % self.log_step == 0:
                self.cost_list.append(self.curr_cost)
                if self.log_cost:
                    print("Cost after %i iterations : %f" % (i, self.curr_cost))
        self.train_accuracy = 100 - np.mean(np.abs(self.predict(self.train_x) - self.train_y) * 100)
        self.log_training_response()

    def __propagate(self):

        # make prediction with current weight
        z = np.dot(self.weight, self.train_x) + self.bias
        predicted_y = sigmoid_activation(z)

        # calculate cost
        # loss function for logistic regression (L) = -(train_y.log(predicted_y) + (1-train_y).log(1-predicted_y))
        self.curr_cost = -np.mean(self.train_y * np.log(predicted_y) + (1 - self.train_y) * np.log(1 - predicted_y))

        # calculate gradients
        # we need to calculate slope of loss function defined above
        # we have, dL/dy_hat = -y/y_hat - (1-y)/(1-y_hat) : y_hat is predicted_y ............(0)
        # dL/dZ = dL/dy_hat * dy_hat/dZ (by Chain Rule)        ..............................(1)
        # also, y_hat = sigmoid(z)
        # so, dy_hat/dZ = sigmoid(z)(1 - sigmoid(z)) or, y_hat(1-y_hat)         .............(2)
        # and, from (0),(1) and (2), dL/dZ = (y_hat - y)                        .............(3)
        # dL/dW = dL/dZ * dZ/dW (by Chain Rule)
        # Z = WX + b                                                            .............(4)
        # using (3) and (4) dW = X.(y_hat - y).T

        self.dW = np.dot(self.train_x, (predicted_y - self.train_y).T) / self.sample_size
        self.dB = np.average(predicted_y - self.train_y)

    def __gradient_descent(self):
        self.weight -= self.learning_rate * self.dW.T
        self.bias -= self.learning_rate * self.dB.T

    def log_training_response(self):
        print("Train Accuracy : ", self.train_accuracy)
        plt.plot(self.cost_list)
        plt.ylabel("costs")
        plt.xlabel("iterations (per hundred)")
        plt.title("Learning Rate")
        plt.show()
        pass

    def analyze_learning_rate(self, learning_rates=[0.01, 0.001, 0.0001]):
        costs = list()
        models = {}
        model = LogisticRegression()
        for lr in learning_rates:
            model.fit(self.train_x, self.train_y, learning_rate=lr)
        pass


if __name__ == "__main__":
    vector_size = 128
    dataset_size = 10000
    test_size = 0.3

    test_size = int(dataset_size * test_size)
    train_size = dataset_size - test_size

    train_input = None
    test_input = None
    train_output = None
    test_output = None
    iterations = None


    def get_random_data():
        global train_input, test_input, train_output, test_output, iterations
        train_input = np.random.randint(1, 255, size=(vector_size, train_size)) / 255
        test_input = np.random.randint(1, 255, size=(vector_size, train_size))
        train_output = np.where(np.random.random((1, train_size)) > 0.5, 1, 0) / 255
        test_output = np.where(np.random.random((1, train_size)) > 0.5, 1, 0)
        iterations = 2000


    learning_rate = 0.05
    get_random_data()

    print(train_input)

    lr = LogisticRegression()
    lr.fit(train_input, train_output)
