import numpy as np
import matplotlib.pyplot as plt
from neural_network.driver import load_data
import pickle


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
        self.orig_train_x = None

    def __initialize_parameters(self):
        self.weight = np.zeros((1, self.feature_size))
        self.bias = 0

    def __set_hyper_parameters(self, learning_rate, iterations, log_cost, log_step):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.log_cost = log_cost
        self.log_step = log_step

    def fit(self, train_x, train_y, learning_rate=0.05, iterations=2000, log_cost=True, log_step=100):
        self.feature_size = train_x.shape[FEATURE_SIZE_INDEX]
        self.sample_size = train_x.shape[SAMPLE_SIZE_INDEX]
        self.orig_train_x = train_x
        self.train_x = LogisticRegression.standardize_input(train_x)
        self.train_y = train_y

        self.__set_hyper_parameters(learning_rate, iterations, log_cost, log_step)
        self.__initialize_parameters()
        self.__train()

    def predict(self, test_x, test_y):
        test_x = LogisticRegression.standardize_input(test_x)
        predict_y = np.where(sigmoid_activation(np.dot(self.weight, test_x) + self.bias) > 0.5, 1, 0)
        print("Accuracy : " + str(np.sum((predict_y == test_y) / test_x.shape[1])))
        return predict_y

    def __train(self):
        self.cost_list = list()

        for i in range(self.iterations):
            self.__propagate()
            self.__gradient_descent()

            if i % self.log_step == 0:
                self.cost_list.append(self.curr_cost)
                if self.log_cost:
                    print("Cost after %i iterations : %f" % (i, self.curr_cost))
        self.train_accuracy = 100 - np.mean(np.abs(self.predict(self.orig_train_x, self.train_y) - self.train_y) * 100)

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

    def analyze_learning_rate(self, learning_rates=(0.01, 0.001, 0.0001)):
        models = {}
        for i in range(len(learning_rates)):
            lr = learning_rates[i]
            model = LogisticRegression()
            model.fit(self.orig_train_x, self.train_y, learning_rate=lr)
            models[str(i)] = model

        for i in range(len(learning_rates)):
            model = models[str(i)]
            plt.plot(np.squeeze(model.cost_list), label=str(model.learning_rate))
            plt.ylabel('cost')
            plt.xlabel('iterations (hundreds)')
            legend = plt.legend(loc='upper right', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
        plt.show()

    pass

    @staticmethod
    def standardize_input(x):
        return x / 255


if __name__ == "__main__":
    train_x, train_y, test_x, test_y, classes = load_data()
    file_name = 'single_neuron.sav'


    def train_n_save():
        model: LogisticRegression = LogisticRegression()
        model.fit(train_x, train_y, learning_rate=0.005)
        pickle.dump(model, open(file_name, 'wb'))


    # train_n_save()
    model: LogisticRegression = pickle.load(open(file_name, 'rb'))
    model.analyze_learning_rate((0.01, 0.001, 0.0001, 0.005))
