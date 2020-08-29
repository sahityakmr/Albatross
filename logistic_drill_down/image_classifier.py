import numpy as np
import matplotlib.pyplot as plt


def initialize_input_weights(training_samples):
    return np.zeros((training_samples, 1)), 0


def activation_function(output):
    return 1 / (1 + np.exp(-output))


def propagate(train_input, train_output, weight, intercept):
    prediction = activation_function(np.dot(weight.T, train_input) + intercept)
    cost = -1 * np.mean(train_output * np.log(prediction) + (1 - train_output) * np.log(1 - prediction))

    dw = np.dot(train_input, (prediction - train_output).T) / train_input.shape[1]
    db = np.average(prediction - train_output)
    return {"dw": dw, "db": db}, np.squeeze(cost)


def train_model(train_input, train_output, weight, intercept, iterations, learning_rate):
    costs = list()
    gradient_descent = None

    for i in range(iterations):
        gradient_descent, cost = propagate(train_input, train_output, weight, intercept)

        dw = gradient_descent["dw"]
        db = gradient_descent["db"]

        weight -= learning_rate * dw
        intercept -= learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            print("Cost after %i iterations : %f" % (i, cost))

    return weight, intercept, gradient_descent, costs


def predict(weight, intercept, input):
    input_size = input.shape[1]
    prediction = activation_function(np.dot(weight.T, input) + intercept)
    return np.where(prediction > 0.5, 1., 0.)


def plot_learning_rate():
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()


def learning_rate_analysis(train_input, train_output, test_input, test_output):
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = build_model(train_input, train_output, test_input, test_output, num_iterations=1500,
                                     learning_rate=i)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


def build_model(train_input, train_output, test_input, test_output, iterations, learning_rate):
    training_samples = train_input.shape[0]

    weight, intercept = initialize_input_weights(training_samples)
    weight, intercept, gradient_descent, costs = train_model(train_input, train_output, weight, intercept, iterations,
                                                             learning_rate)

    prediction_test = predict(weight, intercept, test_input)
    prediction_train = predict(weight, intercept, train_input)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(prediction_train - train_output)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(prediction_test - test_output)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


if __name__ == "__main__":
    vector_size = 128
    dataset_size = 10000
    test_size = 0.3

    test_size = int(dataset_size * test_size)
    train_size = dataset_size - test_size

    train_input = np.random.random((vector_size, train_size))
    test_input = np.random.random((vector_size, train_size))
    train_output = np.where(np.random.random((1, train_size)) > 0.5, 1, 0)
    test_output = np.where(np.random.random((1, train_size)) > 0.5, 1, 0)
    iterations = 10000
    learning_rate = 0.3
    log_cost = True

    build_model(train_input, train_output, test_input, test_output, iterations, learning_rate)
