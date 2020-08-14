import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd


def init_xy(k):
    x = np.random.rand(200, 1) * 5
    y = init_y(x)

    draw_learning_curve(x, y, k)

    x_train, x_test = split_train_test(x)
    y_train = init_y(x_train)
    y_test = init_y(x_test)

    plt.plot(x_train[:, 0], y_train, "bo")
    plt.plot(x_test[:, 0], y_test, "co")

    x_train = add_dimension(x_train, k)
    x_test = add_dimension(x_test, k)

    return x_train, y_train, x_test, y_test


def draw_learning_curve(x, y, k):
    lr_model = LinearRegression(normalize=True)
    train_sizes, train_scores, test_scores = learning_curve(lr_model, x, y)
    plt.plot(train_sizes, sum(train_scores) / len(train_scores), "bo-", label="train scores")
    plt.plot(train_sizes, sum(test_scores) / len(test_scores), "co-", label="test scores")
    plt.title("learning curve, degree:" + str(k))
    plt.ylabel("loss")
    plt.xlabel("data size")
    plt.legend()
    plt.show()


def init_y(x):
    y = 2.554 * (x ** 4) - 20.756 * (x ** 3) + 42.445 * (x ** 2) - 36.358 * x + 20.121
    y = y + np.random.normal(scale=3, size=y.shape)
    y = y.reshape(-1, 1)
    return y


def split_train_test(x):
    x = x[np.random.permutation(x.shape[0])]
    split = int(x.shape[0] * 0.8)

    x_train = x[:split]
    x_test = x[split:]

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    return x_train, x_test


def add_dimension(x, k):
    x_pure = x
    for i in range(2, k + 1):
        x = np.hstack((x, x_pure ** k))
    return x


def init_x_line(k):
    x_line = np.arange(0, 5, 0.01).reshape(-1, 1)
    x_line = add_dimension(x_line, k)
    return x_line


def draw_predicted_line(x, y, k, line_style, line_label):
    lr_model = LinearRegression(normalize=True)
    lr_model.fit(x, y)

    x_line = init_x_line(k)
    y_line = lr_model.predict(x_line)

    h = lr_model.predict(x)
    MSE = np.mean((y - h) ** 2) / 2

    plt.plot(x_line[:, 0], y_line, line_style, label=line_label)
    plt.title(" degree : " + str(k))

    return MSE


MSE_trains = []
MSE_tests = []


def draw_for_degree(k):
    x_train, y_train, x_test, y_test = init_xy(k)
    MSE_train = draw_predicted_line(x_train, y_train, k, "b--", "train data")
    MSE_test = draw_predicted_line(x_test, y_test, k, "c--", "test data")
    plt.legend()
    plt.show()
    MSE_tests.append(MSE_test)
    MSE_trains.append(MSE_train)


# question 1 & 3
draw_for_degree(1)
draw_for_degree(2)
draw_for_degree(4)
draw_for_degree(8)
draw_for_degree(16)

# question 2
plt.plot([1, 2, 4, 8, 16], MSE_trains, "bo-", label="train data")
plt.plot([1, 2, 4, 8, 16], MSE_tests, "co-", label="test data")
plt.xlabel("degree")
plt.ylabel("MSE")
plt.legend()
plt.show()
