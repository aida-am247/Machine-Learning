import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

def init_xy(n , k):
    x = np.random.rand(n, 1) * 20
    y = 2.358 * x - 3.121
    y = y + np.random.normal(scale=3, size=y.shape)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    plt.plot(x[:, 0], y, "ro")
    x_pure = x
    for i in range(2, k + 1):
        x = np.hstack((x, x_pure ** k))
    return x, y

def init_x_line(k):
    x_line_pure = np.arange(0, 20, 0.1).reshape(-1, 1)
    x_line = x_line_pure
    for i in range(2, k+1):
        x_line = np.hstack((x_line, x_line_pure ** k))
    return x_line

def draw_predicted_line(n, k):
    lr_model = LinearRegression(normalize=True)
    x, y = init_xy(n, k)
    lr_model.fit(x, y)
    print(lr_model.coef_, lr_model.intercept_)

    x_line = init_x_line(k)
    y_line = lr_model.predict(x_line)

    plt.plot(x_line[:, 0], y_line, "b--")
    plt.title("number of data : " + str(n) + ", degree : " + str(k))
    plt.show()

def draw_4lines(k):
    draw_predicted_line(5, k)
    draw_predicted_line(10, k)
    draw_predicted_line(25, k)
    draw_predicted_line(100, k)

draw_4lines(1) # question1
draw_4lines(4) # question2
draw_4lines(16) # question3
