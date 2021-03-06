import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

def init_xy(n, k):
    x = np.random.rand(n, 1) * 20
    y = 2.358 * x - 3.121
    y = y + np.random.normal(scale=3, size=y.shape)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x_pure = x
    for i in range(2, k + 1):
        x = np.hstack((x, x_pure ** k))
    return x, y

def draw_line(n, line_style, line_label):  # line_style is an string defines color and style of line
    lr_model = LinearRegression(normalize=True)
    MSE = []
    for k in range(1, 11):
        x, y = init_xy(n, k)
        lr_model.fit(x, y)
        h = lr_model.predict(x)
        MSE.append(np.mean((y - h) ** 2) / 2)

    x_line = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.plot(x_line, MSE, line_style, label=line_label)


draw_line(5, "m-", "5 data")
draw_line(10, "b-", "10 data")
draw_line(25, "c-", "25 data")
draw_line(100, "r-", "100 data")
plt.xlabel("degree")
plt.ylabel("MSE")
plt.legend()
plt.show()
