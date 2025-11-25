import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from tools import predict_

def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a one-dimensional array of size m.
            y: has to be an numpy.array, a one-dimensional array of size m.
            theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
        Returns:
            Nothing.
        Raises:
            This function should not raise any Exceptions.
    """
    # y_hat = predict_(x, theta).reshape(-1)
    # print(y_hat)
    # fig = px.scatter(x=x, y=y, title="Plot mapping x and y and the linear prediction model of y")
    # fig.add_scatter(x=x, y=y_hat.T, mode="lines", name="Prediction line")
    # fig.show()

    y_hat = predict_(x, theta)
    plt.scatter(x, y, label="data")
    plt.plot(x, y_hat, color="red", label="prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__" :
    x = np.arange(1,6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
    # Example 1:
    theta1 = np.array([[4.5],[-0.2]])
    plot(x, y, theta1)

    # Example 2:
    theta2 = np.array([[-1.5],[2]])
    plot(x, y, theta2)

    # Example 3:
    theta3 = np.array([[3],[0.3]])
    plot(x, y, theta3)

