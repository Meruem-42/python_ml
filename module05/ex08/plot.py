import matplotlib.pyplot as plt
import numpy as np
from tools import predict_, loss_

def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, one-dimensional array of size m.
            y: has to be an numpy.ndarray, one-dimensional array of size m.
            theta: has to be an numpy.ndarray, one-dimensional array of size 2.
        Returns:
            Nothing.
        Raises:
            This function should not raise any Exception.
    """

    plt.scatter(x, y, label="data")
    y_hat = predict_(x, theta)
    plt.plot(x, y_hat, color="red", label="prediction line")
    for xi, yi, y_hati in zip(x, y, y_hat) :
        plt.plot([xi, xi], [yi, y_hati], color="gray", linestyle="--")
    plt.legend()
    plt.title(f"Cost : {loss_(y, y_hat)}")
    plt.show()


x = np.arange(1,6)
y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
# Example 1:
theta1= np.array([18,-1])
plot_with_loss(x, y, theta1)

theta2 = np.array([14, 0])
plot_with_loss(x, y, theta2)

theta3 = np.array([12, 0.8])
plot_with_loss(x, y, theta3)