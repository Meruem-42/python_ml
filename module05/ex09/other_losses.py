import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from tools import TinyStatistician

def mse_(y, y_hat):
    """
        Description:
            Calculate the MSE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be a numpy.array, a two-dimensional vector of shape m * 1.
        Returns:
            mse: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
    """
    if y.shape != y_hat.shape :
        return None
    mse = np.sum(pow(y_hat - y, 2) / y.shape[0])
    return mse



def rmse_(y, y_hat):
    """
        Description:
            Calculate the RMSE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
        Returns:
            rmse: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
    """
    if y.shape != y_hat.shape :
        return None
    rmse = sqrt(np.sum(pow(y_hat - y,2) / y.shape[0]))
    return rmse



def mae_(y, y_hat):
    """
        Description:
            Calculate the MAE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
        Returns:
            mae: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
    """
    if y.shape != y_hat.shape :
        return None
    mae = np.sum(abs(y_hat - y)) / y.shape[0]
    return mae


def r2score_(y, y_hat):
    """
        Description:
            Calculate the R2score between the predicted output and the output.
        Args:
            y: has to be a numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
        Returns:
            r2score: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
    """
    if y.shape != y_hat.shape :
        return None

    y_mean = y.mean()
    r2score = 1 - (np.sum(pow(y_hat - y, 2)) / np.sum(pow(y - y_mean, 2)))
    return r2score


# Example 1:
x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
# Mean-squared-error
## your implementation
print(mse_(x,y))
print(rmse_(x,y))
print(mae_(x,y))
print(r2score_(x,y))