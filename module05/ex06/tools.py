import numpy as np

def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
        Args:
            x: has to be a numpy.array. x can be a one-dimensional (m * 1) or two-dimensional (m * n) array.
        Returns:
            X, a numpy.array of dimension m * (n + 1).
            None if x is not a numpy.array.
            None if x is an empty numpy.array.
        Raises:
            This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) :
        return None
    if x.shape[0] == 0 :
        return None
    array = np.ones(x.shape[0])
    intercept_array = np.column_stack((array, x))
    return intercept_array


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a one-dimensional array of size m.
            theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
        Returns:
            y_hat as a numpy.array, a two-dimensional array of shape m * 1.
            None if x and/or theta are not numpy.array.
            None if x or theta are empty numpy.array.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exceptions.
    """
    if not isinstance(x, np.ndarray) :
        return None
    if not isinstance(theta, np.ndarray) :
        return None
    if x.shape[0] == 0 :
        return None
    if len(x.shape) != 1 :
        if len(x.shape) == 2 :
            if x.shape[1] != 1 :
                return  None
        else :
            return None
    if theta.shape[0] != 2 or len(theta.shape) != 2 or theta.shape[1] != 1:
        return None
    x_matrix = add_intercept(x)
    return (x_matrix.dot(theta))