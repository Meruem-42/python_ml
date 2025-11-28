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
    if theta.shape[0] != 2 :
        return None
    if len(theta.shape) != 1 :
        if len(theta.shape) == 2 :
            if theta.shape[1] != 1 :
                return  None
        else :
            return None
    x_matrix = add_intercept(x)
    return (x_matrix.dot(theta))


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.arrays, with a for-loop.
        The three arrays must have compatible shapes.
        Args:
            x: has to be an numpy.array, a vector of shape m * 1.
            y: has to be an numpy.array, a vector of shape m * 1.
            theta: has to be an numpy.array, a 2 * 1 vector.
        Return:
            The gradient as a numpy.array, a vector of shape 2 * 1.
            None if x, y, or theta are empty numpy.array.
            None if x, y and theta do not have compatible shapes.
            None if x, y or theta is not of the expected type.
        Raises:
            This function should not raise any Exception.
    """
    if not all(isinstance(obj, np.ndarray) for obj in [x, y, theta]):
        return None
    if any(obj.size == 0 for obj in [x, y, theta]):
        return None    
    ones_array = np.ones(x.shape[0])
    full_array = np.column_stack((ones_array, x))
    gradient_vector = full_array.T @ (full_array @ theta - y) / len(x)
    return gradient_vector
