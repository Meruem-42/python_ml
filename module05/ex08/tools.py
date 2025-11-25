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


def loss_elem_(y, y_hat):
    """
        Description:
            Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
            y: has to be an numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
        Returns:
            J_elem: numpy.array, a array of dimension (number of the training examples, 1).
            None if there is a dimension matching problem.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) :
        return None
    if y.shape != y_hat.shape :
        return None
    if len(y.shape) != 1 :
        if len(y.shape) == 2 :
            if y.shape[1] != 1 :
                return None
        else :
            return None
    loss_elem = np.array([pow(elem_y_hat - elem_y, 2) for elem_y, elem_y_hat in zip(y, y_hat)])
    return loss_elem


def loss_(y, y_hat):
    """
        Description:
            Calculates the value of loss function.
        Args:
            y: has to be an numpy.array, a two-dimensional array of shape m * 1.
            y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
        Returns:
            J_value : has to be a float.
            None if there is a dimension matching problem.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) :
        return None
    if y.shape != y_hat.shape :
        return None
    if len(y.shape) != 1 :
        if len(y.shape) == 2 :
            if y.shape[1] != 1 :
                return None
        else :
            return None
    loss_array = loss_elem_(y, y_hat)
    sum_error = np.sum(loss_array)
    loss_result = sum_error / (2 * loss_array.shape[0])
    return loss_result
