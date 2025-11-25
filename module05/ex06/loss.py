import numpy as np
import math as math
from tools import predict_

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

if __name__ == "__main__" :
    x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    print(x1.shape)
    y_hat1 = predict_(x1, theta1)
    print(y_hat1.shape)
    y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
    # Example 1:
    print(loss_elem_(y1, y_hat1))
    print(loss_(y1, y_hat1))

    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    theta2 = np.array(np.array([[0.], [1.]]))
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
    # Example 3:
    print(loss_(y2, y_hat2))
    print(loss_(y2, y2))