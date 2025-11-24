import numpy as np

def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a one-dimensional array of size m.
            theta: has to be an numpy.ndarray, a one-dimensional array of size 2.
        Returns:
            y_hat as a numpy.ndarray, a one-dimensional array of size m.
            None if x or theta are empty numpy.ndarray.
            None if x or theta dimensions are not appropriate.
        Raises:
            This function should not raise any Exception.
    """
    x_shape = x.shape
    theta_shape = theta.shape
    print(x_shape)
    print(theta_shape)
    if len(x_shape) != 1 :
        return None
    if theta_shape[0] != 2 or len(theta_shape) != 1:
        return None
    y_hat = np.array([theta[0] + theta[1] * elem for elem in x])
    return y_hat



if __name__ == "__main__" :
    x = np.arange(1,6)
    # Example 1:
    theta1 = np.array([5, 0])
    print(simple_predict(x, theta1))

    theta2 = np.array([0, 1])
    print(simple_predict(x, theta2))

    theta3 = np.array([5, 3])
    print(simple_predict(x, theta3))

    theta4 = np.array([-3, 1])
    print(type(simple_predict(x, theta4)))

    theta4 = np.array([[-3, 1], [1, 2]])
    print(simple_predict(x, theta4))