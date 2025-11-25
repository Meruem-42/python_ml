import numpy as np
from tools import add_intercept

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
        return None
    if theta.shape[0] != 2 or len(theta.shape) != 2 or theta.shape[1] != 1:
        return None
    x_matrix = add_intercept(x)
    return (x_matrix.dot(theta))

if __name__ == "__main__" :
    x = np.arange(1,6)

    print("Example 1\n")
    theta1 = np.array([[5], [0]])
    print(predict_(x, theta1))

    print("\nExample 2\n")
    theta2 = np.array([[0], [1]])
    print(predict_(x, theta2))

    print("\nExample 3\n")
    theta3 = np.array([[5], [3]])
    print(predict_(x, theta3))

    print("\nExample 4\n")
    theta4 = np.array([[-3], [1]])
    print(theta4.shape)
    print(predict_(x, theta4))

    print("\nExample ERROR\n")
    theta4 = np.array([1,2])
    print(predict_(x, theta4))
