import numpy as np

def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
        Args:
            x: has to be an numpy.ndarray, a vector.
        Returns:
            x’ as a numpy.ndarray.
            None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
        Raises:
            This function shouldn’t raise any Exception.
    """
    if not isinstance(x, np.ndarray) :
        return None
    min_x = np.min(x)
    max_x = np.max(x)
    minmax_score = (x - min_x) / (max_x - min_x)
    return minmax_score


# Example 1:
X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
print(minmax(X))
# Example 2:
Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
print(minmax(Y))