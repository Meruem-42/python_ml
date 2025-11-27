import numpy as np

class MyLinearRegression() :
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def loss_elem_(self, y, y_hat):
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

    def loss_(self, y, y_hat):
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
        loss_array = self.loss_elem_(y, y_hat)
        sum_error = np.sum(loss_array)
        loss_result = sum_error / (2 * loss_array.shape[0])
        return loss_result


    def predict_(self, x):
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
        if not isinstance(self.thetas, np.ndarray) :
            return None
        if x.shape[0] == 0 :
            return None
        if len(x.shape) != 1 :
            if len(x.shape) == 2 :
                if x.shape[1] != 1 :
                    return  None
            else :
                return None
        if self.thetas.shape[0] != 2 :
            return None
        if len(self.thetas.shape) != 1 :
            if len(self.thetas.shape) == 2 :
                if self.thetas.shape[1] != 1 :
                    return  None
            else :
                return None
        ones_array = np.ones(len(x))
        x_matrix = np.column_stack((ones_array, x))
        return (x_matrix.dot(self.thetas))


    def fit_(self, x, y):
        """
            Description:
                Fits the model to the training dataset contained in x and y.
            Args:
                x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
                y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
                theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
                alpha: has to be a float, the learning rate
                max_iter: has to be an int, the number of iterations done during the gradient descent
            Returns:
                new_theta: numpy.ndarray, a vector of dimension 2 * 1.
                None if there is a matching dimension problem.
            Raises:
                This function should not raise any Exception.
        """
        new_theta = np.array(self.thetas, dtype =float)
        ones_array = np.ones(x.shape[0])
        full_array = np.column_stack((ones_array, x))
        for i in range(0, self.max_iter) :
            gradient_vector = full_array.T @ (full_array @ new_theta - y) / len(x)
            new_theta -= self.alpha * gradient_vector
        self.thetas = new_theta