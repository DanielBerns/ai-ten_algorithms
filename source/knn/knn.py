import numpy as np

class KNN:
    """
    This implementation defines a KNN class that takes a value k as a parameter during initialization. The fit method is used to train the model on a training set X and corresponding labels y. The predict method takes a set of data points X and returns predicted labels for each data point.

    In the predict method, the distances between each data point in X and 
    all training examples in X_train are calculated using the Euclidean distance formula. 
    The indices of the k nearest neighbors are then obtained using np.argsort to sort 
    the distances in ascending order and select the indices of the k smallest distances. 
    The corresponding labels of the nearest neighbors are then extracted from y_train. 
    Finally, the predicted label for each data point in X is determined by taking 
    the mode of the nearest neighbor labels using np.bincount and argmax.

    Note that this implementation assumes that X and X_train are numpy arrays with 
    the same number of columns/features. Additionally, the labels in y and the predicted labels 
    returned by the predict method are assumed to be integer values representing the class labels.
    """
    def __init__(self):
        self.X_train = None
        self.y_train = None

def fit(self, X_train: np.array, y_train: np.array) -> None:
        x_h, x_w = X.shape
        y_h, y_w = Y.shape
        try:
            assert x_w == y_w
        except AssertionError:
            print("x_w != y_w")
        try:
            assert y_h == 1
        except AssertionError:
            print("y_h != 1")
            
    def predict(self, X: np.array, k: int) -> np.array:
        """
        In this implementation, we take advantage of numpy's ability to perform operations 
        on arrays element-wise to vectorize the calculation of distances between X and self.X_train.
        First, we reshape X and self.X_train using np.newaxis to create a new axis so that 
        we can perform broadcasting, which allows numpy to perform operations on arrays 
        with different shapes.
        Then, we calculate the squared differences between each data point in X and each 
        training example in self.X_train using the ** operator, 
        sum over the feature dimension using np.sum, 
        and take the square root using np.sqrt to obtain the distances.

        Next, we use np.argsort to obtain the indices of the k smallest distances along 
        the second axis (axis 1), corresponding to the k nearest neighbors for each data point in X.

        We then extract the labels of the k nearest neighbors from self.y_train using numpy's array 
        indexing syntax, and calculate the mode of the labels for each data point using np.apply_along_axis 
        with a lambda function that applies np.bincount and argmax along the first axis (axis 1).

        Finally, we return the predicted labels y_pred, which is a numpy array with 
        shape (n_samples,) containing the predicted label for each data point in X.

        We then extract the labels of the k nearest neighbors from self.y_train using numpy's array 
        indexing syntax, and calculate the mode of the labels for each data point using 
        np.apply_along_axis with a lambda function that applies np.bincount and argmax 
        along the first axis (axis 1).

        Finally, we return the predicted labels y_pred, which is a numpy array 
        with shape (n_samples,) containing the predicted label for each data point in X.        
        """
        distances = np.sqrt(np.sum((X[:, np.newaxis, :] - self.X_train) ** 2, axis=2))
        neighbors = np.argsort(distances, axis=1)[:, :k]
        labels = self.y_train[neighbors]
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=labels)
        return y_pred
