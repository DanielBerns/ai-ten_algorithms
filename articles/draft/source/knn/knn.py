import numpy as np

class KNN:
    """
    This implementation defines a KNN class 
    that takes a value k as a parameter 
    during initialization. 
    The fit method is used to train the model 
    on a training set X and corresponding labels y. 
    
    The predict method takes a set of 
    data points X and returns predicted labels 
    for each data point.
    
    In this implementation, we take advantage 
    of numpy's ability to perform operations 
    on arrays element-wise to vectorize 
    the calculation of distances between X 
    and self.X_train.

    First, we reshape X and self.X_train using 
    np.newaxis to create a new axis so that we 
    can perform broadcasting, which allows 
    numpy to perform operations on arrays 
    with different shapes.
    
    Then, we calculate the squared differences 
    between each data point in X and each 
    training example in self.X_train 
    using the ** operator, sum over the 
    feature dimension using np.sum, and take 
    the square root using np.sqrt to obtain 
    the distances.
    
    Next, we use np.argsort to obtain the 
    indices of the k smallest distances 
    along the second axis (axis 1), corresponding 
    to the k nearest neighbors for each 
    data point in X.
    
    We then extract the labels of the 
    k nearest neighbors from self.y_train 
    using numpy's array indexing syntax, 
    and calculate the mode of the labels 
    for each data point using np.apply_along_axis 
    with a lambda function that applies np.bincount 
    and argmax along the first axis (axis 1).
    
    Finally, we return the predicted labels y_pred, 
    which is a numpy array with shape (n_samples,) 
    containing the predicted label for each 
    data point in X.
    """
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = 
        np.sqrt(
            np.sum((X[:, np.newaxis, :] - self.X_train) ** 2, 
                   axis=2))
        neighbors = np.argsort(distances, axis=1)[:, :self.k]
        labels = self.y_train[neighbors]
        y_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=1, 
            arr=labels)
        return y_pred

 
