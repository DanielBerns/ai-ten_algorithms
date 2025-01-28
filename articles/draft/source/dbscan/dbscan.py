import numpy as np

def DBSCAN(X, eps, MinPts):
    """
    Implementation of DBSCAN algorithm using numpy.
    
    Args:
        X (np.array): A numpy array of shape (n_samples, n_features) containing the data points.
        eps (float): The maximum distance between two points for them to be considered as neighbors.
        MinPts (int): The minimum number of points required to form a dense region.
        
    Returns:
        labels (np.array): A numpy array of shape (n_samples,) containing the cluster labels for each data point.
    """
    n_samples, n_features = X.shape
    
    # Initialize all points as unvisited
    visited = np.zeros(n_samples, dtype=bool)
    
    # Initialize all points as noise
    labels = np.full(n_samples, -1, dtype=int)
    
    # Initialize cluster label
    cluster = 0
    
    for i in range(n_samples):
        # If the point has already been visited, skip it
        if visited[i]:
            continue
        
        visited[i] = True
        
        # Find all points within eps distance of the current point
        neighbors = np.linalg.norm(X - X[i], axis=1) < eps
        
        # If there are less than MinPts neighbors, mark the point as noise
        if np.sum(neighbors) < MinPts:
            labels[i] = -1
        else:
            # Expand the cluster
            cluster += 1
            labels[i] = cluster
            expand_cluster(X, visited, labels, neighbors, cluster, eps, MinPts)
    
    return labels

def expand_cluster(X, visited, labels, neighbors, cluster, eps, MinPts):
    """
    Expands the current cluster by adding all reachable points to the cluster.
    
    Args:
        X (np.array): A numpy array of shape (n_samples, n_features) containing the data points.
        visited (np.array): A numpy array of shape (n_samples,) containing boolean values indicating whether a point has been visited or not.
        labels (np.array): A numpy array of shape (n_samples,) containing the cluster labels for each data point.
        neighbors (np.array): A numpy array of shape (n_samples,) containing boolean values indicating whether a point is a neighbor or not.
        cluster (int): The current cluster label.
        eps (float): The maximum distance between two points for them to be considered as neighbors.
        MinPts (int): The minimum number of points required to form a dense region.
    """
    # Find all points within eps distance of the current point
    seeds = np.where(neighbors)[0]
    
    # Iterate over all neighboring points
    for seed in seeds:
        # If the point has already been visited, skip it
        if visited[seed]:
            continue
        
        visited[seed] = True
        
        # Find all points within eps distance of the current point
        neighbors = np.linalg.norm(X - X[seed], axis=1) < eps
        
        # If the point has at least MinPts neighbors, add it to the current cluster
        if np.sum(neighbors) >= MinPts:
            labels[seed] = cluster
            expand_cluster(X, visited, labels, neighbors, cluster, eps, MinPts)
        else:
            # Otherwise, mark the point as noise
            labels[seed] = -1
 


import numpy as np

def vectorized_dbscan(X, eps, min_samples):
    """
    Implementation of DBSCAN algorithm using numpy with vectorized operations.
    
    Args:
        X (np.array): A numpy array of shape (n_samples, n_features) containing the data points.
        eps (float): The maximum distance between two points for them to be considered as neighbors.
        min_samples (int): The minimum number of points required to form a dense region.
        
    Returns:
        labels (np.array): A numpy array of shape (n_samples,) containing the cluster labels for each data point.
    """
    n_samples = X.shape[0]
    
    # Calculate pairwise distances between all points
    distances = np.linalg.norm(X[:, None] - X, axis=-1)
    
    # Identify all neighbors within eps distance
    neighbors = distances < eps
    
    # Mark all points as noise by default
    labels = -1 * np.ones(n_samples, dtype=int)
    
    # Initialize the current cluster label
    current_cluster = 0
    
    # Iterate over all points
    for i in range(n_samples):
        # If the point has already been assigned to a cluster, skip it
        if labels[i] != -1:
            continue
        
        # Find all points within eps distance of the current point
        core_points = np.sum(neighbors[i], axis=0) >= min_samples
        
        # If the point is not a core point, mark it as noise and continue
        if not core_points:
            continue
        
        # Expand the cluster starting from the current point
        current_cluster += 1
        labels[i] = current_cluster
        
        # Identify all points in the same cluster as the current point
        cluster_points = neighbors[i]
        
        # Recursively expand the cluster
        while True:
            # Find all new points within eps distance of the current cluster
            new_neighbors = np.sum(neighbors[cluster_points], axis=0) > 0
            
            # Add the new points to the current cluster
            new_cluster_points = np.logical_and(new_neighbors, ~cluster_points)
            cluster_points = np.logical_or(cluster_points, new_cluster_points)
            
            # If no new points were added to the cluster, we're done
            if not np.any(new_cluster_points):
                break
            
            # Assign the new points to the current cluster
            labels[new_cluster_points] = current_cluster
    
    return labels
