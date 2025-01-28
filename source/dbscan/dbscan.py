import numpy as np


def dbscan(data, eps, min_pts):
    """
    DBSCAN clustering algorithm.

    Args:
        data: A numpy array of shape (n_samples, n_features).
        eps: The maximum distance between two samples to be considered as neighbors.
        min_pts: The minimum number of points required to form a dense region.

    Returns:
        A numpy array of shape (n_samples,) containing the cluster labels.
    """

    def expand_cluster(data, labels, point_index, cluster_id, eps, min_pts):
        """
        Expands a cluster by recursively adding neighbors.

        Args:
            data: A numpy array of shape (n_samples, n_features).
            labels: A numpy array of shape (n_samples,) containing the cluster labels.
            point_index: The index of the current point.
            cluster_id: The ID of the current cluster.
            eps: The maximum distance between two samples to be considered as neighbors.
            min_pts: The minimum number of points required to form a dense region.
        """

        neighbors = range(len(data))
        neighbors = [i for i in neighbors if np.linalg.norm(data[point_index] - data[i]) <= eps]

        if len(neighbors) < min_pts:
            labels[point_index] = -1  # Noise
        else:
            labels[point_index] = cluster_id

            for neighbor_index in neighbors:
                if labels[neighbor_index] == 0:
                    labels[neighbor_index] = cluster_id
                    expand_cluster(data, labels, neighbor_index, cluster_id, eps, min_pts)

    n_samples = data.shape[0]
    labels = np.zeros(n_samples)

    cluster_id = 1
    for i in range(n_samples):
        if labels[i] == 0:
            expand_cluster(data, labels, i, cluster_id, eps, min_pts)
            cluster_id += 1

    return labels

