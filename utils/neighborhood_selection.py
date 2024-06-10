import numpy as np
from collections import defaultdict
from utils import extract_data, constants, data_utils


def perform_pruning(data):
    """
    Function that implements the main pruning logic
    """
    row = len(data)
    dimension = len(data[0])
    neighbors = constants.NUMBER_OF_NEIGHBORS
    W = np.zeros((row, row))
    hash_map = {value: np.array([]) for value in range(len(data))}

    for i in range(row):
        D_i = np.array(data - data[i, :])
        distance = (D_i ** 2).sum(1)
        nearest_neighbor = np.argsort(distance)[1:(neighbors + 1)]
        D_nbrs = D_i[nearest_neighbor, :]
        Q = np.matmul(D_nbrs, D_nbrs.T)
        t = np.trace(Q)
        r = 0.001 * t
        if neighbors >= dimension:
            Q = Q + (r * np.identity(neighbors))
        w = np.linalg.solve(Q, np.ones(neighbors))
        w = w / sum(w)
        W[i, nearest_neighbor] = w

        temp = []
        for ele in w:
            temp.append(ele)
            temp.sort(reverse=True)
            hash_map[i] = np.array(temp)

    store_K = defaultdict()

    for j in range(len(data)):
        count = 0
        for i in range(1, len(hash_map[j])):
            if round((abs(hash_map[j][i - 1] - hash_map[j][i]) / abs(hash_map[j][i - 1])) * 100,
                     1) <= constants.MINIMUM_NEIGHBORHOOD_CONTRIBUTION:
                count += 1
        store_K[j] = count

    for key in store_K:
        store_K[key] += 1  # +1 because all neighbouring algorithms consider the datapoint itself to be a part
        # of its neighbour
    return store_K


def get_pruned_neighborhood_count():
    """
    Calculate or retrieve Local Linear Embedding (LLE) lookup dictionary for a dataset.

    Parameters:
        data (numpy.ndarray or list): Input dataset.
        K (int): Number of neighbors for LLE.

    Returns:
        dict: LLE lookup dictionary where keys are indices and values are LLE results.
    """
    pruned_neighborhood_count = {}
    data = data_utils.get_data(extract_data.get_raw_data_path())
    try:
        pruned_neighborhood_count = perform_pruning(data)
    except Exception as e:
        print(f"Exception occurred during LLE calculation: {e}")
        for i in range(len(data)):
            pruned_neighborhood_count[i] = constants.NUMBER_OF_NEIGHBORS

    return pruned_neighborhood_count
