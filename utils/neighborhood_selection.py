"""
Contains the functions that perform optimal
neighborhood pruning
"""
from collections import defaultdict
import numpy as np
# pylint: disable=E0401
from utils import extract_data, constants, data_utils


# pylint: disable=R0914
def perform_pruning(data):
    """
    Function that implements the main pruning logic
    """
    row = len(data)
    dimension = len(data[0])
    neighbors = constants.NUMBER_OF_NEIGHBORS
    w_val = np.zeros((row, row))
    hash_map = {value: np.array([]) for value in range(len(data))}

    for i in range(row):
        d_i = np.array(data - data[i, :])
        distance = (d_i ** 2).sum(1)
        nearest_neighbor = np.argsort(distance)[1:(neighbors + 1)]
        d_nbrs = d_i[nearest_neighbor, :]
        q_val = np.matmul(d_nbrs, d_nbrs.T)
        t_val = np.trace(q_val)
        r_val = 0.001 * t_val
        if neighbors >= dimension:
            q_val = q_val + (r_val * np.identity(neighbors))
        weights = np.linalg.solve(q_val, np.ones(neighbors))
        weights = weights / sum(weights)
        w_val[i, nearest_neighbor] = weights

        temp = []
        for ele in weights:
            temp.append(ele)
            temp.sort(reverse=True)
            hash_map[i] = np.array(temp)

    store_k = defaultdict()

    for j in range(len(data)):
        count = 0
        for i in range(1, len(hash_map[j])):
            if round((abs(hash_map[j][i - 1] - hash_map[j][i]) / abs(hash_map[j][i - 1])) * 100,
                     1) <= constants.MINIMUM_NEIGHBORHOOD_CONTRIBUTION:
                count += 1
        store_k[j] = count

    for key in store_k:
        store_k[key] += 1  # +1 because all neighbouring algorithms consider
        # the datapoint itself to be a part of its neighbour
    return store_k


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
    except ValueError as exception:
        print(f"ValueError occurred during LLE calculation: {exception}")
        for i in range(len(data)):
            pruned_neighborhood_count[i] = constants.NUMBER_OF_NEIGHBORS

    return pruned_neighborhood_count
