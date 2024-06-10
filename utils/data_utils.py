"""
Lists all utility functions related to data extraction
"""
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
# pylint: disable=E0401
from constants import NUMBER_OF_NEIGHBORS, CLUSTERING_ALGORITHM, ALPHA_FOR_WEIGHTED_AVERAGE
from extract_data import get_raw_data_path


def read_data(data_path):
    """
    Read data from a CSV file using numpy.

    Parameters:
        data_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing the loaded data and its dimension.
    """
    with open(data_path, 'r', encoding='utf-8') as file:
        data = np.genfromtxt(file, delimiter=',')
    dimension = data.shape[1]
    return data, dimension


def get_data(data_path):
    """
    This function returns the raw data after extraction
    """
    data, _ = read_data(data_path)
    return data


def get_data_dimension(data_path):
    """
    This function returns the ground truth
    data after extraction
    """
    _, dimension = read_data(data_path)
    return dimension


def calculate_density():
    """
    Calculate nearest neighbors, density of each point, and density of its K neighbors.
    Returns:
        list: List containing density information for each data point.
    """
    density = []
    data = get_data(get_raw_data_path())
    nbrs = NearestNeighbors(n_neighbors=NUMBER_OF_NEIGHBORS,
                            algorithm=CLUSTERING_ALGORITHM).fit(data)
    distances, indices = nbrs.kneighbors(data)
    hash_map = {i: 0 for i in range(len(data))}
    labels = {i: 0 for i in range(len(data))}
    for i, value in enumerate(data):
        density.append([sum(distances[i]) / (NUMBER_OF_NEIGHBORS - 1), value,
                        max(distances[i]), indices[i][1:], i])
        hash_map[i] = density[i]
    return hash_map, density, labels


def sort_density():
    """
    Sort a list of densities based on the first element of each sublist.

    Returns:
        list: Sorted list of densities.
    """
    _, density_list, _ = calculate_density()
    density_list.sort(key=itemgetter(0), reverse=False)
    return density_list


def ewma_vectorized(values):
    """
    TODO: Explain this function
    """
    span = (2/ALPHA_FOR_WEIGHTED_AVERAGE) - 1
    data_frame = pd.DataFrame(values)
    return data_frame.ewm(span=span).mean().iloc[-1][0]
