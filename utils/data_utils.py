import numpy as np
import pandas as pd
from utils import constants, extract_data
from sklearn.neighbors import NearestNeighbors
from operator import itemgetter


def read_data(data_path):
    """
    Read data from a CSV file using numpy.

    Parameters:
        data_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing the loaded data and its dimension.
    """
    with open(data_path, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')
    dimension = data.shape[1]
    return data, dimension


def get_data(data_path):
    data, _ = read_data(data_path)
    return data


def get_data_dimension(data_path):
    _, dimension = read_data(data_path)
    return dimension


def calculate_density():
    """
    Calculate nearest neighbors, density of each point, and density of its K neighbors.
    Returns:
        list: List containing density information for each data point.
    """
    density = []
    data = get_data(extract_data.get_raw_data_path())
    nbrs = NearestNeighbors(n_neighbors=constants.NUMBER_OF_NEIGHBORS, algorithm=constants.CLUSTERING_ALGORITHM).fit(data)
    distances, indices = nbrs.kneighbors(data)
    Hash_Map = {i: 0 for i in range(len(data))}
    labels = {i: 0 for i in range(len(data))}
    for i in range(len(data)):
        density.append([sum(distances[i]) / (constants.NUMBER_OF_NEIGHBORS - 1), data[i], max(distances[i]), indices[i][1:], i])
        Hash_Map[i] = density[i]
    return Hash_Map, density, labels


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
    span = (2/constants.ALPHA_FOR_WEIGHTED_AVERAGE) - 1
    df = pd.DataFrame(values)
    return df.ewm(span=span).mean().iloc[-1][0]


