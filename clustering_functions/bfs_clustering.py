from collections import deque
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils import data_utils, constants


def bfs_based_clustering(parent_node, hash_map, dataset, list_of_parents_queue, labels, cluster_num,
                         can_form_cluster, tree_structure, roots, store_radius_density, store_parents):
    """
    BFS-based approach for clustering algorithm.

    Args:
    - parent_node: The parent node to start BFS from.
    - hash_map: A hashmap containing nodes and their attributes.
    - dataset: The dataset used for clustering.
    - list_of_parents_queue: Queue of parent nodes.
    - labels: List of labels for nodes.
    - threshold: Threshold value for density criterion.
    - cluster_num: Current cluster number.
    - can_form_cluster: Flag indicating if a cluster can be formed.
    - tree_structure: Structure to store tree relationships.
    - roots: Roots of the trees.
    - store_radius_density: Store adaptive radius and density.
    - store_parents: Store parents' information.
    """

    roots[parent_node[4]] = parent_node[1]
    store_radius_density[parent_node[4]] = [parent_node[2], parent_node[0]]
    queue = deque()
    queue.appendleft(parent_node)
    while queue:
        parent_node = queue.pop()
        density_parent = parent_node[0]
        radius = parent_node[2]
        child_idx_array = parent_node[3]
        list_of_parents = list_of_parents_queue.pop() + [density_parent]
        store_parents[parent_node[4]] = list_of_parents
        child_array = []
        child_datapoint_array = []

        for child_idx in child_idx_array:
            child_array.append(hash_map[child_idx])
            child_datapoint_array.append(hash_map[child_idx][1])

        for i in range(len(child_idx_array)):
            child = child_array[i]
            child_idx = child[-1]
            child_datapoint = child[1]

            if labels[child_idx] > 0:
                continue

            neigh = NearestNeighbors(radius=radius)
            neigh.fit(dataset)
            rng = neigh.radius_neighbors([child_datapoint])

            if len(rng[0][0]) == 0:
                density_child = 0
            else:
                density_child = sum(rng[0][0]) / len(rng[0][0])

            data_parent = np.array(list_of_parents)
            weighted_density = data_utils.ewma_vectorized(data_parent)

            numerator = abs(weighted_density - density_child)
            denominator = weighted_density

            if denominator != 0 and numerator / denominator <= constants.THRESHOLD_FOR_INCLUSION_IN_TREE:
                labels[child_idx] = cluster_num
                list_of_parents_queue.appendleft(list_of_parents)

                child[0] = density_child
                child[2] = max(rng[0][0]) if len(rng[0][0]) > 0 else 0
                child[3] = rng[1][0]

                list_of_parents_1 = list_of_parents + [density_child]
                data_parent_1 = np.array(list_of_parents_1)
                weighted_density_1 = data_utils.ewma_vectorized(data_parent_1)
                store_radius_density[child[4]] = [child[2], weighted_density_1]

                if parent_node[4] not in tree_structure:
                    tree_structure[parent_node[4]] = [child[4]]
                else:
                    tree_structure[parent_node[4]].append(child[4])

                queue.appendleft(child)

                hash_map.update({child[-1]: child})

                can_form_cluster[0] = True

            else:
                labels[child_idx] = -1
