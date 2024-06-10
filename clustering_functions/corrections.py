# TODO: This file needs work

from sklearn.neighbors import NearestNeighbors
from utils import constants
import math
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


def dist(p, q):
    """
    Calculate the Euclidean distance between two points p and q.

    Parameters:
    p (iterable): Coordinates of the first point.
    q (iterable): Coordinates of the second point.

    Returns:
    float: The Euclidean distance between points p and q.
    """
    return math.sqrt(sum((px - qx) ** 2 for px, qx in zip(p, q)))


def update_neighbors(data, index, k):
    """
        Update neighbors based on distance calculations.
        Args:
        - data: The dataset for which neighbors are calculated.
        - index: The index of the data point for which neighbors are updated.
        """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=constants.CLUSTERING_ALGORITHM).fit(data)
    distances, indices = nbrs.kneighbors(data)
    density_mod = [[sum(distances[index]) / (k - 1), data[index], max(distances[index]),
                    indices[index][1:], index]]
    return density_mod[0]


# Function corrects the wrong clustering
def Correcting_wrong_clustering(labels, cluster_centers, data1):
    cluster_labels = []
    for key in labels:
        cluster_labels.append(labels[key])

    Hash_Map_2 = dict()
    X_label = dict()

    for i in range(len(data1)):
        if labels[i] == -1:
            continue
        if labels[i] in Hash_Map_2:
            Hash_Map_2[labels[i]] += 1
            X_label[labels[i]] = X_label[labels[i]] + [data1[i].tolist()]
        else:
            Hash_Map_2[labels[i]] = 1
            X_label[labels[i]] = [data1[i].tolist()]

    for key in Hash_Map_2:
        temp = Hash_Map_2[key]
        for idx in range(len(cluster_centers)):
            if cluster_centers[idx][1] == key:
                break
        Hash_Map_2[key] = [temp, cluster_centers[idx][0]]

    # Function to find Intra Cluster Distance
    def IntraClusterMetric(X):
        Intra_Clus_Dist = 0
        for i in range(len(X)):
            Intra_Clus_Dist += (sum(euclidean_distances(X, [X[i]]))[0]) / (len(X) - 1)
        return Intra_Clus_Dist / (len(X) - 1)

    for key in X_label:
        X = X_label[key]
        Intra_metric = IntraClusterMetric(X)
        Hash_Map_2[key] = Hash_Map_2[key] + [Intra_metric]

    clusnum_card = sorted(Hash_Map_2.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
    cluster_center_new = []
    for ele in clusnum_card:
        cluster_center_new.append(np.array(ele[1][1]))

    ## Following print statements can be uncommented for better intuition
    # print(Hash_Map_2)
    # print("\nAfter Sorting ")
    # print(clusnum_card)
    # print("\nOnly cluster centers")
    # print(cluster_center_new)

    # print(len(cluster_centers))
    # print(len(cluster_center_new))
    if len(cluster_centers) > len(cluster_center_new):
        cluster_centers = cluster_center_new
    K1 = len(cluster_centers)
    nbrs = NearestNeighbors(n_neighbors=K1, algorithm='auto').fit(cluster_center_new)
    distances, indices = nbrs.kneighbors(cluster_center_new)

    # print(indices,"\n")
    # print(distances)

    def GetCutoffDistMod(cluster_centers):
        percent = 0.7
        i = 0
        j = 1
        Count = 0
        Sum = 0
        while i < len(cluster_centers) - 1:
            while j < len(cluster_centers):
                Sum += dist(cluster_centers[i], cluster_centers[j])
                j += 1
                Count += 1
            i += 1
            j = i + 1
        Count = max(Count, 1)
        return math.ceil(Sum / Count) * percent

    dist_val = GetCutoffDistMod(cluster_center_new)
    val2 = 20
    val3 = 15
    percent1 = 0.8
    percent2 = 0.16
    percent3 = 0.10
    percent4 = 0.02

    # Format of clus_to_change: [curr_clus, clus_to_be]
    clus_to_change = []
    for i in range(len(clusnum_card) - 1):
        # We are not considering till the last term since,
        # last term is the most dense and biggest among all
        # the other clusters. It cannot be merged with others

        j = 1
        cardinality_of_self = Hash_Map_2[clusnum_card[i][0]][0]
        Intra_Clus_Self = Hash_Map_2[clusnum_card[i][0]][2]
        cardinality_of_neighbor = Hash_Map_2[clusnum_card[indices[i][j]][0]][0]
        Intra_Clus_Neigh = Hash_Map_2[clusnum_card[indices[i][j]][0]][2]

        # print(cardinality_of_self, cardinality_of_neighbor)
        while cardinality_of_self >= cardinality_of_neighbor and j < len(clusnum_card) - 1:
            j += 1
            cardinality_of_neighbor = Hash_Map_2[clusnum_card[indices[i][j]][0]][0]
            Intra_Clus_Neigh = Hash_Map_2[clusnum_card[indices[i][j]][0]][2]

        # cardinality_of_self = Hash_Map_2[clusnum_card[i][0]]
        # cardinality_of_neighbor = Hash_Map_2[clusnum_card[indices[i][j]][0]][0]
        # print(cardinality_of_neighbor)

        ## Following print statements can be uncommented for better intuition
        # print("Cluster number", clusnum_card[i][0], "Cardinality ", Hash_Map_2[clusnum_card[i][0]][0])
        # print("Neighbor Cluster number", clusnum_card[indices[i][j]][0], "Cardinality ", Hash_Map_2[clusnum_card[indices[i][j]][0]][0])
        # print("Intra_Clus_Self", Intra_Clus_Self, "Intra_Clus_Neigh ", Intra_Clus_Neigh)
        # print("Intra Val Lower =", (Intra_Clus_Neigh - percent1*Intra_Clus_Neigh), "Intra Val Higher =", (Intra_Clus_Neigh + percent1*Intra_Clus_Neigh))
        # print("Percent1*CNeigh", int(percent1*cardinality_of_neighbor))
        # print("Percent2 =", (cardinality_of_self/cardinality_of_neighbor))
        # print("val3", val3)
        # print("Or condition", abs(cardinality_of_self - cardinality_of_neighbor), "val2", val2)
        # print("Distances between self and neighbor", int(round(distances[i][j],0)), "Dist_val", dist_val)

        # Using a boolean merge to keep a track if the self cluster is merged or not
        # Unless and until the cluster is merged, it is processed through every if condition
        merged = False

        # If cardinality of self and cardinality of neighbor are both less than a given threshold
        # where the threshold is very very small; say 10
        if not merged:
            if cardinality_of_self <= val2 and cardinality_of_neighbor <= val2:
                # Distance checking here isn't necessary
                curr_clus = clusnum_card[i][0]
                clus_to_be = clusnum_card[indices[i][j]][0]
                clus_to_change.append([curr_clus, clus_to_be])
                Hash_Map_2[clusnum_card[indices[i][j]][0]][0] += cardinality_of_self
                # print("Merged 1st condition")
                merged = True  # made True so that we don't need to process it over other if conditions

        # If cardinality of self is less than 16% of cardinality of neighbor
        if not merged:
            if (cardinality_of_self / cardinality_of_neighbor) <= percent2 and cardinality_of_self <= val3:
                if int(round(distances[i][j], 0)) <= dist_val:
                    curr_clus = clusnum_card[i][0]
                    clus_to_be = clusnum_card[indices[i][j]][0]
                    clus_to_change.append([curr_clus, clus_to_be])
                    Hash_Map_2[clusnum_card[indices[i][j]][0]][0] += cardinality_of_self
                    # print("Merged 2nd condition")
                    merged = True  # made True so that we don't need to process it over other if conditions

        # If Cardinality of self > val3 but is less than percent6 of the cardinality of neighbor
        if not merged:
            if val3 < cardinality_of_self <= val3 * 2.7 and cardinality_of_self / cardinality_of_neighbor <= percent3:
                Intra_Upper_Limit = Intra_Clus_Neigh + percent1 * Intra_Clus_Neigh
                Intra_Lower_Limit = (Intra_Clus_Neigh - percent1 * Intra_Clus_Neigh)
                if (Intra_Lower_Limit <= Intra_Clus_Self <= Intra_Upper_Limit and int(
                        round(distances[i][j], 0)) <= dist_val):
                    curr_clus = clusnum_card[i][0]
                    clus_to_be = clusnum_card[indices[i][j]][0]
                    clus_to_change.append([curr_clus, clus_to_be])
                    Hash_Map_2[clusnum_card[indices[i][j]][0]][0] += cardinality_of_self
                    # print("Merged 3rd condition")
                    merged = True  # made True so that we don't need to process it over other if conditions

        # If Cardinality of self is less than percent4 of the cardinality of neighbor; where percent4 ~ 2 percent
        if not merged:
            if cardinality_of_self / cardinality_of_neighbor <= percent4:
                # Here we don't need to check the distance and intra cluster density;
                # The self cluster is very very small compared to the neighbor cluster.
                # So just merge them
                curr_clus = clusnum_card[i][0]
                clus_to_be = clusnum_card[indices[i][j]][0]
                clus_to_change.append([curr_clus, clus_to_be])
                Hash_Map_2[clusnum_card[indices[i][j]][0]][0] += cardinality_of_self
                # print("Merged 4th condition")
                merged = True  # made True so that we don't need to process it over other if conditions

        if not merged:  # Some merges by this logic as well
            if (cardinality_of_self <= cardinality_of_neighbor and cardinality_of_self <= val3) or (
                    abs(cardinality_of_self - cardinality_of_neighbor) <= val2 and cardinality_of_self <= val3):
                Intra_Upper_Limit = Intra_Clus_Neigh + percent1 * Intra_Clus_Neigh
                Intra_Lower_Limit = (Intra_Clus_Neigh - percent1 * Intra_Clus_Neigh)
                if (Intra_Lower_Limit <= Intra_Clus_Self <= Intra_Upper_Limit and int(
                        round(distances[i][j], 0)) <= dist_val):
                    curr_clus = clusnum_card[i][0]
                    clus_to_be = clusnum_card[indices[i][j]][0]
                    clus_to_change.append([curr_clus, clus_to_be])
                    Hash_Map_2[clusnum_card[indices[i][j]][0]][0] += cardinality_of_self
                    # print("Merged 5th condition")
                    merged = True  # made True so that we don't need to process it over other if conditions

        # print("\n")
    print("\nChanged Clusters are: [initial cluster --> New Cluster]")
    print(clus_to_change)

    for key in labels:
        for item in clus_to_change:
            if labels[key] == item[0]:
                labels[key] = item[1]


# This returns cluster numbers in a non-missing sorted order
# Before this function many clusters may be removed in the previous step
# So, to keep everything ordered, this function makes sure all clusters have a number
# and the number starts from 1 all the way to n without any missing number in between
def get_count_in_label(labels):
    table = {value: 0 for value in labels}
    for ele in labels:
        table[ele] += 1
    key_list = table.keys()
    for ele in sorted(key_list, reverse=False):
        print(ele, ":", table[ele])


def get_label_list(hash_map):
    label_list = []
    for key in hash_map.keys():
        if hash_map[key] > 0:
            label_list.append(hash_map[key])
    return label_list


def find_missing_idx(hash_map):
    max_idx = float("-inf")
    for key in hash_map:
        if hash_map[key] > max_idx:
            max_idx = hash_map[key]
    min_idx = float("inf")
    for key in hash_map:
        if hash_map[key] < min_idx and hash_map[key] != -1:
            min_idx = hash_map[key]
    label_list = get_label_list(hash_map)
    missing_idx = []
    count = 1
    while count <= max_idx:
        if count not in label_list:
            missing_idx.append(count)
        count += 1
    return missing_idx


def correct_the_idx(hash_map):
    missing_idx = find_missing_idx(hash_map)

    # Can be uncommented for better intuition
    # print(missing_idx)
    while missing_idx:
        for i in missing_idx:
            for key in hash_map:
                if hash_map[key] == i + 1:
                    hash_map[key] = hash_map[key] - 1
        missing_idx = find_missing_idx(hash_map)
