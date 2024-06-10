# TODO: This file too needs work

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from utils import constants, data_utils
import numpy as np


def OutlierPostProcessing(pruned_neighborhood_count, data, labels, percent, Store_Radius_Density,
                          Tree_Structure, Store_Parents):
    # Select a predetermined value of K
    # Find the K nearest Neighbors of an outlier
    # For each of those neighbors:
    #     Find their Intracluster distance with the same K value
    #     If the Intracluster distance of outlier is within +-5% of the intracuster distance of the neighbor,
    #     and the neighbor point is not an outlier,
    #     then consider the outlier to be a part of the smae cluster that the neighbor point belongs.

    for i in range(len(data)):
        if labels[i] == -1:
            # Predicted Outlier, so find its K nearest neighbors
            Opt_K = pruned_neighborhood_count[i]
            nbrs = NearestNeighbors(n_neighbors=Opt_K, algorithm=constants.CLUSTERING_ALGORITHM).fit(data)
            _, indices = nbrs.kneighbors(data)

            PredOutlierNNidx = indices[i]
            PredOutlierNN = []
            for idx in PredOutlierNNidx:
                PredOutlierNN.append(data[idx])
            # Find the intracluster distance for the Predicted Outlier
            PredOutlierICD = IntraClusterDistance(PredOutlierNN)

            # For each of the outlier's K neighbors
            for idx in PredOutlierNNidx[1:]:
                # Check if the neighbor is an outlier or inlier
                if labels[idx] != -1:
                    NNeighboridx = indices[idx]
                    NNeighbor = []
                    for idx1 in NNeighboridx:
                        NNeighbor.append(data[idx1])

                    # Find each neighbor's IntraClusterDistance(ICD)
                    NNICD = IntraClusterDistance(NNeighbor)

                    # Check if outlier's ICD is about +-percent% of neighbor's ICD
                    if (NNICD - percent * NNICD) <= PredOutlierICD <= (NNICD + percent * NNICD):
                        # Change the label of the Predicted Outlier 'i' to the
                        # label of the 1st Nearest Neighbor which satifies the condition
                        labels[i] = labels[idx]
                        if idx not in Tree_Structure:
                            Tree_Structure[idx] = [i]
                        else:
                            Tree_Structure[idx].append(i)

                        # Update its Store_Radius_Density Value
                        Radius = Store_Radius_Density[idx][0]
                        neigh = NearestNeighbors(radius=Radius)
                        neigh.fit(data)
                        rng = neigh.radius_neighbors([data[i]])
                        # Check if there is noneighbour within the radius
                        # If yes then the density of the incoming datapoint is 0
                        if len(rng[0][0]) == 0:
                            Density_Parent = 0
                        else:
                            Density_Parent = sum(rng[0][0]) / len(rng[0][0])

                        list_of_parents = Store_Parents[idx] + [Density_Parent]
                        data_parent = np.array(list_of_parents)
                        weighted_density = data_utils.ewma_vectorized(data_parent)
                        if len(rng[0][0]) == 0:
                            Radius_new = 0
                        else:
                            Radius_new = max(rng[0][0])
                        Store_Radius_Density[i] = [Radius_new, weighted_density]
                        Store_Parents[i] = list_of_parents
                        break


def IntraClusterDistance(X):
    Intra_Clus_Dist = 0
    for i in range(len(X)):
        Intra_Clus_Dist += (sum(euclidean_distances(X, [X[i]]))[0]) / (len(X) - 1)
    return Intra_Clus_Dist / (len(X) - 1)
