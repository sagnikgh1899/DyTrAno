from collections import deque
from utils import data_utils, neighborhood_selection, extract_data, constants
from clustering_functions import corrections, bfs_clustering, filtering
from tqdm import tqdm
from results import all_data_points_info_post_filtering
from visualizations import display_clustering


def perform_bfs_clustering_helper(data):
    """
    Update information for the Parent node
    #   0                         1         2        3                    4
    # Distance-Density Value; Datapoint; Radius; k-nearest neighbors; idx_number

    """
    start_at_data_idx = 0
    cluster_num = 0
    cluster_centers = []
    hold_cluster_val = 0
    density = data_utils.sort_density()
    pruned_neighborhood_count = neighborhood_selection.get_pruned_neighborhood_count()
    Store_Radius_Density = dict()
    Store_Parents = dict()
    Roots = dict()
    Tree_Structure = dict()

    Hash_Map, _, labels = data_utils.calculate_density()

    for data_idx in tqdm(range(start_at_data_idx, len(data))):

        # if (data_idx + 1) % 100 == 0:
        #     print("**************************Computed {} Datapoints**************************".format(data_idx + 1))
        list_of_parents_queue = deque()
        list_of_parents_queue.appendleft([])

        if labels[density[data_idx][-1]] == 0:
            cluster_num += 1
            labels[density[data_idx][-1]] = cluster_num
            can_form_cluster = [False]
            K_mod = pruned_neighborhood_count[density[data_idx][-1]]
            density[data_idx] = corrections.update_neighbors(data, density[data_idx][-1], K_mod)

            bfs_clustering.bfs_based_clustering(density[data_idx], Hash_Map, data, list_of_parents_queue, labels,
                                                cluster_num, can_form_cluster, Tree_Structure, Roots,
                                                Store_Radius_Density, Store_Parents)
            # print(Tree_Structure)

            # # Visualization: For lower dimensions
            # #
            # Dimension = data_utils.get_data_dimension(extract_data.get_raw_data_path())
            # if Dimension <= 2:
            #     plt.figure(figsize=(8, 8))
            #     for i in range(len(data)):
            #         if labels[i] == cluster_num:
            #             plt.scatter(data[i][0], data[i][1], c='pink')
            #         elif labels[i] == -1:
            #             plt.scatter(data[i][0], data[i][1], c='yellow')
            #         else:
            #             plt.scatter(data[i][0], data[i][1], c='blue')
            #     plt.scatter(density[data_idx][1][0], density[data_idx][1][1], c='red', s=200, marker='+')
            #     plt.show()
            # # """

            # If the last Parent Node cannot form a cluster of its own then it is a part of a cluster with only 1
            # datapoint which means " a possible Anomaly" then change its label to -1 and reduce the cluster num by 1
            if not can_form_cluster[0]:
                labels[density[data_idx][-1]] = -1
                cluster_num = hold_cluster_val
            else:
                cluster_centers.append([density[data_idx][1].tolist(), cluster_num])
                hold_cluster_val += 1

    corrections.Correcting_wrong_clustering(labels, cluster_centers, data)
    corrections.correct_the_idx(labels)
    filtering.OutlierPostProcessing(pruned_neighborhood_count, data, labels,
                                    constants.PERCENTAGE_FOR_FILTERING_POST_PROCESSING,
                                    Store_Radius_Density, Tree_Structure, Store_Parents)
    return labels, data, cluster_num


def perform_bfs_clustering(displayStats):
    print("Clustering in Progress...")
    labels, data, cluster_num = perform_bfs_clustering_helper(data_utils.get_data(extract_data.get_raw_data_path()))
    print("Clustering Completed.")

    if displayStats:
        print("\nDisplaying Inlier-Outlier Statistics...")
        all_data_points_info_post_filtering.get_outlier_inlier_info_post_filtering(data, labels)
    else:
        print("\nIgnoring Inlier-Outlier Statistics")

    dimension = data_utils.get_data_dimension(extract_data.get_raw_data_path())
    if dimension <= 2:
        print("\nDisplaying Clustering Results...")
        display_clustering.display_clustering(dimension, labels, data, cluster_num)
