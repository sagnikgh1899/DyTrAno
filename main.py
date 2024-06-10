from utils import neighborhood_selection, constants
from clustering_functions import perform_clustering
import argparse
import warnings


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Add arguments below
    parser.add_argument('--numNeigh', type=int, required=True, help='Number of Neighbors Estimate')
    parser.add_argument('--datasetName', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--displayStats', type=str, default=True, help='Display inlier-outlier stats at the end')

    arguments = parser.parse_args()

    constants.NUMBER_OF_NEIGHBORS = arguments.numNeigh
    constants.DATASET_NAME = arguments.datasetName
    constants.DISPLAY_DATA_POINT_STATS = arguments.displayStats


def main():
    """
    This is the main function
    """
    warnings.filterwarnings('ignore')
    parse_arguments()
    perform_clustering.perform_bfs_clustering(constants.DISPLAY_DATA_POINT_STATS)


if __name__ == "__main__":
    main()
