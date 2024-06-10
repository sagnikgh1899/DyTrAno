import utils.constants as constants
import os


def get_raw_data_path():
    return os.path.join('data', 'clustering', 'raw_data', f"{constants.DATASET_NAME}.csv")


def get_ground_truth_data_path():
    return os.path.join('data', 'clustering', 'ground_truth', f"{constants.DATASET_NAME}_gt.csv")
