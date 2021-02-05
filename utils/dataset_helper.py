import random
import os

from utils.logger import get_logger
logger = get_logger()

def split_dataset(data_paths, labels, val_rate):
    indexes = list(range(len(data_paths)))
    random.shuffle(indexes)

    train_indexes = indexes[:int(len(data_paths)*val_rate)]
    val_indexes = indexes[int(len(data_paths)*val_rate):]

    val_paths = [data_paths[ind] for ind in train_indexes]
    val_labels = [labels[ind] for ind in train_indexes]
    train_paths = [data_paths[ind] for ind in val_indexes]
    train_labels = [labels[ind] for ind in val_indexes]

    return train_paths, train_labels, val_paths, val_labels

def path_check(data_paths):
    for ind, data_path in enumerate(data_paths):
        if not os.path.exists(data_path):
            logger.error(f"{ind}: {data_path} is not exsits")
            return False
        
    return True