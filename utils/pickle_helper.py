import pickle
import os


def load_pickle(data_path:str):
    """
    @input
        data_path
    """
    if not os.path.exists(data_path):
        return None
    with open(data_path, "rb") as f:
        return pickle.load(f)


def write_pickle(data, data_path:str):
    """
    @input
        data, data_path
    """
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
    return data_path
    