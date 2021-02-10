import os
from torch.utils import data
import cv2
import random

from utils.json_helper import json_open
from utils.dataset_helper import split_dataset, path_check



class Food101Dataset(data.Dataset):
    def __init__(self, img_paths, labels, transform) -> None:
        self._transform = transform
        self._img_paths = img_paths
        self._labels = labels
        
    def __len__(self) -> int:
        return len(self._img_paths)
    
    def __getitem__(self, index: int):
        img_path = self._img_paths[index]
        label = self._labels[index]

        img = cv2.imread(img_path)
        if img is None:
            ValueError(f"Can not read {img_path} in dataloader")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        # albumentations transform
        img = self._transform(image=img)["image"]
        
        return img, label

class Food101ContrastiveDataset(data.Dataset):
    def __init__(self, img_paths, labels, transform, double_aug_rate=0.2) -> None:
        self._transform = transform
        self._img_paths = img_paths
        self._labels = labels
        self._img_dict_by_keys = self._label_dict_set()
        self._double_aug_rate = double_aug_rate
        

    def __len__(self) -> int:
        return len(self._img_paths)
    
    def __getitem__(self, index: int):
        img_path = self._img_paths[index]
        label = self._labels[index]

        img1 = cv2.imread(img_path)
        if img1 is None:
            ValueError(f"Can not read {img_path} in dataloader")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        if random.random() > self._double_aug_rate:
            img_path = random.choice(self._img_dict_by_keys[label])
        
        img2 = cv2.imread(img_path)
        if img2 is None:
            ValueError(f"Can not read {img_path} in dataloader")
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) 
        
        
        # albumentations transform
        img1 = self._transform(image=img1)["image"]
        img2 = self._transform(image=img2)["image"]
        
        return img1, img2, label
    
    def _label_dict_set(self):
        data_dict = {}
        for img_path, label in zip(self._img_paths, self._labels):
            if label not in data_dict:
                data_dict[label] = []
            
            data_dict[label].append(img_path)
        
        return data_dict


def create_dataset(cnf):
    """
    データセットのパスとラベルを返す
    @input:
        cnf
    @output:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
    """
    _food101 = cnf.food101
    train_paths = json_open(_food101.train_json)
    test_paths = json_open(_food101.test_json)

    # label
    label_names = list(train_paths.keys())

    # dataset path
    _dataset_prefix = _food101.dataset_dir
    train_paths, train_labels = dataset_info_parser(train_paths, label_names, prefix=_dataset_prefix, ext=_food101.img_ext)
    test_paths, test_labels = dataset_info_parser(test_paths, label_names, prefix=_dataset_prefix, ext=_food101.img_ext)

    ok = path_check(train_paths)
    if not ok:
        FileExistsError(f"train dataset path is not exsits.")

    ok = path_check(test_paths)
    if not ok:
        FileExistsError(f"test dataset path is not exsits.")


    # dataset split
    _val_rate = cnf.dataloader.val_rate
    train_paths, train_labels, val_paths, val_labels = split_dataset(train_paths, train_labels, _val_rate)

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def dataset_info_parser(train_paths, label_names, prefix="", ext=".png"):
    """
    train_paths: jsonから読み取った辞書
    label_names: ラベル情報
    """
    data_paths = []
    labels = []

    for ind, label in enumerate(label_names):
        _data_paths = train_paths[label]
        _data_paths = [os.path.join(prefix, _data_path + ext) for _data_path in _data_paths]
        _labels = [ind] * len(_data_paths)

        data_paths.extend(_data_paths)
        labels.extend(_labels)
    
    return data_paths, labels

