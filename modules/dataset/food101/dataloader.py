from torch.utils import data

from modules.dataset.food101.dataset import Food101Dataset, create_dataset
from modules.dataset.food101.transformes import generate_train_transformes, generate_test_transforms

from utils.logger import get_logger
logger = get_logger()

def create_dataloader(cnf):
    _dataloader = cnf.dataloader
    batch_size = _dataloader.batch_size
    num_workers = _dataloader.num_workers
    pin_memory = _dataloader.pin_memory
    
    img_height = _dataloader.target_size_h
    img_width = _dataloader.target_size_w
    target_size = (img_height, img_width)

    logger.info(f"create dataset loader: batch_size:{batch_size}, num_workers: {num_workers}, pin_memory: {pin_memory}")

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = create_dataset(cnf)

    train_transformer = generate_train_transformes(target_size=target_size)
    test_transformer = generate_test_transforms(target_size=target_size)

    train_dataset = Food101Dataset(train_paths, train_labels, train_transformer)
    val_dataset = Food101Dataset(val_paths, val_labels, test_transformer)
    test_dataset = Food101Dataset(test_paths, test_labels, test_transformer)

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    logger.info(f"train dataloader: {len(train_dataloader)}")
    logger.info(f"val dataloader: {len(val_dataloader)}")
    logger.info(f"test dataloader: {len(test_dataloader)}")

    return train_dataloader, val_dataloader, test_dataloader

    

