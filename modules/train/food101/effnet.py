import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchsummary import summary
from torch.cuda import amp
from tqdm import tqdm
import glob


import sys
import os
import pathlib

module_path = pathlib.Path(__file__, "..", "..", "..", "..").resolve()
if module_path not in sys.path:
    sys.path.append(str(module_path))

from utils.logger import get_logger
logger = get_logger("effnet train for food101")

from modules.dataset.food101.dataloader import create_dataloader
from modules.models.efficient_net import efficientnet_b0
from modules.train.train_helper import EarlyStopping
from utils.config import Config
from utils.seeder import fix_seed


SEED = 42
fix_seed(SEED)
TEST_KEY = "test"

def train():
    """
    EfficientNetの学習
    """
    cnf = Config.get_cnf("conf")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader(cnf)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
    }

    model = create_model(cnf)

    #loss
    criterion = nn.CrossEntropyLoss()

    _learning_rate = float(cnf.effnet.learning_rate)
    optimizer = optim.AdamW(model.parameters(), _learning_rate)

    # scheduler
    _schedule_patience = cnf.effnet.scheduler_patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", patience=_schedule_patience
    )

    early_stoping = train_process(cnf, model, dataloader, criterion, optimizer, scheduler)

    logger.info(f"val: best_epoch: {early_stoping.epoch()}")
    logger.info(f"val: best_acc: {early_stoping.acc()}")
    logger.info(f"val: best_loss: {early_stoping.loss()}")
    logger.info(f"val: best_model: {early_stoping.model_path()}")

    test_process(model, dataloader, criterion, early_stoping.model_path())

    logger.info("train fin!!!")

def create_model(cnf):
    """
    モデルの作成
    """
    _num_classes = cnf.food101.num_classes
    model = efficientnet_b0(input_ch=3, num_classes=_num_classes)
    return model

def test_process(model, dataloaders_dict, criterion, model_path):
    # モデルの準備
    model.load_state_dict(torch.load(model_path))
    torch.backends.cudnn.benchmark = True
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    epoch_loss = 0.
    epoch_corrects = 0.
    for inputs, labels in tqdm(dataloaders_dict[TEST_KEY]):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            if torch.cuda.is_available():
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            epoch_loss += loss.item() * inputs.size(0)  # lossの合計を計算
            epoch_corrects += torch.sum(preds == labels.data)

    epoch_loss = epoch_loss / len(dataloaders_dict[TEST_KEY].dataset)
    epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[TEST_KEY].dataset)
    epoch_acc = float(epoch_acc)

    logger.info(f"test: model: {model_path}")
    logger.info(f"test: acc: {epoch_acc}")
    logger.info(f"test: loss: {epoch_loss}")
    return True

def train_process(cnf, model, dataloaders_dict, criterion, optimizer, scheduler):
    logger.info(f"Log output dir: {os.getcwd()}")
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    scaler = None
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler() 
    
    logger.info(f"{device}で学習を行います。")
    model.to(device)

    # ネットワークがある程度固定ならば高速化
    torch.backends.cudnn.benchmark = True

    # Early stopping
    early_stopping = EarlyStopping(cnf.effnet.early_stopping)
    _early_stop = False

    # モデルの保存に関して
    model_w_dir = os.path.join(os.getcwd(), cnf.effnet.model_weight_dir)
    model_w_prefix = os.path.join(
        model_w_dir, cnf.effnet.model_weight_prefix)
    os.makedirs(model_w_dir, exist_ok=True)

    # 学習開始
    learning_rate = float(optimizer.param_groups[0]['lr'])
    for epoch in range(cnf.effnet.num_epochs):
        logger.info(f"Epoch {epoch + 1} / {cnf.effnet.num_epochs}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.
            epoch_corrects = 0.

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # optimizer を初期化
                optimizer.zero_grad()

                # forwardを計算
                with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        with amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        if torch.cuda.is_available():
                            scaler.scale(loss).backword()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を計算
                    epoch_corrects += torch.sum(preds == labels.data)
            
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)
            epoch_acc = float(epoch_acc)
            # lossを考慮して、learning rateを更新
            if phase == 'val':
                scheduler.step(epoch_loss)
                learning_rate = float(optimizer.param_groups[0]['lr'])

            # Early Stoping
            if phase == "val":
                _early_stop, best_model = early_stopping.check_loss(epoch_loss)

                # lossがbestに更新されたら保存。
                if best_model:
                    model_weight_path = model_w_prefix + \
                        '_' + str(epoch) + '.pth'
                    early_stopping.best_model_param(
                        epoch=epoch, model_path=model_weight_path, acc=epoch_acc)
                    torch.save(model.state_dict(), model_weight_path)
            
            logger.info(
                f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} lr: {learning_rate}")
    

        if _early_stop:
            logger.info(
                f"Early Stoping: {early_stopping.epoch()}, Loss: {early_stopping.loss()}, Acc: {early_stopping.acc()}")
            break

    return early_stopping

    
if __name__ == "__main__":
    train()