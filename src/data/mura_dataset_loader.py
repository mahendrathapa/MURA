import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader
from src.data.mura_dataset import ValMuraDataset, TrainMuraDataset


class MuraDataSetLoader:

    def __init__(self, config):
        self.config = config

    def load_train_data(self):

        train_positive_data = pd.read_csv(
            Path(self.config.data_path) / "train_positive_data.csv")

        train_negative_data = pd.read_csv(
            Path(self.config.data_path) / "train_negative_data.csv")

        return DataLoader(TrainMuraDataset(train_positive_data,
                                           train_negative_data),
                          batch_size=int(
                              self.config.TRAIN_BATCH_SIZE / 2),
                          shuffle=self.config.TRAIN_SHUFFLE,
                          num_workers=self.config.TRAIN_WORKERS)

    def load_val_data(self):

        val_data = pd.read_csv(
            Path(self.config.data_path) / "val_data.csv")

        return DataLoader(ValMuraDataset(val_data),
                          batch_size=int(
                              self.config.VAL_BATCH_SIZE / 2),
                          shuffle=self.config.VAL_SHUFFLE,
                          num_workers=self.config.VAL_WORKERS)
