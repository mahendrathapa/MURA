import pandas as pd

from torch.utils.data import DataLoader

from src.constants import Constants
from src.data.mura_dataset import ValMuraDataset, TrainMuraDataset


class MuraDataSetLoader:

    def __init__(self, config):
        self.config = config

    def load_train_data(self, transformation=None):

        train_positive_data = pd.read_csv(self.config.TRAIN_POS_DATA_CSV)

        train_negative_data = pd.read_csv(self.config.TRAIN_NEG_DATA_CSV)
        if transformation is not None:
            pass
            # TO DO: Add different transformation based on list of transformation arguments
        else:
            data_set = TrainMuraDataset(
                    self.config,
                    train_positive_data,
                    train_negative_data,
            )
            return DataLoader(
                data_set,
                batch_size=self.config.TRAIN_BATCH_SIZE,
                shuffle=self.config.TRAIN_SHUFFLE,
                num_workers=self.config.TRAIN_WORKERS
            )

    def load_val_data(self, transformation=None):

        val_data = pd.read_csv(self.config.VAL_DATA_CSV)

        return DataLoader(
            ValMuraDataset(self.config, val_data),
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=self.config.VAL_SHUFFLE,
            num_workers=self.config.VAL_WORKERS
        )
