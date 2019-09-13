import pandas as pd

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from src.constants import Constants
from src.data.mura_dataset import ValMuraDataset, TrainMuraDataset
from src.utils import data_utils


class MuraDataSetLoader:

    def __init__(self, config):
        self.config = config

    def load_train_data(self):

        train_positive_data = pd.read_csv(self.config.TRAIN_POS_DATA_CSV)

        train_negative_data = pd.read_csv(self.config.TRAIN_NEG_DATA_CSV)

        data_set = TrainMuraDataset(
                self.config,
                train_positive_data,
                train_negative_data,
        )
        transformations = Constants.TRANSFORMATIONS
        print(f"Applied transformations: {transformations}")
        if transformations is not None:
            dataset_list = list()
            dataset_list.append(data_set)

            for transform in transformations:
                dataset_list.append(
                        TrainMuraDataset(
                            self.config,
                            train_positive_data,
                            train_negative_data,
                            transform=data_utils.get_transforms(transform)
                        )
                )
            concat_data_set = ConcatDataset(dataset_list)
            print(f"Initial total data: {data_set.__len__()}")
            print(f"Final total data: {concat_data_set.__len__()}")
            return DataLoader(
                    concat_data_set,
                    batch_size=self.config.TRAIN_BATCH_SIZE,
                    shuffle=self.config.TRAIN_SHUFFLE,
                    num_workers=self.config.TRAIN_WORKERS
            )

        else:
            return DataLoader(
                data_set,
                batch_size=self.config.TRAIN_BATCH_SIZE,
                shuffle=self.config.TRAIN_SHUFFLE,
                num_workers=self.config.TRAIN_WORKERS
            )

    def load_val_data(self):

        val_data = pd.read_csv(self.config.VAL_DATA_CSV)

        return DataLoader(
            ValMuraDataset(self.config, val_data),
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=self.config.VAL_SHUFFLE,
            num_workers=self.config.VAL_WORKERS
        )
