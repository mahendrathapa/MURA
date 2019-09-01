import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from src.constants import Constants
from src.data.mura_dataset import ValMuraDataset, TrainMuraDataset


class MuraDataSetLoader:

    def __init__(self, config):
        self.config = config

    def load_train_data(self):

        int_dir = "MURA-v1.1"

        train_positive_data = pd.read_csv(
            Path(self.config.data_path) / int_dir / "train_positive_data.csv"
        )

        train_negative_data = pd.read_csv(
            Path(self.config.data_path) / int_dir / "train_negative_data.csv"
        )

        data_transform = transforms.Compose([
            transforms.Resize((Constants.IMAGE_SIZE, Constants.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # )
        ])

        data_set = TrainMuraDataset(
                self.config,
                train_positive_data,
                train_negative_data,
                transform=data_transform
        )
        return DataLoader(
            data_set,
            batch_size=self.config.TRAIN_BATCH_SIZE,
            shuffle=self.config.TRAIN_SHUFFLE,
            num_workers=self.config.TRAIN_WORKERS
        )

    def load_val_data(self):

        val_data = pd.read_csv(
            Path(self.config.data_path) / "MURA-v1.1" / "val_data.csv"
        )

        data_transform = transforms.Compose([
            transforms.Resize((Constants.IMAGE_SIZE, Constants.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # )
        ])

        return DataLoader(
            ValMuraDataset(val_data, data_transform),
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=self.config.VAL_SHUFFLE,
            num_workers=self.config.VAL_WORKERS
        )
