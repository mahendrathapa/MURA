import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset

from src.utils.data_utils import get_image


class ValMuraDataset(Dataset):

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        # image_path = Path()
        image = get_image(row['image_path'])
        image = self.transform(image)

        label = torch.from_numpy(row['label']).float()

        data_set = {'image': row['image_path'], 'x': image, 'y': label}

        return data_set


class TrainMuraDataset(Dataset):

    def __init__(self, model_config, positive_label_data, negative_label_data,
                 transform=None):

        self.model_config = model_config
        self.positive_label_data = positive_label_data
        self.negative_label_data = negative_label_data
        self.transform = transform

    def __len__(self):
        min_len = min([len(self.positive_label_data),
                       len(self.negative_label_data)])
        return int(min_len * self.model_config.SAMPLING_RATIO)

    def __getitem__(self, idx):

        positive_row = self.positive_label_data.iloc[idx]
        positive_row_path = (
            Path(self.model_config.data_path) / positive_row['image_path']
        )
        positive_image = get_image(positive_row_path, self.transform)
        # positive_image = self.transform(positive_image)

        positive_label = torch.from_numpy(
            np.array([positive_row['label']])
        ).float()

        negative_row = self.negative_label_data.iloc[idx]
        negative_row_path = (
            Path(self.model_config.data_path) / negative_row['image_path']
        )
        negative_image = get_image(negative_row_path, self.transformgit )
        # negative_image = self.transform(negative_image)

        negative_label = torch.from_numpy(
            np.array([negative_row['label']])
        ).float()

        data_set = {'positive_x': positive_image, 'positive_y': positive_label,
                    'negative_x': negative_image, 'negative_y': negative_label}

        return data_set
