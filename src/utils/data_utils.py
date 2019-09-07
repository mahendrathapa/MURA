import torch
import numpy as np
import pandas as pd
import random
from PIL import Image
from pathlib import Path

from torchvision import transforms as pytorch_transforms

from src.constants import Constants
from src.config.config import ServerConfig

random.seed(123456789)


def get_image(image_path, transforms=None, unsqueeze_dim=1):

    image = Image.open(image_path).convert('L')

    resize_tf = pytorch_transforms.Resize(
            (Constants.IMAGE_SIZE, Constants.IMAGE_SIZE)
    )
    image = resize_tf(image)

    if transforms is not None:
        for transform in transforms:
            image = transform(image)

    image = np.array(image)
    image = image.astype('float')

    if Constants.NORMALIZE:
        image = image - Constants.GLOBAL_MEAN

    image = torch.from_numpy(image).float()

    if unsqueeze_dim:
        for _ in range(unsqueeze_dim):
            image = image.unsqueeze(0)

    return image


def get_transforms(transform):
    transform_dict = dict()
    transform_dict["flip"] = pytorch_transforms.RandomHorizontalFlip(p=1.0)
    transform_dict["jitter"] = pytorch_transforms.ColorJitter(
                                brightness=0.5,
                                contrast=0.5,
                                saturation=0,
                                hue=0
    )
    transform_dict["rotate"] = pytorch_transforms.RandomRotation(
                                (-10, 10),
                                resample=False,
                                expand=False,
                                center=None
    )
    transform_dict["compose"] = list()
    for transform_type, prob in Constants.COMPOSE_PROBABILITIES.items():
        p = random.random()
        if p >= prob:
            transform_dict["compose"].append(transform_dict[transform_type])
    try:
        tr = transform_dict[transform]
        return tr if transform == "compose" else [tr]
    except KeyError:
        print("Required transformation is not available")

def get_global_mean(verbose=False):
    config  = ServerConfig()
    train_positive_data = pd.read_csv(config.TRAIN_POS_DATA_CSV)
    train_negative_data = pd.read_csv(config.TRAIN_NEG_DATA_CSV)
    data = pd.concat([train_positive_data, train_negative_data], ignore_index=True)
    global_sum_list = list()
    # images = []
    total = 0
    print("Total Data: ", len(data))
    for index, rows in data.iterrows():
        image_path = (Path(config.data_path) / rows['image_path'])
        image = Image.open(image_path).convert('L')
        image = np.array(image)
        if verbose:
            print("Image: {}, Mean: {}".format(image_path, image.mean()))
        global_sum_list.append(image.sum())
        total += image.size
        # images.append(image.reshape(-1))
    global_mean = np.array(global_sum_list).sum() / total
    # images = np.concatenate(images, 1)
    # print(images.mean())
    # print(images.std())
    # print(global_mean)
    return global_mean
