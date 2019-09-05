import torch
import numpy as np
import random
from PIL import Image

from torchvision import transforms as pytorch_transforms

from src.constants import Constants

random.seed(123456789)


def get_image(image_path, transform=None, unsqueeze_dim=1):

    image = Image.open(image_path).convert('L')

    resize_tf = pytorch_transforms.Resize(
            (Constants.IMAGE_SIZE, Constants.IMAGE_SIZE)
    )
    image = resize_tf(image)

    if transform is not None:
        for tr in transform:
            image = tr(image)

    image = np.array(image)
    image = image.astype('float')

    if Constants.NORMALIZE:
        image = image - np.mean()

    image = torch.from_numpy(image).float()

    if unsqueeze_dim:
        for _ in range(unsqueeze_dim):
            image = image.unsqueeze(0)

    return image


def get_transforms(transform):
    transform_dict = dict()
    transform_dict["flip"] = pytorch_transforms.RandomHorizontalFlip(p=1.0)
    transform_dict["jitter"] = pytorch_transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0, hue=0
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
