import torch
import numpy as np
from PIL import Image

from torchvision import transforms as pytorch_transforms

from src.constants import Constants


def get_image(image_path, transform=None, unsqueeze_dim=1):

    image = Image.open(image_path).convert('L')

    resize_tf = pytorch_transforms.Resize(
            (Constants.IMAGE_SIZE, Constants.IMAGE_SIZE)
    )
    image = resize_tf(image)

    if transform is not None:
        image = transform(image)

    image = np.array(image)
    image = image.astype('float')
    image = torch.from_numpy(image).float()

    if unsqueeze_dim:
        for _ in range(unsqueeze_dim):
            image = image.unsqueeze(0)

    return image
