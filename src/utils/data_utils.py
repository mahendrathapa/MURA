import torch
import numpy as np
from PIL import Image


def get_image(image_path, transform=None, add_dim=False):

    image = Image.open(image_path).convert('L')
    if transform is not None:
        image = transform(image)
    image = np.array(image)
    image = image.astype('float')
    image = torch.from_numpy(image).float()

    if add_dim:
        image = image.unsqueeze(0).unsqueeze(0)

    return image
