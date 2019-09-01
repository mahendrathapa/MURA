import numpy as np
from PIL import Image
import torch


def get_image(image_path, transform=None):

    image = Image.open(image_path).convert('L')
    if transform is not None:
        image = transform(image)
    image = np.array(image)
    image = image.astype('float')
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)

    return image
