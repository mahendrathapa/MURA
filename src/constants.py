from collections import OrderedDict


class Constants:
    DENSE_BLOCK_STRUCTURE = (6, 12, 32, 32)

    IMAGE_SIZE = 320

    NORMALIZE = False

    TRANSFORMATIONS = ["flip", "jitter", "rotate", "compose"]

    COMPOSE_PROBABILITIES = OrderedDict({
        "flip": 0.5,
        "jitter": 0,
        "rotate": 0.5
    })
