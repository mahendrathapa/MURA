from collections import OrderedDict


class Constants:
    DENSE_NET = {
            "densenet-169": (6, 12, 32, 32),
            "densenet-121": (6, 12, 24, 16)
    }
    DENSE_BLOCK_STRUCTURE = DENSE_NET["densenet-121"]

    IMAGE_SIZE = 320

    NORMALIZE = True

    TRANSFORMATIONS = ["flip", "jitter", "rotate", "compose"]

    COMPOSE_PROBABILITIES = OrderedDict({
        "flip": 0.5,
        "jitter": 0.5,
        "rotate": 0.5
    })
