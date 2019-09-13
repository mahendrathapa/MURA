from collections import OrderedDict


class Constants:
    PRETRAINED = False

    DENSE_NET = {
        "densenet-169": (6, 12, 32, 32),
        "densenet-121": (6, 12, 24, 16)
    }
    DENSE_BLOCK_STRUCTURE = DENSE_NET["densenet-121"]

    IMAGE_SIZE = 320

    NORMALIZE = False if PRETRAINED else True

    GLOBAL_MEAN = 52.60

    LR_DECAY = {"no_of_epochs": 4, "factor": 0.1}

    TRANSFORMATIONS = ["flip", "jitter", "rotate", "compose"]

    COMPOSE_PROBABILITIES = OrderedDict({
        "flip": 0.5,
        "jitter": 0.5,
        "rotate": 0.5
    })

    # For CAM
    FINAL_CONV_LAYER = "relu"
    FC_LAYER = "fc1"

    # For prediction
    RUN_ID = ""
    MODEL_NAME = ""
