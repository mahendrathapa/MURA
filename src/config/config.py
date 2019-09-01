import os


class Config:
    ROOT_PATH = "./unet"
    OUTPUT_ROOT_PATH = os.path.join(ROOT_PATH, "out")
    DATA_ROOT_PATH = os.path.join(ROOT_PATH, "data")

    TRAIN_DATA_DIR = os.path.join(DATA_ROOT_PATH, "train_data")
    VAL_DATA_DIR = os.path.join(DATA_ROOT_PATH, "val_data")
    TEST_DATA_DIR = os.path.join(DATA_ROOT_PATH, "test_data")

    TRAIN_DATA_CSV = "train_data.csv"
    VAL_DATA_CSV = "val_data.csv"
    TEST_DATA_CSV = "test_data.csv"

    JSON_CONFIG = "config.json"
    JSON_INDENT = 4

    ACCEPTED_DATA_TYPES = ["/*.png", "/*.jpg"]
    IGNORE_FILES = ["edge", "annotated"]

    def __init__(self, train_batch_size, val_batch_size, test_batch_size,
                 train_num_workers, val_num_workers, test_num_workers,
                 train_shuffle=True, val_shuffle=True, test_shuffle=True,
                 learning_rate=0.0001, epoch=100, model_dump_gap=2,
                 cutoff_thersh=0.5):

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers

        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle

        self.learning_rate = learning_rate
        self.epoch = epoch
        self.model_dump_gap = model_dump_gap
        self.cutoff_thersh = cutoff_thersh


class LocalConfig(Config):
    def __init__(self):
        super().__init__(
            train_batch_size=1,
            test_batch_size=1,
            val_batch_size=1,
            train_num_workers=os.cpu_count(),
            val_num_workers=os.cpu_count(),
            test_num_workers=os.cpu_count()
        )


class ServerConfig(Config):
    def __init__(self):
        super().__init__(
            train_batch_size=1,
            test_batch_size=1,
            val_batch_size=1,
            train_num_workers=os.cpu_count(),
            val_num_workers=os.cpu_count(),
            test_num_workers=os.cpu_count()
        )