import argparse
import glob
import json
import os
import time

from src.architecture.densenet import Densenet
from src.config.config import LocalConfig, ServerConfig
from src.data.mura_dataset_loader import MuraDataSetLoader
from src.model.DenseNet_model import DenseNetModel


def main():
    network = Densenet()
    global_config = get_args()

    if global_config.env == "server":
        global_config.model_config = ServerConfig()
    elif global_config.env == "local":
        global_config.model_config = LocalConfig()

    if global_config.model_config_json:
        print("Hyperparameters configuration loading from {}".format(
            global_config.model_config_json))
        with open(global_config.model_config_json, "r") as f:
            global_config.model_config.__dict__ = json.load(f)
        print("Hyperparameters configuration loaded from {}".format(
            global_config.model_config_json))

    model_config = global_config.model_config
    dataset_loader = MuraDataSetLoader(model_config)

    if global_config.mode == "full":
        train_data = dataset_loader.load_train_data()

        val_data = dataset_loader.load_val_data()

        # test_data = dataset_loader.load_test_data()

        model = DenseNetModel(
            network, global_config, train_data, val_data
        )
        model.train()

    elif global_config.mode == "train":
        train_data = dataset_loader.load_train_data()
        val_data = dataset_loader.load_val_data()

        model = DenseNetModel(network, global_config, train_data, val_data)
        model.train()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        default="local",
        choices=["server", "local"],
        type=str
    )

    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "train"],
        type=str
    )

    parser.add_argument(
        "--model_config_json",
        default="",
        type=str
    )

    parser.add_argument(
        "--run_id",
        default=str(int(time.time())),
        type=str
    )

    parser.add_argument(
        "--model_checkpoint",
        default='',
        type=str,
        help='path to latest model checkpoint (default: none)'
    )

    parser.add_argument(
        "--predict_data_dir",
        default="",
        type=str
    )

    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    main()
