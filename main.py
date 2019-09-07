import argparse
import json
import time
import torch

import pandas as pd

from pathlib import Path
from collections import defaultdict
from PIL import Image

from src.architecture.densenet import Densenet
from src.config.config import LocalConfig, ServerConfig
from src.data.mura_dataset_loader import MuraDataSetLoader
from src.model.DenseNet_model import DenseNetModel
from src.constants import Constants

torch.manual_seed(123456789)


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
    print(f"Sampling Ratio: {model_config.SAMPLING_RATIO}")
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

    elif global_config.mode == "predict":
        results = defaultdict(list)
        model = DenseNetModel(
                network, global_config
        )

        predictions_path = (Path(model_config.OUTPUT_ROOT_PATH) / global_config.run_id / "predictions")
        predictions_path.mkdir(exist_ok=True, parents=True)

        image_list = list()
        image_path = Path(global_config.predict_data_dir)
        for data_types in model_config.ACCEPTED_DATA_TYPES:
            image_list.extend(image_path.glob(data_types))

        for index, i in enumerate(image_list):
            input_img_name = i.name
            if not any(x in input_img_name for x in model_config.IGNORE_FILES):
                image, label = model.predict(i)
                print(input_img_name, label)
                results['image'].append(input_img_name)
                results['label'].append(label)
                results['index'].append(index)
                img_save_path = predictions_path / input_img_name
                Image.fromarray(image, mode='L').save(img_save_path)
        csv_save_path = predictions_path / "predictions.csv"
        pd.DataFrame(results, index=results['index']).drop(columns='index').\
            to_csv(csv_save_path, index=False)


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
        choices=["full", "train", "predict"],
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
