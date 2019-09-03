import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn.functional as functional

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from torchsummary import summary
from tensorboardX import SummaryWriter

from src.constants import Constants
from src.utils.data_utils import get_image


class DenseNetModel:
    def __init__(self, network, config,
                 train_data=None, val_data=None, test_data=None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.network = network.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.config = config
        self.model_config = self.config.model_config

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.model_config.learning_rate,
            betas=(0.9, 0.999)
        )

        self.loss_function = functional.binary_cross_entropy_with_logits

        self.start_epoch = 1
        self.base_out_dir = (
            Path(self.model_config.OUTPUT_ROOT_PATH) / self.config.run_id
        ).mkdir(exist_ok=True, parents=True)
        self.base_out_dir = (
            Path(self.model_config.OUTPUT_ROOT_PATH) / self.config.run_id
        )

        self.model_out_dir = (
            Path(self.base_out_dir) / "checkpoints"
        ).mkdir(parents=True, exist_ok=True)
        self.model_out_dir = (
            Path(self.base_out_dir) / "checkpoints"
        )

        self.config_out_dir = (
            Path(self.base_out_dir) / "config"
        ).mkdir(parents=True, exist_ok=True)
        self.config_out_dir = (
            Path(self.base_out_dir) / "config"
        )

        self.results = defaultdict(list)
        self.best_result = defaultdict(list)

        if self.config.model_checkpoint:
            self.load_model()

    def load_model(self):
        checkpoint_path = (
            Path(self.model_out_dir) / self.config.model_checkpoint
        )

        print("Loading checkpoint {}".format(checkpoint_path))

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.network.load_state_dict(checkpoint["network_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
        self.results = checkpoint["results"]

        print("Loaded checkpoint {} epoch {}".format(
            checkpoint_path, checkpoint["epoch"])
        )

    def save_model(self, epoch, tag=None):
        print("Model saved for epoch {}".format(epoch))

        if tag is not None:
            model_name = tag + str(epoch) + "_model.pth"
        else:
            model_name = str(epoch) + "_model.pth"

        state = {
            "epoch": epoch,
            "network_dict": self.network.state_dict(),
            "optimizer_dict": self.optimizer.state_dict(),
            "results": self.results,
        }

        torch.save(
            state,
            Path(self.model_out_dir) / model_name
        )

    def train(self):
        img_size = Constants.IMAGE_SIZE
        summary(self.network, (1, img_size, img_size))
        writer = SummaryWriter(
            Path(self.base_out_dir) / "tensorboard"
        )

        print(f"Run ID : {self.config.run_id}")
        print("Training started at: {}".format(
            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        ))

        self.best_result["train_loss"] = np.inf
        self.best_result["val_loss"] = np.inf

        start = self.start_epoch
        end = start + self.model_config.epoch

        for epoch in range(start, end):
            self.network = self.network.train()

            for batch_set in self.train_data:
                inputs = torch.cat((batch_set["positive_x"].to(self.device), batch_set["negative_x"].to(self.device)))
                target = torch.cat((batch_set["positive_y"].to(self.device), batch_set["negative_y"].to(self.device)))

                self.optimizer.zero_grad()
                predictions = self.network(inputs)
                loss = self.loss_function(predictions, target)
                loss.backward()
                self.optimizer.step()

            train_loss = self.test(self.train_data, ds_type="train_set")
            self.results["epochs"].append(epoch)
            self.results["train_loss"].append(train_loss)

            print("{} Epoch: {}, Train loss: {:.4f}".format(
                time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
                epoch,
                train_loss
            ))
            writer.add_scalars("Loss", {"train_loss": train_loss}, epoch)

            if self.val_data:
                val_loss = self.test(self.val_data)
                self.results["val_loss"].append(val_loss)

                print("{} Epoch: {}, Val loss: {:.4f}".format(
                    time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
                    epoch,
                    val_loss
                ))
                writer.add_scalars("Loss", {"val_loss": val_loss}, epoch)

            # Now plotting
            fig = plt.figure()
            plt.plot(
                self.results["epochs"],
                self.results["train_loss"],
                label="train"
            )
            if self.val_data:
                plt.plot(
                    self.results["epochs"],
                    self.results["val_loss"],
                    label="val"
                )
            plt.title("Loss Progression")
            plt.xlabel("epochs")
            plt.ylabel("Loss")
            plt.legend(loc="upper right")

            writer.add_figure("Loss", fig)
            fig.savefig(Path(self.base_out_dir) / "loss.png")
            plt.show()
            plt.close(fig)

            if (self.results["train_loss"][-1] <=
                self.best_result["train_loss"] and
                self.results["val_loss"][-1] < self.best_result["val_loss"]):

                self.best_result["train_loss"] = self.results["train_loss"][-1]
                self.best_result["val_loss"] = self.results["val_loss"][-1]
                self.best_result["state"] = {
                    "epoch": epoch,
                    "network_dict": deepcopy(self.network.state_dict()),
                    "optimizer_dict": deepcopy(self.optimizer.state_dict()),
                    "results": deepcopy(self.results)
                }

            if epoch % self.model_config.model_dump_gap == 0:
                self.save_model(epoch)

        print("Best results: Epoch{}, Train Loss: {}, Val Loss: {}".format(
            self.best_result["epoch"],
            self.best_result["train_loss"],
            self.best_result["val_loss"]
        ))

        model_save_dir = "best_model_{}.pth".format(
            str(self.best_result["state"]["epoch"])
        )

        torch.save(
            self.best_result["state"],
            Path(self.model_out_dir) / model_save_dir
        )

        if self.test_data:
            test_loss = self.test(self.test_data)
            print("{} Epoch: {}, Test loss: {.4f}".format(
                time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
                epoch,
                test_loss
            ))

    def test(self, data_set, ds_type=None):
        self.network = self.network.eval()
        running_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for batch_set in data_set:
                if ds_type == "train_set":
                    inputs = torch.cat((batch_set["positive_x"].to(self.device), batch_set["negative_x"].to(self.device)))
                    target = torch.cat((batch_set["positive_y"].to(self.device), batch_set["negative_y"].to(self.device)))
                else:
                    inputs = batch_set["x"].to(self.device)
                    target = batch_set["y"].to(self.device)
                predictions = self.network(inputs)
                loss = self.loss_function(predictions, target)
                running_loss += loss.item()
                total_count += 1

        return (running_loss / total_count)

    def predict(self, image_path):
        self.network = self.network.eval()
        image = get_image(image_path, unsqueeze_dim=2)

        with torch.no_grad():
            image = image.to(self.device)
            predictions = self.network(image)

            label = torch.sigmoid(predictions)

            return image.squeeze(0).squeeze(0).cpu().numpy(), label.item()
