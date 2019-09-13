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
from src.utils.evaluate_utils import kappa_cohen
from src.utils.plot_utils import generate_plot


class DenseNetModel:
    def __init__(self, network, config,
                 train_data=None, val_data=None, test_data=None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Device in use: {self.device}")
        if torch.cuda.device_count() > 1:
            print("GPUs in use:  ", torch.cuda.device_count())
            network = torch.nn.DataParallel(network)

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

        self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=Constants.LR_DECAY["no_of_epochs"],
                gamma=Constants.LR_DECAY["factor"]
        )

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

        if tag is not None:
            model_name = f"epoch_{epoch}_iter_{tag}_model.pth"
        else:
            model_name = f"{epoch}_model.pth"

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
        print(f"Model saved as {model_name}")

    def train(self):
        img_size = Constants.IMAGE_SIZE
        if not Constants.PRETRAINED:
            summary(self.network, (1, img_size, img_size))
        else:
            summary(self.network, (3, img_size, img_size))
        writer = SummaryWriter(
            Path(self.base_out_dir) / "tensorboard"
        )

        total_data = (len(self.train_data) * 2 * self.model_config.TRAIN_BATCH_SIZE) / self.model_config.SAMPLING_RATIO
        print(f"Run ID : {self.config.run_id}")
        print("Training started at: {}".format(
            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        ))

        self.best_result["train_kappa"] = np.inf
        self.best_result["val_kappa"] = np.inf

        start = self.start_epoch
        end = start + self.model_config.epoch

        for epoch in range(start, end):
            self.network = self.network.train()

            for iteration, batch_set in enumerate(self.train_data):
                inputs = torch.cat((batch_set["positive_x"].to(self.device), batch_set["negative_x"].to(self.device)))
                target = torch.cat((batch_set["positive_y"].to(self.device), batch_set["negative_y"].to(self.device)))
                self.optimizer.zero_grad()
                predictions = self.network(inputs)
                loss = self.loss_function(predictions, target)
                loss.backward()
                self.optimizer.step()

                if iteration % self.model_config.model_dump_gap == 0 \
                        and iteration != 0:
                    train_loss, train_kappa = self.test(self.train_data, ds_type="train_set")
                    self.results["epochs"].append(epoch + iteration / len(self.train_data))
                    self.results["train_loss"].append(train_loss)
                    self.results["train_kappa"].append(train_kappa)

                    print("lr: {:.2E}".format(self.optimizer.param_groups[0]['lr']))
                    print("{} Epoch: {}, Iteration: {}, Train loss: {:.4f}, Train Cohen Kappa Score: {:.4f}".format(
                        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
                        epoch,
                        iteration,
                        train_loss,
                        train_kappa
                    ))
                    writer.add_scalars("Loss", {"train_loss": train_loss}, epoch)

                    if self.val_data:
                        val_loss, val_kappa = self.test(self.val_data)
                        self.results["val_loss"].append(val_loss)
                        self.results["val_kappa"].append(val_kappa)

                        print("{} Epoch: {}, Iteration: {}, Val loss: {:.4f}, Val Cohen Kappa Score: {:.4f}".format(
                            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
                            epoch,
                            iteration,
                            val_loss,
                            val_kappa
                        ))
                        writer.add_scalars("Loss", {"val_loss": val_loss}, epoch)

                    # Now plotting
                    loss_fig = generate_plot(
                            self.results["epochs"],
                            self.results["train_loss"],
                            self.results["val_loss"],
                            title="Loss Progression",
                            y_label="loss"
                    )
                    writer.add_figure("Loss", loss_fig)
                    loss_fig.savefig(Path(self.base_out_dir) / "loss.png")
                    plt.show()
                    plt.close(loss_fig)

                    kappa_score_fig = generate_plot(
                            self.results["epochs"],
                            self.results["train_kappa"],
                            self.results["val_kappa"],
                            title="Cohen Kappa Score",
                            y_label="CKS"
                    )
                    writer.add_figure("Cohen Kappa Score", kappa_score_fig)
                    loss_fig.savefig(Path(self.base_out_dir) / "kappa_score.png")
                    plt.show()
                    plt.close(kappa_score_fig)

                    if (self.results["train_kappa"][-1] <=
                            self.best_result["train_kappa"] and
                            self.results["val_kappa"][-1] < self.best_result["val_kappa"]):

                        self.best_result["train_kappa"] = self.results["train_kappa"][-1]
                        self.best_result["val_kappa"] = self.results["val_kappa"][-1]
                        self.best_result["state"] = {
                            "epoch": epoch,
                            "network_dict": deepcopy(self.network.state_dict()),
                            "optimizer_dict": deepcopy(self.optimizer.state_dict()),
                            "results": deepcopy(self.results)
                        }
                    self.save_model(epoch, tag=iteration)

            self.scheduler.step()

        print("Best results: Epoch{}, Train Cohen Kappa Score: {}, Val Cohen Kappa Score: {}".format(
            self.best_result["epoch"],
            self.best_result["train_kappa"],
            self.best_result["val_kappa"]
        ))

        model_save_dir = "best_model_{}.pth".format(
            str(self.best_result["state"]["epoch"])
        )

        torch.save(
            self.best_result["state"],
            Path(self.model_out_dir) / model_save_dir
        )

        if self.test_data:
            test_loss, test_kappa = self.test(self.test_data)
            print("{} Epoch: {}, Test loss: {:.4f}, Test Cohen Kappa Score: {:.4f}".format(
                time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
                epoch,
                test_loss,
                test_kappa
            ))

    def test(self, data_set, ds_type=None):
        self.network = self.network.eval()
        running_loss = 0.0
        running_kappa = 0.0
        total_count = 0
        target_list = list()
        predictions_list = list()

        with torch.no_grad():
            for batch_set in data_set:
                if ds_type == "train_set":
                    inputs = torch.cat((batch_set["positive_x"].to(self.device), batch_set["negative_x"].to(self.device)))
                    target = torch.cat((batch_set["positive_y"].to(self.device), batch_set["negative_y"].to(self.device)))
                else:
                    inputs = batch_set["x"].to(self.device)
                    target = batch_set["y"].to(self.device)

                predictions = self.network(inputs)

                target_list.append(target)
                predictions_list.append(torch.sigmoid(predictions))

                loss = self.loss_function(predictions, target)

                running_loss += loss.item()
                total_count += 1

            running_kappa = kappa_cohen(
                    torch.cat(target_list),
                    torch.cat(predictions_list)
            )

        return (running_loss / total_count), running_kappa

    def predict(self, image_path):
        self.network = self.network.eval()
        image = get_image(image_path, unsqueeze_dim=2)

        with torch.no_grad():
            image = image.to(self.device)
            predictions = self.network(image)

            label = torch.sigmoid(predictions)

            return image.squeeze(0).squeeze(0).cpu().numpy(), label.item()
