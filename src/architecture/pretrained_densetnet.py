import torch
import torchvision
import torch.nn as nn


class PretrainedDensenet(nn.Module):
    def __init__(self, num_class=1):
        super().__init__()
        self.channels = 1664
        densenet_169 = torchvision.models.densenet169(pretrained=True)
        self.features = nn.Sequential(*list(densenet_169.features.children()))
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.channels, num_class)

    def forward(self, x):
        features = self.features(x)
        out = self.relu(features)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, self.channels)
        return self.fc1(out)
