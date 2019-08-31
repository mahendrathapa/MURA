import torch
import torch.nn as nn

from constants import Constants


class DenseLayer(nn.module):
    def __init__(self, in_channel, growth_rate, bn_size, drop_rate):
        super().__init__

        self.norm1 = nn.BatchNorm2D(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2D(
            in_channel,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
            )
        self.conv2 = nn.Conv2D(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self._drop_rate = drop_rate

    def forward(self, x):
        normalized = self.relu(self.norm1(x))
        bottleneck = self.conv1(normalized)
        output = self.conv2(bottleneck)

        if self._drop_rate > 0:
            output = nn.functional.dropout(
                output, p=self._drop_rate, training=self.training
            )

        return torch.cat([x, output], 1)


class DenseBlock(nn.module):
    def __init__(self, in_channel, num_layers, growth_rate, bn_size=4,
                 drop_rate=0.0):
        super().__init__
        layers = []

        for i in range(num_layers):
            layers.append(DenseLayer(
                in_channel + i * growth_rate, growth_rate, bn_size, drop_rate
            ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TransitionBlock(nn.Module):
    def __init__(self, in_channel, compression_rate=1.0, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2D(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2D(
            in_channel,
            compression_rate * in_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self._drop_rate = drop_rate

    def forward(self, x):
        normalized = self.relu(self.norm1(x))
        output = self.conv1(normalized)
        if self._drop_rate > 0:
            output = nn.functional.dropout(
                output, p=self._drop_rate, training=self.training
            )
        return nn.AvgPool2D(output, kernel_size=2, stride=2)


class Densenet(nn.module):
    def __init__(self, growth_rate=32, compression_rate=0.5, in_channel=64,
                 bn_size=4, drop_rate=0, num_class=1):

        super().__init__()

        block_config = Constants.BLOCK_CONFIG

        # Initial convolution
        self.conv1 = nn.Conv2(
            in_channel,
            2 * growth_rate,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Dense block -> TransitionBlock
        in_channel = 2 * growth_rate

        # Block 1
        self.block1 = DenseBlock(
            in_channel, block_config[0], growth_rate, bn_size, drop_rate
        )
        in_channel += growth_rate * block_config[0]
        self.trans1 = TransitionBlock(in_channel, compression_rate, drop_rate)
        in_channel *= compression_rate

        # Block 1
        self.block2 = DenseBlock(
            in_channel, block_config[1], growth_rate, bn_size, drop_rate
        )
        in_channel += growth_rate * block_config[1]
        self.trans2 = TransitionBlock(in_channel, compression_rate, drop_rate)
        in_channel *= compression_rate

        # Block 1
        self.block3 = DenseBlock(
            in_channel, block_config[2], growth_rate, bn_size, drop_rate
        )
        in_channel += growth_rate * block_config[2]
        self.trans3 = TransitionBlock(in_channel, compression_rate, drop_rate)
        in_channel *= compression_rate

        # Block 4
        self.block4 = DenseBlock(
            in_channel, block_config[3], growth_rate, bn_size, drop_rate
        )
        in_channel += growth_rate * block_config[3]

        # Block Final
        self.norm1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_channel, num_class)
        self.channels = in_channel

        # Official init from torch repo.
        # ----------Not clear about this part------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.trans1(self.block1(out1))
        out3 = self.trans2(self.block2(out2))
        out4 = self.trans3(self.block3(out3))
        out5 = self.block4(out4)
        out6 = self.relu(self.norm1(out5))
        out7 = nn.AvgPool2D(out6, stride=1, kernel_size=7)
        out = out7.view(-1, self.channels)
        return self.fc(out)







