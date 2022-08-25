"""
Implementation of Liu2022 (DCASE 2022 rank 1)
    classification-based
    MobileNetV2
"""

from torch import nn
import torch.nn.functional as F


def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            # nn.ReLU6(inplace=True),
            nn.PReLU(),
        )
    )

def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU6(inplace=True)
            nn.PReLU()
        )
    )

def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            # nn.ReLU6(inplace=True)
            nn.PReLU()
        )
    )

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            #dw
            dwise_conv(hidden_dim, stride=stride),
            #pw
            conv1x1(hidden_dim, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)


class Liu2022(nn.Module):
    def __init__(self, config):
        super(Liu2022, self).__init__()
        self.ch_in = 1
        self.p_dropout = 0.2
        self.n_class = config['model']['n_class']

        self.bottleneck_configs = [
            [2, 128, 2, 2],  # t, c, n, s
            [4, 128, 2, 2],  # expand_ratio, out channel, number of inverted blocks, stride
            [4, 128, 2, 2],
        ]

        self.conv2d_1 = conv3x3(self.ch_in, 64, stride=2)
        self.conv2d_2 = conv3x3(64, 64, stride=1)

        bottlenecks = []
        input_channel = 64
        for t, c, n, s in self.bottleneck_configs:
            for i in range(n):
                bottlenecks.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=s))
                input_channel = c
        self.bottlenecks = nn.Sequential(*bottlenecks)

        self.conv2d_3 = conv1x1(c, 512)
        # self.linear_gdconv2d = nn.Conv2d(512, 512, kernel_size=7, padding=0, stride=1)  # kernel size == input size ??
        self.linear_gdconv2d = nn.Conv2d(512, 512, kernel_size=1, padding=0, stride=1)  # kernel size == input size ??
        self.linear_conv2d = nn.Conv2d(512, 128, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(p=self.p_dropout)
        self.flatten = nn.Flatten()
        # self.linear = nn.Linear(128, self.n_class)
        self.linear = nn.Linear(384, self.n_class)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.bottlenecks(x)
        x = self.conv2d_3(x)
        x = self.linear_gdconv2d(x)
        x = self.linear_conv2d(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x