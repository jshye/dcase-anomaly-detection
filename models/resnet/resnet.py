import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, projection=False):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.projection = projection
        if self.projection:
            self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=2, padding=(1, 1))
        else:
            self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(self.out_channel)
        if self.projection:
            self.downsample = nn.Conv2d(self.in_channel, self.out_channel, stride=2, kernel_size=1)
        else:
            self.downsample = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.projection:
            skip = self.downsample(x)
        else:
            skip = x
        out += skip
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()
        self.n_channel = 8
        self.n_class = config['model']['n_class']
        self.conv1 = nn.Conv2d(1, self.n_channel, kernel_size=7, stride=2, padding=(3, 2))
        self.bn1 = nn.BatchNorm2d(self.n_channel)
        self.relu = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))
        self.block1 = ResidualBlock(self.n_channel, self.n_channel)
        self.block2 = ResidualBlock(self.n_channel, self.n_channel)
        self.block3 = ResidualBlock(self.n_channel, self.n_channel * 2, True)
        self.block4 = ResidualBlock(self.n_channel * 2, self.n_channel * 2)
        self.block5 = ResidualBlock(self.n_channel * 2, self.n_channel * 4, True)
        self.block6 = ResidualBlock(self.n_channel * 4, self.n_channel * 4)
        self.block7 = ResidualBlock(self.n_channel * 4, self.n_channel * 8, True)
        self.block8 = ResidualBlock(self.n_channel * 8, self.n_channel * 8)
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.n_channel * 8, self.n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.gap1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x