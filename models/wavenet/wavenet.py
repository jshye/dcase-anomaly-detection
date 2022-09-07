import torch
from torch import nn


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = self.causal_conv(self.in_channels, self.out_channels, self.kernel_size, self.dilation)
        self.padding = self.conv1.padding[0]

    def causal_conv(self, in_channels, out_channels, kernel_size, dilation):
        pad = (kernel_size - 1) * dilation
        return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = x[:, :, :-self.padding]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_channel, n_mul, kernel_size, dilation_rate):
        super(ResidualBlock, self).__init__()
        self.n_channel = n_channel
        self.n_mul = n_mul
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_filter = self.n_channel * self.n_mul

        self.sigmoid_group_norm = nn.GroupNorm(1, self.n_filter)
        self.sigmoid_conv = CausalConv1d(self.n_filter, self.n_filter, self.kernel_size, self.dilation_rate)
        self.tanh_group_norm = nn.GroupNorm(1, self.n_filter)
        self.tanh_conv = CausalConv1d(self.n_filter, self.n_filter, self.kernel_size, self.dilation_rate)

        self.skip_group_norm = nn.GroupNorm(1, self.n_filter).to(DEVICE)
        self.skip_conv = nn.Conv1d(self.n_filter, self.n_channel, 1)
        self.residual_group_norm = nn.GroupNorm(1, self.n_filter)
        self.residual_conv = nn.Conv1d(self.n_filter, self.n_filter, 1)

    def forward(self, x):
        x1 = self.sigmoid_group_norm(x)
        x1 = self.sigmoid_conv(x1)
        x2 = self.tanh_group_norm(x)
        x2 = self.tanh_conv(x2)
        x1 = nn.Sigmoid()(x1)
        x2 = nn.Tanh()(x2)
        x = x1 * x2
        x1 = self.skip_group_norm(x)
        skip = self.skip_conv(x1)
        x2 = self.residual_group_norm(x)
        residual = self.residual_conv(x2)
        return skip, residual


class WaveNet(nn.Module):
    def __init__(self, config):
        super(WaveNet, self).__init__()
        self.n_channel = config['model']['n_channel']
        self.n_mul = config['model']['n_mul']
        self.kernel_size = config['model']['kernel_size']

        self.n_filter = self.n_channel * self.n_mul
        self.group_norm1 = nn.GroupNorm(1, self.n_channel)
        self.conv1 = nn.Conv1d(self.n_channel, self.n_filter, 1)

        self.block1 = ResidualBlock(self.n_channel, self.n_mul, self.kernel_size, 1)
        self.block2 = ResidualBlock(self.n_channel, self.n_mul, self.kernel_size, 2)
        self.block3 = ResidualBlock(self.n_channel, self.n_mul, self.kernel_size, 4)

        self.relu1 = nn.ReLU()
        self.group_norm2 = nn.GroupNorm(1, self.n_channel)
        self.conv2 = nn.Conv1d(self.n_channel, self.n_channel, 1)
        self.relu2 = nn.ReLU()
        self.group_norm3 = nn.GroupNorm(1, self.n_channel)
        self.conv3 = nn.Conv1d(self.n_channel, self.n_channel, 1)

    def forward(self, x):
        # x = self.group_norm1(x)
        x = self.group_norm1(torch.squeeze(x, 1))
        x = self.conv1(x)
        skip1, x = self.block1(x)
        skip2, x = self.block2(x)
        skip3, x = self.block3(x)
        skip = skip1 + skip2 + skip3
        x = self.relu1(skip)
        x = self.group_norm2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.group_norm3(x)
        x = self.conv3(x)
        # output = x[:, :, self.get_receptive_field() - 1:-1]
        output = torch.unsqueeze(x, 1)  ##
        return output

    def get_receptive_field(self):
        receptive_field = 1
        for _ in range(3):
            receptive_field = receptive_field * 2 + self.kernel_size - 2
        return receptive_field