"""
Class definition of AutoEncoder in PyTorch.

Copyright (C) 2021 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F


class Lopez2020Encoder(nn.Module):
    def __init__(self):
        super(Lopez2020Encoder, self).__init__()

        self.conv2d_1 = nn.Conv2d(1, 64, 7, stride=1)
        self.bn2d_1 = nn.BatchNorm2d(64)

        self.conv2d_2 = nn.Conv2d(64, 32, 5, stride=1)
        self.conv2d_3 = nn.Conv2d(32, 6, 3, stride=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, input):
        out = F.relu(self.bn2d_1(self.conv2d_1(input)))
        out = F.relu(self.conv2d_2(out))
        out = self.conv2d_3(out)
        out = F.relu(self.max_pool(out))
        return out



class Lopez2020Xvector(nn.Module):
    def __init__(self):
        super(Lopez2020Xvector, self).__init__()

        self.flatten = nn.Flatten()

        self.dense_1 = nn.Linear(348, 348, 3)
        self.bn1d_1 = nn.BatchNorm1d(348)
        
        self.dense_2 = nn.Linear(348, 348, 3)
        self.bn1d_2 = nn.BatchNorm1d(348)
        
        self.dense_3 = nn.Linear(348, 348, 3)
        self.bn1d_3 = nn.BatchNorm1d(348)
        
        self.dense_4 = nn.Linear(348, 1500, 1)
        self.bn1d_4 = nn.BatchNorm1d(1500)

        self.fc_1 = nn.Linear(1500, 128)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input):
        input = self.flatten(input)
        out = F.relu(self.bn1d_1(self.dense_1(input)))
        out = F.relu(self.bn1d_2(self.dense_2(out)))
        out = F.relu(self.bn1d_3(self.dense_3(out)))
        out = self.bn1d_4(self.dense_4(out))        
        out = self.dropout(self.fc_1(out))
        return out


class Lopez2020(nn.Module):
    """
    https://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Lopez_90_t2.pdf
    """
    def __init__(self, num_sections, n_frames, n_mels):
        super(Lopez2020, self).__init__()
        self.encoder = Lopez2020Encoder()
        self.stats_pool = Lopez2020Xvector()
        self.fc = nn.Linear(128, num_sections)  # replace with AMS(added margin softmax)
        self.n_frames = n_frames
        self.n_mels = n_mels

    def forward(self, input):
        out = self.encoder(input)
        out = self.stats_pool(out)
        out = self.fc(out)
        return out

class XVector(nn.Module):
    """
    ref) https://github.com/manojpamk/pytorch_xvectors/blob/master/models.py
    """
    def __init__(self, in_channels, numSpkrs, p_dropout):
        super(XVector, self).__init__()

        self.tdnn_1 = nn.Conv1d(in_channels, 348, kernel_size=3, stride=1, dilation=2)
        self.bn_tdnn_1 = nn.BatchNorm1d(348)
        self.dropout_tdnn_1 = nn.Dropout(p=p_dropout)

        self.tdnn_2 = nn.Conv1d(348, 348, kernel_size=3, stride=1, dilation=3)
        self.bn_tdnn_2 = nn.BatchNorm1d(348)
        self.dropout_tdnn_2 = nn.Dropout(p=p_dropout)

        self.tdnn_3 = nn.Conv1d(348, 348, kernel_size=1, stride=1, dilation=1)
        self.bn_tdnn_3 = nn.BatchNorm1d(348)
        self.dropout_tdnn_3 = nn.Dropout(p=p_dropout)

        self.tdnn_4 = nn.Conv1d(348, 1500, kernel_size=1, stride=1, dilation=1)
        self.bn_tdnn_4 = nn.BatchNorm1d(1500)
        self.dropout_tdnn_4 = nn.Dropout(p=p_dropout)

        self.fc_1 = nn.Linear(3000, 128)
        self.bn_fc_1 = nn.BatchNorm1d(128)
        self.dropout_fc_1 = nn.Dropout(p=p_dropout)

        self.fc_2 = nn.Linear(128, numSpkrs)

    # def forward(self, x, eps):
    def forward(self, x):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        x = self.dropout_tdnn_1(self.bn_tdnn_1(F.relu(self.tdnn_1(x))))
        x = self.dropout_tdnn_2(self.bn_tdnn_2(F.relu(self.tdnn_2(x))))
        x = self.dropout_tdnn_3(self.bn_tdnn_3(F.relu(self.tdnn_3(x))))
        x = self.dropout_tdnn_4(self.bn_tdnn_4(self.tdnn_4(x)))

        # if self.training:
        #     x = x + torch.randn(x.size()).cuda()*eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.dropout_fc_1(self.bn_fc_1(F.relu(self.fc_1(stats))))
        x = self.fc_2(x)
        return x


class Lopez2021Encoder(nn.Module):
    def __init__(self):
        super(Lopez2021Encoder, self).__init__()

        self.conv1d_1 = nn.Conv1d(1, 64, 7, stride=1)
        self.bn_1 = nn.BatchNorm1d(64)

        self.conv1d_2 = nn.Conv1d(64, 32, 5, stride=1)
        self.conv1d_3 = nn.Conv1d(32, 6, 3, stride=1)
        self.max_pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        out = F.relu(self.bn_1(self.conv1d_1(x)))
        out = F.relu(self.conv1d_2(out))
        out = self.conv1d_3(out)
        out = F.relu(self.max_pool(out))
        return out


class Lopez2021(nn.Module):
    """ 
    precit the section ID meta-data parameter using the categorical cross entropy loss fn
    """
    def __init__(self):
        super(Lopez2021, self).__init__()

        self.encoder = Lopez2021Encoder()
        self.xvector = XVector(in_channels=1, numSpkrs=3, p_dropout=0.1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0), 1, -1)
        x = self.xvector(x)
        return x


############ Liu 2022 #############
"""
references)
https://github.com/jmjeon94/MobileNet-Pytorch/blob/master/MobileNetV2.py
"""

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
    def __init__(self, ch_in=1, n_classes=3, p_dropout=0.2):
        super(Liu2022, self).__init__()

        self.configs = [
            [2, 128, 2, 2],  # t, c, n, s
            [4, 128, 2, 2],
            [4, 128, 2, 2],
        ]

        # self.conv2d_1 = nn.Conv2d(ch_in, 64, 3, stride=2)
        # self.conv2d_2 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv2d_1 = conv3x3(ch_in, 64, stride=2)
        self.conv2d_2 = conv3x3(64, 64, stride=1)

        bottlenecks = []
        input_channel = 64
        for t, c, n, s in self.configs:
            for i in range(n):
                bottlenecks.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=s))
                input_channel = c
        self.bottlenecks = nn.Sequential(*bottlenecks)

        self.conv2d_3 = conv1x1(c, 512)
        # self.linear_gdconv2d = nn.Conv2d(512, 512, kernel_size=7, padding=0, stride=1)  # kernel size == input size ??
        self.linear_gdconv2d = nn.Conv2d(512, 512, kernel_size=1, padding=0, stride=1)  # kernel size == input size ??
        self.linear_conv2d = nn.Conv2d(512, 128, kernel_size=1, stride=1)

        self.dropout = nn.Dropout(p=p_dropout)
        self.flatten = nn.Flatten()
        # self.linear = nn.Linear(128, n_classes)
        self.linear = nn.Linear(384, n_classes)

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