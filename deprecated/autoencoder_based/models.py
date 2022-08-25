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


class AEBaseline(nn.Module):
    """
    hidden_dim: 128
    latent_dim: 8
    n_hidden: 3
    """
    def __init__(self, x_dim, h_dim, z_dim, n_hidden):
        super(AEBaseline, self).__init__()

        self.n_hidden = n_hidden  # number of hidden layers

        layers = nn.ModuleList([])

        layers += [nn.Linear(x_dim, h_dim)]
        layers += [nn.Linear(h_dim, h_dim) for _ in range(self.n_hidden)]
        layers += [nn.Linear(h_dim, z_dim)]
        layers += [nn.Linear(z_dim, h_dim)]
        layers += [nn.Linear(h_dim, h_dim) for _ in range(self.n_hidden)]
        layers += [nn.Linear(h_dim, x_dim)]
        self.layers = nn.ModuleList(layers)

        bnorms = [nn.BatchNorm1d(h_dim) for _ in range(self.n_hidden + 1)]
        bnorms += [nn.BatchNorm1d(z_dim)]
        bnorms += [nn.BatchNorm1d(h_dim) for _ in range(self.n_hidden + 1)]
        self.bnorms = nn.ModuleList(bnorms)

        self.activation = nn.ReLU()
        # self.criterion = nn.MSELoss()

    def forward(self, inputs):
        """
        Reconstruct inputs through AutoEncoder.
        """
        hidden = self.activation(self.bnorms[0](self.layers[0](inputs.squeeze())))
        for i in range(2 * (self.n_hidden + 1)):  # i : 0 to 2 * self.n_hidden + 1
            hidden = self.activation(self.bnorms[i + 1](self.layers[i + 1](hidden)))
        output = self.layers[-1](hidden)

        return output

    # def get_loss(self, inputs):
    #     """
    #     Calculate loss function of AutoEncoder.
    #     """
    #     recon_x = self.forward(inputs)
    #     recon_loss = self.criterion(recon_x, inputs)

    #     return recon_loss
