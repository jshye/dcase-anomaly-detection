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

from torch import nn


class AutoEncoder(nn.Module):
    """
    AutoEncoder
    """

    def __init__(self, x_dim, h_dim, z_dim, n_hidden):
        super(AutoEncoder, self).__init__()

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
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        """
        Reconstruct inputs through AutoEncoder.
        """
        hidden = self.activation(self.bnorms[0](self.layers[0](inputs)))
        for i in range(2 * (self.n_hidden + 1)):  # i : 0 to 2 * self.n_hidden + 1
            hidden = self.activation(self.bnorms[i + 1](self.layers[i + 1](hidden)))
        output = self.layers[-1](hidden)

        return output

    def get_loss(self, inputs):
        """
        Calculate loss function of AutoEncoder.
        """
        recon_x = self.forward(inputs)
        recon_loss = self.criterion(recon_x, inputs)

        return recon_loss
